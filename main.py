from policy_nets import GATModel
from ray.rllib.models import ModelCatalog
from ray.tune.registry import get_trainable_cls
from ray.tune.logger import pretty_print
import os
from env import RLSearchEnv
from globals import *
import ray
import time

os.environ["TUNE_RESULT_DIR"] = './ray_results'


ray.init()
ModelCatalog.register_custom_model("GATModel", GATModel)

RLSearchEnv_config = {
    'max_variables': MAX_VARIABLES,
    'max_statements': MAX_STATEMENTS,
    'max_steps': MAX_STEPS,
    'init_program_length': INIT_PROGRAM_LENGTH
}

train_batch_size = 4096
sgd_minibatch_size = 512
num_sgd_iter = 5
lr_schedule = [
    [0, 1e-6],
    [train_batch_size * 20, 1e-4],
    [train_batch_size * 100, 0]
]

config = (
    get_trainable_cls('PPO')
    .get_default_config()
    .rl_module(_enable_rl_module_api=False)
    .training(
        model={
            "custom_model": "GATModel",
        },
        train_batch_size=train_batch_size,
        sgd_minibatch_size=sgd_minibatch_size,
        num_sgd_iter=num_sgd_iter,
        grad_clip_by='global_norm',
        grad_clip=None,
        lr=0,
        lr_schedule=lr_schedule,
        vf_clip_param=1000,
        vf_loss_coeff=1e-5,
        _enable_learner_api=False
    )
    .environment(RLSearchEnv, env_config=RLSearchEnv_config)
    .framework("torch")
    .rollouts(num_rollout_workers=32)
    .resources(num_gpus=2, num_gpus_per_worker=1/16, num_cpus_per_worker=2)
)

stop = {
    "training_iteration": 100,
    "timesteps_total": 10000000,
}

algo = config.build()
config = algo.get_config()

for i in range(stop["training_iteration"]):
    print("Training iteration:", i)
    result = algo.train()
    print(pretty_print(result))
    if result["timesteps_total"] >= stop["timesteps_total"]:
        break

model_name = config.model['custom_model']
learning_rate = config['lr']
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
checkpoint_path = f'./checkpoint_{model_name}_lr{learning_rate}_{current_time}'
algo.save(checkpoint_path)
print("Checkpoint has been created inside directory:", checkpoint_path)

algo.stop()
ray.shutdown()
