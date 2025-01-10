import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.algorithm import Algorithm
from policy_nets import GATModel
from env import RLSearchEnv
from update_rule import *
ray.init(local_mode=0)
ModelCatalog.register_custom_model("GATModel", GATModel)
recovered_algo = Algorithm.from_checkpoint('checkpoint_GATModel_lr0_2025-01-09_20-03-33')

env_config = {
    'max_variables': MAX_VARIABLES,
    'max_statements': MAX_STATEMENTS,
    'max_steps': MAX_STEPS,
    'init_program_length': INIT_PROGRAM_LENGTH
}
env = RLSearchEnv(env_config)

adamw_text = '''
v_5 = torch.square(grad)
v_6 = torch.mul(0.1, grad)
v_7 = torch.mul(0.9, v1)
v1 = torch.add(v_6, v_7)
v_8 = torch.mul(0.001, v_5)
v_9 = torch.mul(0.999, v2)
v2 = torch.add(v_8, v_9)
v_10 = torch.sqrt(v2)
v_10 = torch.add(v_10, 0.001)
update = torch.true_divide(v1, v_10)
update = torch.mul(update, 0.1)
'''
update_rule = text_to_update_rule(adamw_text)

lion_text = '''
v_6 = torch.mul(0.1, grad)
v_7 = torch.mul(0.9, v1)
v_5 = torch.add(v_6, v_7)
update = torch.sign(v_5)
v_6 = torch.mul(0.01, grad)
v_7 = torch.mul(0.99, v1)
v1 = torch.add(v_6, v_7)
update = torch.mul(update, 0.01)
'''

update_rule = text_to_update_rule(lion_text)

episode_reward, max_reward = 0, 0
env.reset()
env.observation = env.update_rule_to_observation(update_rule)
terminated, truncated = False, False
while not (terminated or truncated):
    action = recovered_algo.compute_single_action(env.observation)
    _, reward, terminated, truncated, info = env.step(action)
    max_reward = max(max_reward, reward)
    episode_reward += reward

print(f"Total reward from the episode: {episode_reward}")
print(f"Max reward from the episode: {max_reward}")
