import torch
import os
# Define the global variables and params
operations = [
    {'func': torch.add, 'n_operands': 2},
    {'func': torch.sub, 'n_operands': 2},
    {'func': torch.mul, 'n_operands': 2},
    {'func': torch.true_divide, 'n_operands': 2},
    {'func': torch.abs, 'n_operands': 1},
    {'func': torch.exp, 'n_operands': 1},
    {'func': torch.log, 'n_operands': 1},
    {'func': torch.sign, 'n_operands': 1},
    {'func': torch.exp2, 'n_operands': 1},
    {'func': torch.square, 'n_operands': 1},
    {'func': torch.sqrt, 'n_operands': 1},
    {'func': torch.reciprocal, 'n_operands': 1}
]

# Define all the possible constants. Only includes the constants often used in typical optimizers
constants = [-1, -0.999, -0.99, -0.98, -0.95, -0.9, -0.5, -0.1, -0.05, -0.02, -0.01, -0.001,
             0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 0.9, 0.95, 0.98, 0.99, 0.999, 1]

# You may not change these settings. Change the `operations` or `constants` lists above instead.
NUM_OPERATIONS = len(operations)
NUM_OPERANDS = max(o['n_operands'] for o in operations)
NUM_CONSTANTS = len(constants)

# The following global variables can be overrided by the dict passed to the environment
# ------------------ ENVIRONMENT CONFIGURATION ---------------------
# The max variables and statements in the program, which affect the size of the state and action spaces
MAX_VARIABLES = 20
MAX_STATEMENTS = 50

# The max allowed number of steps in one episode
MAX_STEPS = 10000

# The number of statements when randomly generating a new update rule. Good to set to int(MAX_VARIABLES/2)
INIT_PROGRAM_LENGTH = int(MAX_VARIABLES/2)

# Caching settings. It may be disabled in debugging, or the user believes that the program evaluation process would not provide reliable data to store into the cache.
ENABLE_CACHE = True

# Prediction settings. Whether to use the predictor to predict the performance of the optimizer
ENABLE_PREDICTION = True
# ---------------- END ENVIRONMENT CONFIGURATION ------------------

# ----------------------------- REWARDS ---------------------------
# The metric when the program is useless. Task specific. E.g. The metric is accuracy, 0.1, for the Cifar10 image classification task
TRIVIAL_METRIC = 0.1

# The reward when the action/post observation is invalid.
INVALID_REWARD = -1

# The reward when all of the statements are removed and the game is terminated
TERMINATION_REWARD = -100

# The scaling factor for runnable programs, including the useless ones. (The useless optimizer is also runnable and produces a trivial metric)
# This is intended for magnifying the usefulness of the runnable programs
SCALING_FACTOR = 500
# --------------------------- END REWARDS -------------------------

# --------------------- EVALUATION SETTINGS -----------------------
# The number of epochs which trigger the prediction. Needs to be a list. 1-indexed
TRIAL_EPOCHS = [5, 10]
# Total evaluation epochs
TOTAL_EPOCHS = 30
# The metric thershold. Early stop if the predicted metric is lower than this threshold
EARLY_STOP_THRESHOLD = 0.13
# ----------------- END EVALUATION SETTINGS -----------------------

# ------------------ OPTIMIZATION SETTINGS ------------------------
# Each optimizer program is ended with p.sub_(lr * update) with a lr scheduler. See optim.py.
# For different tasks we need to adjust the learning rate accordingly, even for the same optimizer program.
# Best to achieve that the RL agent does not need to learn to adjust the lr,
# in which way the performance of a candidate will be consistently good (or bad) across multiple tasks.
LEARNING_RATE = 0.1
# ---------------- END OPTIMIZATION SETTINGS ----------------------

