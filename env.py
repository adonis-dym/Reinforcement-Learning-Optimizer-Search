import warnings
from gymnasium.spaces import Tuple, Discrete, Dict
import gymnasium as gym
import numpy as np
from copy import deepcopy
from gymnasium.utils.env_checker import check_env
from graph import *
import pipeline
from cache import Cache
from collections import OrderedDict

class RLSearchEnv(gym.Env):
    def __init__(self, config):

        self.episode_length = 0
        self.episode_count = 0

        super(RLSearchEnv, self).__init__()
        self.n_operations = NUM_OPERATIONS
        self.n_operands = NUM_OPERANDS
        self.n_constants = NUM_CONSTANTS
        # maximum number of variables in the update rule
        self.n_variables = config['max_variables']
        # maximum number of statements in the update rule
        self.n_statements = config['max_statements']
        # Maximum number of steps per episode
        self.max_steps = config['max_steps']

        # The initial length of the program. This is used when generating the initial update rule in the reset method
        self.init_program_length = config['init_program_length']
        # After initialization, we must call self.reset() to get the first observation
        self.observation = None
        self.current_step = None  # Current step count. Call reset to define its initial value

        # The cache for storing the (graph, reward) pairs. Call reset to initialize
        self.cache = None

        # Each operation is represented as a tuple of (operation, operand1, operand2, result_index)
        operation_space = Dict({
            # '+1' to account for 'no operation' placeholder
            'func': Discrete(self.n_operations + 1),
            'operands': Tuple([
                Dict({
                    # Variable type: 0 for 'v', 1 for 'c', and 2 to account for 'no operand' placeholder
                    'type': Discrete(3),
                    # '+1' to account for 'no constant' placeholder
                    'constant_index': Discrete(self.n_constants + 1),
                    # '+1' to account for 'no variable' placeholder
                    'variable_index': Discrete(self.n_variables + 1)
                }) for _ in range(self.n_operands)
            ]),
            # '+1' to account for 'no result' placeholder
            'result_index': Discrete(self.n_variables + 1)
        })

        # Special "no operand" value
        self.no_operand = {'type': 2, 'constant_index': self.n_constants, 'variable_index': self.n_variables}

        # Special "no operation" value
        self.no_operation = {
            'func': self.n_operations,
            'operands': [self.no_operand, self.no_operand],
            'result_index': self.n_variables
        }

        # Define action space
        # Action type: 0 - add, 1 - remove, 2 - mutate
        # Statement index: index of the statement to add, remove, or mutate
        # For add action: the new statement to be added
        # For mutate action: the part of the statement to mutate and the new value
        self.action_space = Dict({
            'action_type': Discrete(3),
            'statement_index': Discrete(self.n_statements),
            # This will be ignored if action type is not 'add'
            'add_statement': operation_space,
            # This will be ignored if action type is not 'mutate', only allows mutation of operands
            'mutate_part': Discrete(self.n_operands),
            # This will be ignored if action type is not 'mutate' or operand is not a constant, represents the index of the new constant to mutate to
            'mutate_constant_index': Discrete(self.n_constants),
            # This will be ignored if action type is not 'mutate' or operand is not a variable, represents the new variable to mutate to
            'mutate_variable_index': Discrete(self.n_variables)
        })

        # Define observation space
        # The update rule is a sequence of operations
        self.observation_space = Tuple([operation_space]*self.n_statements)

        self.pipeline = pipeline.Pipeline()

    def update_rule_to_observation(self, update_rule):
        observation = []
        for operation, operands, result_index in update_rule:
            # Find the index of the operation function in the operations list
            operation_index = next(i for i, op in enumerate(operations) if op['func'] == operation['func'])

            operation_dict = {
                'func': operation_index,
                'operands': [],
                'result_index': result_index,
            }

            for i in range(self.n_operands):
                if i < operation['n_operands']:
                    type, value = operands[i]
                    if type == 'c':
                        operand_dict = {
                            'type': 1,
                            'constant_index': value,  # Use constant index directly
                            'variable_index': self.n_variables  # No variable used
                        }
                    else:  # type == 'v'
                        operand_dict = {
                            'type': 0,
                            'constant_index': self.n_constants,  # No constant used
                            'variable_index': value
                        }
                else:
                    operand_dict = self.no_operand

                operation_dict['operands'].append(operand_dict)

            observation.append(operation_dict)

        # Pad the observation with "no operation" statements until it reaches the maximum length
        while len(observation) < self.n_statements:
            observation.append(self.no_operation)

        return observation

    def observation_to_update_rule(self, observation):
        update_rule = []
        for statement in observation:
            # Check if the statement is a 'no operation' placeholder
            if statement['func'] == self.n_operations:
                continue

            # Find the operation function in the operations list using the index
            operation = operations[statement['func']]

            operands = []
            for operand_dict in statement['operands']:
                # Check the type of the operand
                if operand_dict['type'] == 0:  # 'v'
                    operands.append(('v', operand_dict['variable_index']))
                elif operand_dict['type'] == 1:  # 'c'
                    operands.append(('c', operand_dict['constant_index']))
                elif operand_dict['type'] == 2:  # 'no operand' placeholder
                    continue

            result_index = statement['result_index']

            update_rule.append([operation, operands, result_index])
        return update_rule

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the current step count
        self.current_step = 0

        # Generate an update rule
        update_rule = generate_update_rule(self.init_program_length)
        # Convert update rule to observation
        self.observation = self.update_rule_to_observation(update_rule)

        if ENABLE_CACHE:
            # Initialize cache
            self.cache = Cache()
        else:
            warnings.warn("Cache is disabled. To enable cache please set ENABLE_CACHE to True in globals.py")
        return self.observation, {}

    def get_program_data(self, observation):
        '''
        Returns the valid program length and valid variable indices in the program. 
        Note that it accepts the whole observation as well as partial of the statements as input.

        When use the partial one, we statistic the maximum index of the variables, to determine what variable to use as the destination of mutation, or the index of new variable to add.
        See the step() function for its usage.
        '''

        # Get the valid length of the current observation
        nop_count = observation.count(self.no_operation)
        program_length = len(observation) - nop_count

        # Get the valid variable indices in the program
        valid_variable_indices = {stmt['result_index']
                                  for stmt in observation if stmt != self.no_operation} | {0, 1, 2, 3}  # w,g,v1,v2

        return program_length, valid_variable_indices

    def step(self, action):
        # Convert the action into native Python data format
        action = convert_action(action)
        # pre-check on the whether the action can be applied to the given index of statement
        def _pre_check(valid_length, action_type, statement_index):
            # # If all of the statements are removed, the agent loses the game
            # if valid_length == 0:
            #     return self.observation, -1.0, True, False, {}

            # We can assume that valid_length > 0, as it is already checked in the post_check function in the previous step
            # Check if the statement_index is within the valid length
            if statement_index >= valid_length:
                # Statement_index is invalid. Return a small negative reward,
                # keep the state unchanged, and continue with the next action.
                return False

            # If there are no valid operations left, only "add" actions are considered valid
            # if valid_length == 0 and action_type != 0:
            #     return False

            # # Check if the statement_index is within the valid length
            # if statement_index >= valid_length:
            #     # We need to make an exception for the case in which the valid length is 0 and the action type is 0, which adds a statement to a null observation
            #     # In this case, although statement_index >= valid_length holds (which usually means the action is invalid)
            #     # we still allow the insertion to the 0-th position
            #     if not (valid_length == 0 and action_type == 0 and statement_index == 0):
            #         # Statement_index is invalid. Return a small negative reward,
            #         # keep the state unchanged, and continue with the next action.
            #         return False

            return True

        # Function to perform action based on action type
        # Also it contains some sanity checks and validity checks
        def _perform_action(action_type, statement_index):
            if action_type == 1:  # remove
                # Sanity check. The assertion should never fail if implemented correctly
                assert self.observation[statement_index] != self.no_operation, "Invalid action: Can't remove a no operation"

                # Set the operation at the statement_index to nop
                self.observation[statement_index] = self.no_operation

                # Realign the operations to ensure all nops are at the end
                self.observation.sort(key=lambda op: op == self.no_operation)
                return True

            elif action_type == 2:  # mutate
                # Extract mutation details
                mutate_part = action['mutate_part']
                mutate_constant_index = action['mutate_constant_index']
                mutate_variable_index = action['mutate_variable_index']

                operands_list = self.observation[statement_index]['operands']

                # Make sure mutate_part is valid
                valid_operands = len(operands_list) - operands_list.count(self.no_operand)
                if mutate_part >= valid_operands:
                    # mutate_part exceeds the scope of valid operands, making the action invalid.
                    # Return a small negative reward, keep the state unchanged, and continue with the next action.
                    return False

                # Get the operand to mutate
                operand = operands_list[mutate_part]

                # Sanity check. The assertion should never fail if implemented correctly
                assert operand != self.no_operand, "Trying to mutate a 'no_operand'"

                # If operand type is 0 (variable), we can only mutate the variable_index
                # If operand type is 1 (constant), we can only mutate the constant_index
                if operand['type'] == 0:  # variable
                    # When mutating a variable, after we know the statement to mutate, we need to know the list of valid variables when the program executes to this statement
                    _, indices = self.get_program_data(self.observation[:statement_index])
                    if mutate_variable_index not in indices or mutate_variable_index == operand['variable_index']:
                        # mutate_variable_index is not in the scope of valid variables, or the new variable index is the same as the old one, making the action invalid.
                        return False
                    else:
                        operand['variable_index'] = mutate_variable_index

                elif operand['type'] == 1:  # constant
                    if mutate_constant_index == operand['constant_index']:
                        # mutate_constant_index is the same as the old one, making the action invalid.
                        return False
                    else:
                        operand['constant_index'] = mutate_constant_index
                return True

            elif action_type == 0:  # add
                # Extract the new statement details
                new_statement = action['add_statement']
                operand_list = new_statement['operands']

                # Get the length of the current program
                program_length, _ = self.get_program_data(self.observation)

                # Check if there are any `no_operand` slots available
                if program_length == len(self.observation):
                    # No slots available; the add action is invalid.
                    return False

                # Validate the new statement
                # 1. Check the function is not the one accounting for no_operation
                if new_statement['func'] == self.n_operations:
                    # The function is no_operation; the add action is invalid.
                    return False

                # 2. Check the number of valid operands and their order
                expected_operands = operations[new_statement['func']]['n_operands']
                count_valid_operands = 0
                for operand in operand_list:
                    # we use whether the type is 2 to determine whether it is a no_operand, a little loose
                    if operand['type'] != 2:
                        count_valid_operands += 1
                    else:
                        break  # stop counting at the first no_operand

                if count_valid_operands != expected_operands or any(op['type'] != 2 for op in operand_list[count_valid_operands:]):
                    # Invalid number of operands or invalid operand order; the add action is invalid.
                    return False

                # 3. Check the validity of operands
                _, indices = self.get_program_data(self.observation[:statement_index])
                for operand in operand_list:
                    if operand['type'] == 0 and operand['variable_index'] not in indices:  # variable
                        # variable_index is not within the scope of valid variables; the add action is invalid.
                        return False
                    if operand['type'] == 1 and operand['constant_index'] == self.n_constants:  # constant
                        # constant_index is the 'no constant' placeholder; the add action is invalid.
                        return False
                # 4. Check the validity of result_index
                if new_statement['result_index'] == self.n_variables:
                    # result_index corresponds to the 'nop' slot; the add action is invalid.
                    return False

                # If validation passed,
                # 1. In the new statement, assign all the operands whose 'type' == 2 to be `self.no_operand`, to keep a consistent format
                operand_list = [self.no_operand if operand['type'] == 2 else operand for operand in operand_list]

                # 2. Make room for the new statement
                # This involves shifting all valid statements (not no_operation) after statement_index one step to the right.
                # Use list.insert() method to insert the new statement and shift subsequent statements
                self.observation.insert(statement_index, new_statement)

                # Sanity check. The assertion should never fail if implemented correctly
                assert self.observation[-1] == self.no_operation, "Last item is not a 'no_operation'"
                # Use the del statement to remove the last item from the list
                del self.observation[-1]

                return True

        # post check on the program length
        def _post_check():
            """
            Post-check after applying the action.
            If all of the statements are removed (i.e., valid_length == 0), the agent loses the game
            Returns the termination variable
            """
            # Important: after executing the action, the observation is changed, so we need to get the new valid_length
            valid_length, _ = self.get_program_data(self.observation)
            return valid_length == 0

        def get_reward(backup_observation):
            # Try to build and evaluate optimizer based on the post observation
            update_rule = self.observation_to_update_rule(self.observation)

            print('Original Update Rule', show_update_rule(update_rule), sep='\n')

            # Build computational graph from the post observation. Note that it still contains the post observation validity check
            graph = ComputationalGraph(update_rule)
            # If invalid, roll back to the original observation and return a small negative reward
            if not graph.valid:
                self.observation = backup_observation
                return INVALID_REWARD
            # 'useful' variable to indicate whether the graph is not trivial. See the 'refine' function in graph.py for details
            useful = graph.refine()
            if not useful:
                # Runnable, but trivial program. It returns a trivial metric and also needs scaling
                return TRIVIAL_METRIC*SCALING_FACTOR
            # Check whether the (refined) graph is cached
            if ENABLE_CACHE:
                found, list_name, id, metric = self.cache.find_item(graph)
                if found:
                    print(f'Graph found in cache {list_name} with id {id}, using cached performance: {metric}')
                    # If the graph is in the cache, use the cached metric and return. Again, it returns metric, we need to scale it
                    return metric*SCALING_FACTOR

            # If the code execution reaches here, it means the graph was not found in the cache
            # Fully evaluate it, and store the result into the cache in case the cache is enabled
            net = self.pipeline.create_net()
            # We refine update rule with the help of graph. To save computation we should use the refined rule to build optimizer
            refined_update_rule = graph_to_update_rule(graph)

            print('Refined Update Rule', show_update_rule(refined_update_rule), sep='\n')

            optimizer = self.pipeline.create_optimizer(net, refined_update_rule)
            # 'useful' indicates that whether the program incurs RuntimeError such as ZeroDivisionError
            # The reward returns here needs to be scaled
            useful, performance = self.pipeline.train(net, optimizer)

            if not useful:
                return INVALID_REWARD

            if ENABLE_CACHE:
                # Store the new graph and its performance into the cache
                self.cache.add_item(graph, performance)

            # Scale the reward only when useful
            reward = SCALING_FACTOR*performance

            return reward

        # -----Main function body----- #
        # Increment the current step count
        self.current_step += 1

        # Create a backup. Once the action is detected invalid, we can restore the self.observation to be the backup
        backup_observation = deepcopy(self.observation)

        # Extract the action components
        action_type = action['action_type']
        statement_index = action['statement_index']

        # Get the data of the program
        valid_length, _ = self.get_program_data(self.observation)

        # Check whether the action is valid both on the pre-check and executing the action
        is_valid = _pre_check(valid_length, action_type, statement_index) and \
            _perform_action(action_type, statement_index)

        # Perform post-check to see if the game is terminated
        terminated = _post_check()

        # Check the observation data is in the expected format
        # When the code goes to this, the observation has already been applied with the action
        check_observation(self.observation)

        # Set reward
        # The action is invalid. Return a small negative reward
        if not is_valid:
            reward = INVALID_REWARD
        # All of the statements are removed, and the game is terminated. Return a large negative reward
        elif terminated:
            reward = TERMINATION_REWARD
        else:
            reward = get_reward(backup_observation)

        # Truncation condition
        truncated = self.current_step >= self.max_steps
        assert isinstance(reward, (int, float)), f"Reward is not an int or float: {reward, type(reward)}"

        # Stepping and Printing
        self.episode_length += 1
        print(f"Episode {self.episode_count} Step {self.current_step} finished, reward: {reward}")
        # The end of an episode
        if terminated or truncated:
            if ENABLE_CACHE:
                # Save the cache to disk
                self.cache.save_to_disk()

            print(f"Episode {self.episode_count} finished after {self.episode_length} timesteps.")
            self.episode_length = 0
            self.episode_count += 1
        return self.observation, reward, terminated, truncated, {}

# Helper functions
# 2024.12.02 [bug fix]: The gymnasium library always returns `collections.OrderedDict` when the observation space is a `Dict` space.
# Also, the values returned for `Discrete` space will be numpy ints.
# For `Tuple` space, the values are still python default tuple class.
# The step() method directly assign the action values to the observations, causing the observations to have mixed types of representations.
# We fix it by data conversion each time we need that assignments.

def check_observation(observation):
    """
    Check if the observation data is in the expected native Python format.

    Args:
    observation: The observation data from the environment.

    Raises:
    AssertionError: If any data type is not as expected.
    """

    def check_operation(operation):
        # Assert that the operation dictionary is not an OrderedDict
        assert not isinstance(operation, OrderedDict), "Operation must be a dict, not OrderedDict"

        # Check 'func' and 'result_index' are integers
        assert isinstance(operation['func'], int), f"Function index must be int, found {type(operation['func'])}"
        assert isinstance(operation['result_index'], int), f"Result index must be int, found {type(operation['result_index'])}"

        # Check each operand within the tuple
        for operand in operation['operands']:
            assert not isinstance(operand, OrderedDict), "Operand must be a dict, not OrderedDict"
            assert isinstance(operand['type'], int), f"Operand type must be int, found {type(operand['type'])}"
            assert isinstance(operand['constant_index'], int), f"Constant index must be int, found {type(operand['constant_index'])}"
            assert isinstance(operand['variable_index'], int), f"Variable index must be int, found {type(operand['variable_index'])}"

    # Iterate over each operation in the observation tuple
    for operation in observation:
        check_operation(operation)

def convert_action(action):
    """
    Convert OrderedDict action sample from the RL environment to a native Python dict with proper types.
    The action data format may be of two types:
        1. `collections.OrderedDict` object (in gymnasium `Dict` space) with numpy integers (in gymnasium `Discrete` space)
        2. Vanilla Python dict with native integers
    This function handles both case and converts the action to the second case, 
    which is essential for the environment to process the action (like `isinstance` checks).

    Args:
    action: The sampled action.

    Returns:
    dict: The converted action dictionary.
    """
    # Initialize the converted action dictionary
    converted_action = {}
    # Process the top-level keys
    for key, value in action.items():
        if key == 'action_type':
            # Convert the action type directly to int
            converted_action[key] = int(value)
        elif key == 'statement_index':
            # Convert the statement index directly to int
            converted_action[key] = int(value)
        elif key == 'mutate_part' or key == 'mutate_constant_index' or key == 'mutate_variable_index':
            # Convert integers directly
            converted_action[key] = int(value)
        elif key == 'add_statement':
            # Convert the nested OrderedDict in add_statement
            converted_action[key] = {
                'func': int(value['func']),
                'operands': tuple({k: int(v) if isinstance(v, np.integer) else v
                                    for k, v in op.items()}
                                    for op in value['operands']),
                'result_index': int(value['result_index'])
            }
    return converted_action
        
# Debug the env without RLLib
if __name__ == "__main__":
    env_config = {
        'max_variables': MAX_VARIABLES,
        'max_statements': MAX_STATEMENTS,
        'max_steps': MAX_STEPS,
        'init_program_length': INIT_PROGRAM_LENGTH
    }

    for _ in range(10000):
        env = RLSearchEnv(env_config)
        check_env(env)
        done = False
        obs = env.reset()
        while not done:
            action = env.action_space.sample()
            # In testing, the gymnasium sample method always returns an OrderedDict with numpy integers
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

