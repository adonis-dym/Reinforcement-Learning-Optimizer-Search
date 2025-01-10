import random
import re
from globals import *
# Variable names list for mapping the variable indices to names
var_names = ['weight', 'grad', 'v1', 'v2', 'update']
for _i in range(5, MAX_VARIABLES):
    var_names.append(f'v_{_i}')


def generate_update_rule(n_statements):
    '''
    Randomly generate an update rule with a given number of statements.
    '''
    # In the beginning we have 4 variables. The result of each statement may be stored in a new variables
    # We need to guarantee that in the worst case, when all statement require new variables, the available slots are still enough
    if 4 + n_statements > MAX_VARIABLES:
        raise ValueError(
            "The variable slots are not enough. This will cause an error when all slots are used up and it still prompts to add a new variable.")
    while True:  # Keep generating rules until a valid one is produced
        update_rule = []

        # We have 4 variables to start with: w, g, v1, and v2
        variable_indices = [0, 1, 2, 3]

        for _ in range(n_statements):
            operation = random.choice(operations)
            operands = []

            # For each operand, decide whether to use a variable or a constant
            for _ in range(operation['n_operands']):
                if random.random() < 0.2:
                    constant_index = random.randint(0, NUM_CONSTANTS - 1)
                    operands.append(('c', constant_index))
                else:
                    operands.append(('v', random.choice(variable_indices)))

            # Create a new variable with probability 1/(len(variable_indices)+1)
            if random.random() < 1/(len(variable_indices)+1):
                # Get a set of unused indices
                unused_indices = set(range(MAX_VARIABLES)) - set(variable_indices)
                # Randomly select an unused index
                new_index = random.choice(list(unused_indices))
                result_index = new_index
                variable_indices.append(result_index)
            # Otherwise, pick an existing variable to overwrite
            else:
                result_index = random.choice(variable_indices)

            update_rule.append([operation, operands, result_index])

        # If the 'update' variable (corresponds to index 4) has been created, return the rule
        if 4 in variable_indices:
            return update_rule


def show_update_rule(update_rule):
    # Translate each step of the update rule into a readable format
    readable_steps = []
    for operation, operands, result_index in update_rule:
        operand_names = []
        for t, i in operands:
            if t == 'v':
                operand_names.append(var_names[i])
            elif t == 'c':
                operand_names.append(str(constants[i]))  # Use actual constant value for readability
        result_name = var_names[result_index]
        op_func_name = operation['func'].__name__
        readable_step = f"{result_name} = torch.{op_func_name}({', '.join(operand_names)})"
        readable_steps.append(readable_step)

    return '\n'.join(readable_steps)


def text_to_update_rule(rule_text):
    # Function mapping names to torch functions
    def name_to_torch_func(name):
        for op in operations:
            if op['func'].__name__ == name:
                return op
        raise ValueError(f"Unrecognized torch function: {name}")

    # Function mapping names to variable indices
    def name_to_var_index(name):
        if name in var_names:
            return var_names.index(name)
        raise ValueError(f"Unrecognized variable name: {name}")

    def name_to_const_index(const_value):
        try:
            # Convert string to float and find its index in the constants list
            const_value = float(const_value)
            return constants.index(const_value)
        except ValueError:
            raise ValueError(f"Unrecognized constant value: {const_value}")

    update_rule = []
    lines = rule_text.strip().split('\n')  # Remove leading and trailing whitespaces

    for line in lines:
        # Parse the line with regex
        match = re.match(r"(\w+) = torch.(\w+)\((.+)\)", line)
        if not match:
            raise ValueError(f"Line format is incorrect: {line}")

        # Extract components
        result_name, op_func_name, operands_text = match.groups()
        # Convert the function name to the actual torch function
        operation = name_to_torch_func(op_func_name)

        # Convert the operand names to variable indices or floats
        operands = []
        for operand_text in re.split(', ', operands_text):
            if operand_text in var_names:
                operands.append(('v', name_to_var_index(operand_text)))
            else:
                operands.append(('c', name_to_const_index(operand_text)))

        # Convert the result variable name to its index
        result_index = name_to_var_index(result_name)

        # Add the parsed operation to the rule
        update_rule.append([operation, operands, result_index])

    return update_rule
