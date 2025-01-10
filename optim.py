import warnings
from update_rule import *


class RLSearchOptimizer(torch.optim.Optimizer):
    def __init__(self, params, update_rule, lr=LEARNING_RATE):
        defaults = dict(lr=lr)
        super(RLSearchOptimizer, self).__init__(params, defaults)
        self.compiled_update_rule = self.compile_update_rule(update_rule)

        # Show the update rule when building the optimizer
        print('Update Rule'.center(30, '-'))
        print(show_update_rule(update_rule))
        print('-'*30)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # Initialize the state variables to be a small nonzero value
                # Some algorithms involve the division of the state variables, like Adam
                # If the state variables are initialized to 0, the algorithm will break, hinders the potential of the discovery
                # Also we set with a small positive value as Adam also involves sqrt
                state['v1'] = torch.zeros_like(p).uniform_(0, 1e-10).detach()
                state['v2'] = torch.zeros_like(p).uniform_(0, 1e-10).detach()

    def compile_update_rule(self, update_rule):
        '''
        Compile the update rule into a list of functions that can be executed with Python closure mechanism.
        Its advantage is that we don't need to parse the update rule every time we execute it.
        Obtains an ~2.5x speedup in training
        '''
        compiled_operations = []
        for i, (operation, operands, result_index) in enumerate(update_rule):
            op_func = operation['func']
            n_operands = operation['n_operands']

            # Pre-bind the operands
            bound_operands = []
            for operand in operands:
                operand_type, operand_value = operand
                if operand_type == 'c':  # constant using index
                    # Bind a tensor with the constant value from the constants list
                    bound_operands.append(
                        lambda var, idx=operand_value: torch.tensor(constants[idx], device=var[0].device))
                elif operand_type == 'v':  # variable
                    # Bind a function that will retrieve the variable from the variables list
                    bound_operands.append(
                        lambda var, idx=operand_value: var[idx])

            def make_executable(op_func, bound_operands, n_operands, result_index, statement_index):
                if n_operands == 1:
                    def exec_op(variables):
                        op1 = bound_operands[0](variables)
                        result = op_func(op1)
                        # The warning module heavily impacts the running time, so only uncomment this when debugging
                        # if torch.isnan(result).any() or torch.isinf(result).any():
                        #     warning_message = show_update_rule(
                        #         [update_rule[statement_index]])
                        #     warning_message += " resulted in nan or inf values."
                        #     warnings.warn(warning_message)
                        variables[result_index] = result
                elif n_operands == 2:
                    def exec_op(variables):
                        op1 = bound_operands[0](variables)
                        op2 = bound_operands[1](variables)
                        result = op_func(op1, op2)
                        # The warning module
                        # if torch.isnan(result).any() or torch.isinf(result).any():
                        #     warning_message = show_update_rule(
                        #         [update_rule[statement_index]])
                        #     warning_message += " resulted in nan or inf values."
                        #     warnings.warn(warning_message)
                        variables[result_index] = result
                else:
                    raise ValueError("Invalid number of operands")
                return exec_op

            compiled_operations.append(make_executable(
                op_func, bound_operands, n_operands, result_index, i))
        return compiled_operations

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                # Initialize the variables list
                variables = [p, p.grad, self.state[p]['v1'],
                             self.state[p]['v2']] + [None]*(MAX_VARIABLES - 4)
                # Execute the compiled update rule
                for exec_op in self.compiled_update_rule:
                    exec_op(variables)

                update, self.state[p]['v1'], self.state[p]['v2'] = variables[4], variables[2], variables[3]
                p.sub_(lr * update)  # Scale the update by the learning rate
