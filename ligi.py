import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List


class ligi(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0, gamma=0.9, eps=1e-15, foreach: bool = True):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1]))
        if not 0.0 <= gamma <= 1.0:
            raise ValueError('Invalid gamma value: {}'.format(gamma))

        defaults = dict(lr=lr, betas=betas,
                        weight_decay=weight_decay, gamma=gamma, eps=eps, foreach=foreach)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []

            beta1, beta2 = group['betas']
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction_ct = beta1 * \
                (1 - beta2 ** (group['step'] - 1)) + 1 - beta1

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avgs.append(state['exp_avg'])

            if not params_with_grad:
                continue

            kwargs = dict(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                beta1=beta1,
                beta2=beta2,
                gamma=group['gamma'],
                eps=group['eps'],
                bias_correction_ct=bias_correction_ct,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
            )

            if group['foreach']:
                self._multi_tensor_op(**kwargs)
            else:
                raise ValueError('foreach=False is not supported yet')

        return loss

    def _multi_tensor_op(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        *,
        beta1: float,
        beta2: float,
        gamma: float,
        eps: float,
        bias_correction_ct: float,
        lr: float,
        weight_decay: float,
    ):
        if len(params) == 0:
            return

        c = torch._foreach_mul(exp_avgs, beta1)
        torch._foreach_add_(c, grads, alpha=1 - beta1)  # c_t
        torch._foreach_div_(c, bias_correction_ct)  # c_t_hat
        torch._foreach_mul_(exp_avgs, beta2)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta2) # m_t

        denom = torch._foreach_abs(c)
        torch._foreach_mul_(denom, gamma) # gamma |c_t_hat|
        torch._foreach_abs_(grads)
        torch._foreach_add_(denom, grads, alpha=1 - gamma)
        # for numerical stability, to prevent denom==0
        torch._foreach_add_(denom, eps)

        # weight decay
        torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_addcdiv_(params, c, denom, value=-lr)
