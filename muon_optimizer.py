import torch
from torch.optim.optimizer import Optimizer

class Muon(Optimizer):
    """A simple Muon optimizer implementation.

    This implementation follows an AdamW-like update but without bias correction.
    It is intended as a lightweight drop-in for experimentation.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group.get('weight_decay', 0.0)
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                step = exp_avg / denom

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.add_(step, alpha=-lr)

        return loss
