import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(__file__), 'csrc', 'fused_adam.cu')
    return load(name='xsim_fused_adam', sources=[src], verbose=False)


class FusedAdam(torch.optim.Optimizer):
    """Adam with a single-thread-per-element CUDA kernel. CUDA + float only."""

    # options FusedAdam does not implement; a non-falsy value for any of these
    # (whether in defaults or a param group) is a hard error rather than a
    # silent no-op, so a misconfigured `opt_cls: FusedAdam` fails loudly.
    UNSUPPORTED = ('weight_decay', 'amsgrad', 'maximize')

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0.0:
            raise ValueError('invalid lr {}'.format(lr))
        if eps < 0.0:
            raise ValueError('invalid eps {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError('invalid betas {}'.format(betas))
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
        for group in self.param_groups:
            bad = [f for f in self.UNSUPPORTED if group.get(f)]
            if bad:
                raise ValueError(
                    'FusedAdam does not support {}; use torch.optim.Adam'.format(
                        ', '.join(bad)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ext = _ext()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr, eps = group['lr'], group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse grads')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                t = state['step']
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                if not p.is_contiguous():
                    raise ValueError("Parameter is not contiguous: {} {}".format(group['name'], p.shape))
                ext.step(p, grad.contiguous(), state['exp_avg'], state['exp_avg_sq'],
                         lr, beta1, beta2, eps, bc1, bc2)
        return loss
