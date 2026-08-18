import os
from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class ParamOptConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    scale_factor: Literal['scene_radius'] | None = None
    scheduler_type: type = None
    scheduler_cfg: dict = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    params: dict[str, ParamOptConfig] = field(default_factory=dict)
    opt_cls: type = torch.optim.Adam


def make_opt_cfg(cfg: dict) -> OptimizationConfig:
    params = cfg['params']
    return OptimizationConfig(
        params={
            name: (
                ParamOptConfig(**params[name])
                if not isinstance(params[name], ParamOptConfig)
                else params[name]
            ) for name in params
        },
        opt_cls=cfg['opt_cls'] if 'opt_cls' in cfg else torch.optim.Adam
    )


_announced_fused = False  # print the substitution notice only once per process


def _resolve_opt_cls(opt_cls, groups):
    """Substitute our one-thread-per-element CUDA Adam (``xsim.optim.FusedAdam``)
    for the stock ``torch.optim.Adam``. It's numerically equivalent but launches a
    data-sized grid, so our multi-million-row Gaussian tensors saturate the GPU
    instead of running on PyTorch's tiny list-oriented fused grid (~2.4x faster
    end-to-end). Only when every param is a CUDA floating-point tensor and no
    group requests a feature FusedAdam lacks (weight decay, amsgrad, maximize) --
    otherwise the stock optimizer is kept. Set ``XSIM_DISABLE_FUSED_ADAM=1`` to
    force the stock optimizer."""
    if opt_cls is not torch.optim.Adam:
        return opt_cls
    if os.environ.get('XSIM_DISABLE_FUSED_ADAM', '') not in ('', '0', 'false', 'False'):
        return opt_cls
    tensors = [t for g in groups for t in g['params']]
    if not tensors or not all(t.is_cuda and t.is_floating_point() for t in tensors):
        return opt_cls
    if any(g.get(feat) for g in groups
           for feat in ('weight_decay', 'amsgrad', 'maximize')):
        return opt_cls
    from xsim.optim import FusedAdam
    global _announced_fused
    if not _announced_fused:
        _announced_fused = True
        print('[optimization] replacing torch.optim.Adam with xsim.optim.FusedAdam; '
              'set XSIM_DISABLE_FUSED_ADAM=1 to opt out')
    return FusedAdam


def init_optimization(
        opt_cfg: OptimizationConfig,
        params: dict[str, torch.nn.Parameter],
        lr_scale_factor_resolver,
        cls_name: str
):
    opt_cfg = make_opt_cfg(opt_cfg) if isinstance(opt_cfg, dict) else opt_cfg
    module_params = set(list(params.keys()))
    opt_cfg_params = set(list(opt_cfg.params.keys()))

    if module_params != opt_cfg_params:
        raise ValueError(
            'Parameters list of module and optimization config '
            'does not match for node {}. Module params: {}. '
            'Optimization config params: {}'.format(
                cls_name,
                ', '.join(list(module_params)),
                ', '.join(list(opt_cfg_params))
            )
        )

    groups = [
        {
            "params": (list(params[param].parameters())
                       if isinstance(params[param], torch.nn.Module)
                       else [params[param]]),
            "name": param,
            "lr": (opt_cfg.params[param].lr *
                   lr_scale_factor_resolver(opt_cfg.params[param].scale_factor)),
            "weight_decay": opt_cfg.params[param].weight_decay
        } for param in opt_cfg_params
    ]
    opt = _resolve_opt_cls(opt_cfg.opt_cls, groups)(groups)

    schedulers = [
        opt_cfg.params[param].scheduler_type(
            opt, param,
            lr_scale_factor_resolver(opt_cfg.params[param].scale_factor),
            opt_cfg.params[param].scheduler_cfg
        ) for param in opt_cfg_params
        if opt_cfg.params[param].scheduler_type is not None
    ]

    return opt, schedulers

