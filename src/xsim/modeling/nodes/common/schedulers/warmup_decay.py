from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


class BasicScheduler:
    def __init__(self, opt: torch.optim.Optimizer, param: str, lr_scale_factor: float):
        self.opt = opt
        self.param = param
        self.lr_scale_factor = lr_scale_factor
        self._step = 0
        
        for idx, param_group in enumerate(self.opt.param_groups):
            if param_group['name'] == param:
                self.param_group_idx = idx
                break

    def compute_lr(self, step: int) -> float:
        raise NotImplementedError

    def step(self):
        group = self.opt.param_groups[self.param_group_idx]
        assert group['name'] == self.param
        group['lr'] = self.compute_lr(self._step)
        self._step += 1


class WarmupDecayScheduler(BasicScheduler):
    @dataclass
    class Config:
        lr_init: float
        lr_final: float = None
        lr_pre_warmup: float = None

        opt_start_iter: int = 0
        warmup_fn: Literal['cosine'] | Literal['linear'] = 'linear'
        warmup_iters: int = 0
        total_iters: int = None

    def __init__(
            self,
            opt: torch.optim.Optimizer,
            param: str,
            lr_scale_factor: float,
            cfg: Config
    ):
        super().__init__(opt=opt, param=param, lr_scale_factor=lr_scale_factor)
        self.cfg = WarmupDecayScheduler.Config(**cfg) if isinstance(cfg, dict) else cfg
        self.cfg.lr_final = (
            self.cfg.lr_final if self.cfg.lr_final is not None else self.cfg.lr_init
        )
        self.cfg.lr_pre_warmup = (
            self.cfg.lr_pre_warmup if self.cfg.lr_pre_warmup is not None else 0
        )
        self.cfg.lr_init *= self.lr_scale_factor
        self.cfg.lr_final *= self.lr_scale_factor
        self.cfg.lr_pre_warmup *= self.lr_scale_factor

        self.warmup_ratio = self.cfg.lr_init - self.cfg.lr_pre_warmup
        self.warmup_offset = self.cfg.lr_pre_warmup
        self.warmup_offset_iters = self.cfg.warmup_iters + self.cfg.opt_start_iter
        self.decay_iters = (
            self.cfg.total_iters - self.cfg.warmup_iters - self.cfg.opt_start_iter
            if self.cfg.total_iters is not None else 1
        )
        assert self.decay_iters > 0,\
            "total_iters must be >= (warmup_iters + opt_start_iter) or None"

    def compute_lr(self, step: int) -> float:
        if step < self.cfg.opt_start_iter:
            return 0.0

        if step < (self.cfg.warmup_iters + self.cfg.opt_start_iter):
            _step = step - self.cfg.opt_start_iter
            match self.cfg.warmup_fn:
                case 'cosine':
                    warmup_coef = np.clip(_step / self.cfg.warmup_iters, 0, 1)
                    angle = np.pi * 0.5 * warmup_coef
                    return self.warmup_offset + self.warmup_ratio * np.sin(angle)
                case 'linear':
                    return (self.warmup_offset +
                            self.warmup_ratio * _step / self.cfg.warmup_iters)

        t = np.clip(
            (step - self.warmup_offset_iters) / self.decay_iters, 0, 1
        )
        return np.exp(np.log(self.cfg.lr_init) * (1 - t) + np.log(self.cfg.lr_final) * t)
