
from .scene_node import SceneNode, LossesType
from .node_collection import NodeCollection

from .optimization import (
    init_optimization, make_opt_cfg, OptimizationConfig, ParamOptConfig
)
from .schedulers import *


def sched_cfg(
        lr_init: float,
        lr_final: float,
        warmup: int = 500,
        total: int = 40_000,
        opt_start_iter: int = 0,
        **kwargs
):
    return ParamOptConfig(
        lr=0,
        scheduler_type=WarmupDecayScheduler,
        scheduler_cfg=dict(
            lr_init=lr_init,
            lr_final=lr_final,
            warmup_iters=warmup,
            total_iters=total,
            opt_start_iter=opt_start_iter
        ),
        **kwargs
    )
