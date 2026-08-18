from dataclasses import dataclass
import torch

from .model import InitContext, ParamSpec


class Instanced:
    @dataclass
    class InitCfg:
        reset_ids: bool = False

    @staticmethod
    def init_instance_ids(ctx: InitContext, config: InitCfg):
        keys = (
            list(range(len(ctx.instances)))
            if config.reset_ids
            else list(ctx.instances.keys())
        )
        return torch.cat([
            torch.full(
                (len(inst.points), 1), int(k),
                device=ctx.device, dtype=torch.int32
            ) for k, inst in zip(keys, ctx.instances.values())
        ], dim=0)

    param_specs = {
        "instance_ids": ParamSpec(1, ret=False, non_optimizable=True),
    }
    param_init = {
        "instance_ids": init_instance_ids,
    }
