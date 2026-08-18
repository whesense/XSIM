from dataclasses import dataclass
import torch

from .model import ParamSpec, inv_sigmoid, InitContext


class LidarDensity:
    field_name = "lidar_density"

    @dataclass
    class InitCfg:
        opacity_init: float = 0.1

    @staticmethod
    def init_lidar_density(ctx: InitContext, cfg: InitCfg) -> torch.Tensor:
        logit = float(inv_sigmoid(torch.as_tensor(cfg.opacity_init)))
        return torch.full((ctx.num_particles, 1), logit, device=ctx.device)

    param_specs = {
        field_name: ParamSpec(1, act=torch.sigmoid, inv_act=inv_sigmoid),
    }
    param_init = {field_name: init_lidar_density}


class SkyLogit:
    field_name = "sky_logit"

    @dataclass
    class InitCfg:
        init_value: float = 0.5

    @staticmethod
    def init_sky_logit(ctx: InitContext, cfg: InitCfg) -> torch.Tensor:
        logit = float(inv_sigmoid(torch.as_tensor(cfg.init_value)))
        return torch.full((ctx.num_particles, 1), logit, device=ctx.device)

    param_specs = {
        field_name: ParamSpec(1, act=torch.sigmoid, inv_act=inv_sigmoid),
    }
    param_init = {field_name: init_sky_logit}

