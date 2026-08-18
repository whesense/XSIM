from dataclasses import dataclass
from typing import Literal

import torch
from toast import quat_random

from xsim.structures.init_instance import knn_points
from .init_utils import InitContext
from .activations import inv_sigmoid


@dataclass
class GaussianInitCfg:
    opacity_init: float = 0.1
    scale_k_neighbours: int = 3
    scale_min: float = 2e-3
    scale_max: float = 100.0
    scale_factor: float = 1.0

    specular_init: Literal['zeros', 'randn'] = "zeros"
    specular_fixed_dim: int = 0


def scale_from_points(
        points: torch.Tensor,
        k: int = 3,
        scale_factor: float= 1.0,
        min_scale: float= 2e-3,
        max_scale: float= 100.0
):
    dists = knn_points(points[None], points[None], K=k + 1).dists[0, :, 1:]
    dists = dists.sqrt().mean(dim=-1).mul(scale_factor)
    return dists.clamp(min_scale, max_scale).log().view(-1, 1).repeat(1, 3)


def init_positions(ctx: InitContext, cfg: GaussianInitCfg) -> torch.Tensor:
    return torch.cat([inst.points for inst in ctx.instances.values()], dim=0)


def init_scale(ctx: InitContext, cfg: GaussianInitCfg) -> torch.Tensor:
    return torch.cat([
        scale_from_points(
            inst.points, cfg.scale_k_neighbours, cfg.scale_factor,
            cfg.scale_min, cfg.scale_max
        ) for inst in ctx.instances.values()
    ], dim=0)


def init_rotation(ctx: InitContext, cfg: GaussianInitCfg) -> torch.Tensor:
    return quat_random(ctx.num_particles, device=ctx.device)


def init_density(ctx: InitContext, cfg: GaussianInitCfg) -> torch.Tensor:
    logit = float(inv_sigmoid(torch.as_tensor(cfg.opacity_init)))
    return torch.full((ctx.num_particles, 1), logit, device=ctx.device)


def init_features_albedo(ctx: InitContext, cfg: GaussianInitCfg) -> torch.Tensor:
    return torch.cat([
        inst.colors.float().mul(1 / 255)
        for inst in ctx.instances.values()
    ], dim=0)


def init_features_specular(ctx: InitContext, cfg: GaussianInitCfg) -> torch.Tensor:
    assert cfg.specular_init in ['zeros', 'randn']
    fn = torch.zeros if cfg.specular_init == 'zeros' else torch.randn

    return fn(ctx.num_particles, cfg.specular_fixed_dim, device=ctx.device)
