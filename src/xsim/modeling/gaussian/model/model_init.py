import torch

from .init_utils import InitContext, build_params
from xsim.structures import SceneInitInstance


def device_of(instances: dict[int, SceneInitInstance]) -> torch.device:
    return next(iter(instances.values())).points.device


def init_params(model_cls, ctx: InitContext, configs) -> dict:
    out = dict(params=build_params(model_cls, ctx, configs))
    return out


def init_from_instances(
        model_cls,
        instances: dict[int, SceneInitInstance],
        configs=None,
        device=None
) -> dict:
    ctx = InitContext(instances=instances, device=device or device_of(instances))
    return init_params(model_cls, ctx, configs)


def init_from_instance(
        model_cls,
        instance: SceneInitInstance,
        configs=None,
        device=None
) -> dict:
    return init_from_instances(model_cls, {-1: instance}, configs, device)


def random_initialization(
        model_cls,
        num_particles: int = 1_000,
        init_size: float = 3.0,
        configs=None,
        device: str = "cuda"
) -> dict:
    points = torch.rand(num_particles, 3, device=device) * (2 * init_size) - init_size
    colors = torch.rand(num_particles, 3, device=device).mul(255).byte()
    instance = SceneInitInstance(points, colors)
    return init_from_instance(model_cls, instance, configs, device)
