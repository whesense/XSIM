from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from xsim.data.loaders import SensorType
from xsim.modeling.scene import Scene

from xsimgs.rendering.sh import (
    sh_degree_to_specular_dim, spherical_harmonics_fused, rgb_to_sh0
)

from .activated_model import ActivatedGaussians
from xsim.modeling.gaussian.model import GaussianModel
from .specs import ParamSpec
from .fields import GaussianField as GF
from .init_utils import InitContext


def sh_resolve(self):
    return sh_degree_to_specular_dim(self.cur_sh_degree)


def specular_dim_to_sh_degree(dim: int) -> int:
    """Inverse of sh_degree_to_specular_dim: dim == 3 * ((deg + 1) ** 2 - 1)."""
    bands = dim // 3 + 1
    return int(round(bands ** 0.5)) - 1



@dataclass
class SHScheduler:
    init_sh_degree: int = 0

    # Filled from the model's allocated features_specular width.
    # Leave as None to inherit the model's max degree.
    max_sh_degree: Optional[int] = None

    # Will increase SH degree every N iterations
    sh_incr_interval: int = 1000
    # If list of iterations is defined, SH degree will be increased at these specific
    # iterations
    sh_steps: list[int] = None

    def cur_degree(self, step: int) -> int:
        if self.sh_steps is None:
            return min(
                self.init_sh_degree + step // self.sh_incr_interval,
                self.max_sh_degree
            )
        else:
            return min(
                self.init_sh_degree + sum(map(lambda x: x <= step, self.sh_steps)),
                self.max_sh_degree
            )



class SphericalHarmonicsGaussianModel(GaussianModel):
    @dataclass
    class InitCfg:
        max_sh_degree: int = 3

    @staticmethod
    def init_features_albedo(ctx: InitContext, cfg: InitCfg) -> torch.Tensor:
        return torch.cat([
            rgb_to_sh0(inst.colors.float().mul(1 / 255))
            for inst in ctx.instances.values()
        ], dim=0)

    @staticmethod
    def init_specular(ctx: InitContext, cfg: InitCfg):
        return torch.zeros(
            ctx.num_particles, sh_degree_to_specular_dim(cfg.max_sh_degree),
            device=ctx.device
        )

    # SH colors are view-dependent (depend on the camera origin), so they must
    # be recomputed every frame rather than cached once in the container.
    features_are_static: bool = False

    param_specs = {
        GF.features_specular: ParamSpec(size=sh_resolve, ret=False),
    }
    param_init = {
        GF.features_albedo: init_features_albedo,
        GF.features_specular: init_specular
    }

    @property
    def color_dim(self) -> int:
        return 3

    def __init__(
            self,
            params,
            noopt_params=None,
            sh_scheduler: Optional[SHScheduler | dict] = None,
            **kwargs
    ):
        super().__init__(params, noopt_params=noopt_params, **kwargs)

        if isinstance(sh_scheduler, dict):
            sh_scheduler = SHScheduler(**sh_scheduler)
        self.sh_scheduler = sh_scheduler or SHScheduler()
        self.sh_scheduler.max_sh_degree = self.max_sh_degree

        self.register_buffer(
            "_cur_sh_degree",
            torch.tensor(self.sh_scheduler.init_sh_degree, dtype=torch.int,
                         device=self.device)
        )

    @property
    def cur_sh_degree(self) -> int:
        return int(self._cur_sh_degree)

    @cur_sh_degree.setter
    def cur_sh_degree(self, value: int):
        self._cur_sh_degree.fill_(int(value))

    @property
    def num_specular_features(self) -> int:
        return sh_degree_to_specular_dim(self.cur_sh_degree)

    @property
    def max_sh_degree(self) -> int:
        return specular_dim_to_sh_degree(
            self.get_param_value(GF.features_specular).shape[1]
        )

    @torch.no_grad()
    def increase_specular_degree(self):
        self.cur_sh_degree = min(self.cur_sh_degree + 1, self.max_sh_degree)

    def on_optimizer_step(self, step: int):
        new_value = self.sh_scheduler.cur_degree(step)
        if new_value != self.cur_sh_degree:
            print('[{}][SHScheduler] Updating SH degree from {} to {}'.format(
                self.__class__.__name__, self.cur_sh_degree, new_value
            ))
        self.cur_sh_degree = new_value

    def compute_colors(
            self,
            gaussians: ActivatedGaussians,
            scene: Scene
    ) -> torch.Tensor:
        if "sh_origin" not in scene.render_data:
            origin = torch.stack([
                render.camera.world_se3_camera.pose.t
                for render in scene.renders
                if render.sensor_type == SensorType.CAMERA
            ], dim=0).mean(dim=0)
            scene.render_data["sh_origin"] = origin

        return spherical_harmonics_fused(
            coeffs_dc=self.params[GF.features_albedo],
            coeffs=self.params[GF.features_specular].view(self.num_particles, -1, 3),
            positions=self.params[GF.positions],
            origin=scene.render_data["sh_origin"],
            degree_to_use=self.cur_sh_degree,
            layout_3kn=False
        )
