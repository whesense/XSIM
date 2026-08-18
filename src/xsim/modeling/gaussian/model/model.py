import copy
from dataclasses import replace
from typing import Optional, Callable

import torch

from .activated_model import ActivatedGaussians
from .init_default import (
    init_positions,
    init_scale,
    init_rotation,
    init_density,
    init_features_albedo,
    init_features_specular,
    GaussianInitCfg
)
from .specs import ParamSpec, SpecMetaClass
from .fields import GaussianField as GF
from .activations import inv_sigmoid, identity
from toast import quat_std

from xsim.modeling.scene import Scene
from xsim.utils.buffer_dict import BufferDict


def load_state_dict_pre_hook(
        module: "GaussianModel",
        state_dict: dict[str, torch.Tensor],
        *args, **kwargs
):
    """Resize params/buffers to match the checkpoint's particle count"""
    gs_params = {
        k: v for k, v in state_dict.items()
        if ('.params.' in k) or ('.param_buffers.' in k)
    }
    num_particles = next(iter(gs_params.values())).shape[0]

    for name in module.params:
        p = module.params[name]
        module.params[name] = torch.nn.Parameter(
            torch.zeros(num_particles, *p.shape[1:], device=p.device, dtype=p.dtype),
            requires_grad=p.requires_grad,
        )
    for name in module.param_buffers.keys():
        p = module.param_buffers[name]
        setattr(module.param_buffers, name, torch.nn.Buffer(
            torch.zeros(num_particles, *p.shape[1:], device=p.device, dtype=p.dtype)
        ))


class GaussianModel(torch.nn.Module, metaclass=SpecMetaClass):
    num_specular_features: int = 0

    # Whether compute_colors() produces view-independent colors. When True the
    # inference container can compute features once; when False (e.g. spherical
    # harmonics) they must be recomputed per frame.
    features_are_static: bool = True

    param_specs = {
        GF.positions: ParamSpec(3),
        GF.rotation: ParamSpec(4, act=quat_std),
        GF.scale: ParamSpec(3, act=torch.exp, inv_act=torch.log),
        GF.density: ParamSpec(1, act=torch.sigmoid, inv_act=inv_sigmoid),
        GF.features_albedo: ParamSpec(3, ret=False),
        GF.features_specular: ParamSpec(
            size=lambda self: self.num_specular_features,
            ret=False,
        ),
    }

    param_init = {
        GF.positions: init_positions,
        GF.scale: init_scale,
        GF.rotation: init_rotation,
        GF.density: init_density,
        GF.features_albedo: init_features_albedo,
        GF.features_specular: init_features_specular,
    }
    InitCfg = GaussianInitCfg

    @property
    def specs(self) -> dict[str, ParamSpec]:
        return type(self)._param_specs

    def __init__(
            self,
            params: dict[str, torch.Tensor],
            noopt_params: Optional[list[str]] = None,
            **kwargs,
    ):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

        noopt = set(noopt_params or []).union(
            set([k for k, spec in self.specs.items() if spec.non_optimizable])
        )

        self.params = torch.nn.ParameterDict({
            name: torch.nn.Parameter(params[name], requires_grad=True)
            for name in params if name not in noopt
        })
        self.param_buffers = BufferDict({
            name: torch.nn.Buffer(params[name])
            for name in params if name in noopt
        })
        self.register_load_state_dict_pre_hook(load_state_dict_pre_hook)

    def get_param_value(self, name: str) -> torch.Tensor:
        return self.params[name] if name in self.params else self.param_buffers[name]

    def on_optimizer_step(self, step: int):
        pass

    def __getattr__(self, name):
        specs = self.specs
        if name in specs:
            return specs[name].act(self.get_param_value(name))

        return super().__getattr__(name)

    @property
    def device(self) -> torch.device:
        return self.get_param_value(GF.positions).device

    @property
    def num_particles(self) -> int:
        return len(self.get_param_value(GF.positions))

    @property
    def color_dim(self) -> int:
        """Width of the per-particle color tensor returned by compute_colors."""
        return self.param_size(GF.features_albedo) + self.param_size(GF.features_specular)

    def param_size(self, name: str) -> int:
        return self._param_specs[name].resolve_size(self)

    def param_act(self, name: str) -> Callable:
        return self._param_specs[name].act

    def param_inv_act(self, name: str) -> Callable:
        return self._param_specs[name].inv_act

    def compute_colors(
            self,
            gaussians: ActivatedGaussians,
            scene: Scene
    ) -> torch.Tensor:
        return torch.cat([
            self.features_albedo,
            self.features_specular
        ], dim=-1)

    def apply_activations(self) -> "GaussianModel":
        new = copy.deepcopy(self)
        for key in list(new.params.keys()):
            new.params[key] = torch.nn.Parameter(getattr(self, key).detach())
        new.param_buffers = BufferDict({
            key: torch.nn.Buffer(getattr(self, key).detach())
            for key in self.param_buffers.keys()
        })
        new._param_specs = {
            name: replace(spec, act=identity, inv_act=identity)
            for name, spec in self._param_specs.items()
        }
        return new

    def apply_mask(self, mask: torch.Tensor) -> "GaussianModel":
        new = copy.deepcopy(self)
        for key in list(new.params.keys()):
            new.params[key] = torch.nn.Parameter(self.get_param_value(key).detach()[mask])
        new.param_buffers = BufferDict({
            key: torch.nn.Buffer(self.param_buffers[key].detach()[mask])
            for key in self.param_buffers.keys()
        })
        return new

    @staticmethod
    def _mask_to_local_ids(mask, positions) -> torch.Tensor:
        if isinstance(mask, torch.Tensor):
            assert mask.ndim == 1
            return torch.where(mask)[0] if mask.dtype == torch.bool else mask
        return torch.arange(len(positions), device=positions.device)[mask]

    def forward(
            self,
            mask: Optional[torch.Tensor] = None,
            exclude_keys: list[str] = []
    ) -> ActivatedGaussians:
        has_mask = mask is not None
        data: dict[str, torch.Tensor] = {}
        for name, spec in self._param_specs.items():
            if (not spec.ret) or (name in exclude_keys):
                continue
            raw = self.get_param_value(name)
            data[name] = spec.act(raw[mask] if has_mask else raw)
        if has_mask:
            data[GF.local_ids] = self._mask_to_local_ids(mask, data[GF.positions])
        return ActivatedGaussians(data, self._param_specs)

    def __repr__(self):
        lines = ["{}(".format(self.__class__.__name__)]
        for name in self.params:
            shape = " x ".join(map(str, self.params[name].shape))
            lines.append(f"    [opt] {name} ({shape})")
        for name in self.param_buffers.keys():
            shape = " x ".join(map(str, self.param_buffers[name].shape))
            lines.append(f"    {name} ({shape})")
        lines.append(")")
        return "\n".join(lines)
