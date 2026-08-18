# Contains modified code from gsplat (https://github.com/nerfstudio-project/gsplat,
# Apache-2.0, Copyright 2024-2025 the Regents of the University of California,
# Nerfstudio Team and contributors) and 3DGUT (https://github.com/nv-tlabs/3dgrut,
# Apache-2.0, Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES).
# See THIRD_PARTY_LICENSES.md.

from typing import Callable, Union, Optional

import torch
import torch.nn as nn
from toast import quat_apply

from ..model import GaussianField as GF


def check_step_condition(step: int, start: int, end: int, freq: int) -> bool:
    """Checks if an operation should occur for the given step."""
    return (0 <= start < step) and (step < end or end == -1) and step % freq == 0


def make_param(new_t, t):
    new_t = new_t.detach()
    if isinstance(t, torch.nn.Parameter):
        new_t = torch.nn.Parameter(new_t, requires_grad=t.requires_grad)
    if isinstance(t, torch.nn.Buffer):
        new_t = torch.nn.Buffer(new_t)

    return new_t


class BasicStrategy(nn.Module):
    _prune_callback: Optional[Callable]

    density_params: set[str] = {GF.density, "lidar_density"}

    def __init__(
            self,
            scene_extent: float = 1.0,
            config = None,
            verbose: bool = False
    ):
        super().__init__()
        self.config = config
        self._step = 0
        self.scene_extent = scene_extent
        self.verbose = verbose
        self._prune_callback = None
        self.uses_render_info = False
        self._warned_empty = False

    def __repr__(self):
        return repr(self.config)

    @property
    def step(self):
        return self._step

    def init(self, model) -> "BasicStrategy":
        self._step = 0
        self._warned_empty = False
        return self

    def log_particle_change(self, model, action: str, n_changed: int):
        """One densification stat line: "<action> <n_changed> / <n_before>".

        ``n_before`` can be zero -- a model whose particles have all been pruned
        away -- so the share is only reported when there is something to take a
        share of, rather than dividing by it.
        """
        if not self.verbose:
            return

        n_before = model.num_particles
        share = f" ({n_changed / n_before * 100:.2f}%)" if n_before else ""
        print('[Step {}] {}'.format(self._step, model.__class__.__name__),
              f"{action} {n_changed} / {n_before}{share} gaussians")

    def model_is_empty(self, model) -> bool:
        """True once a model has no particles left, warning the first time.

        Densification has nothing to work on then, and a node that renders
        nothing feeds 0/0 into the losses -- which is how a run reaches
        ``loss=nan`` a couple of steps after its last particle goes. Emptying a
        model is a modelling problem (an instance pruned out of existence), so
        say it once instead of failing silently.
        """
        if model.num_particles:
            return False

        if not self._warned_empty:
            self._warned_empty = True
            print('[Step {}] {}'.format(self._step, model.__class__.__name__),
                  'has no particles left -- skipping densification. Its renders '
                  'are empty from here on, which will show up as nan losses.')

        return True

    def set_prune_callback(self, fn):
        self._prune_callback = fn

    def modify_prune_mask(self, prune_mask):
        if self._prune_callback is not None:
            prune_mask = prune_mask | self._prune_callback()
        return prune_mask

    def modify_prune_valid_mask(self, valid_mask):
        if self._prune_callback is not None:
            valid_mask = valid_mask & (~self._prune_callback())
        return valid_mask

    def pre_backward(self, model, opt: torch.optim.Optimizer, batch, render_info):
        pass

    def post_backward(self, model, opt: torch.optim.Optimizer, batch, render_info):
        pass

    def post_optimizer_step(self, model, opt: torch.optim.Optimizer, batch, render_info):
        self._step += 1

    def clone_by_mask(self, model, opt, mask: torch.Tensor):
        self.log_particle_change(model, 'Cloned', len(mask))

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.cat([param, param[mask]])

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            # print('clone_by_mask update optimizer fn. key={} v={} mask={}'.format(
            #     key, v.shape, mask.shape
            # ))
            return torch.cat(
                [v, torch.zeros((len(mask), *v.shape[1:]), device=v.device)])

        self._update_param_with_optimizer(model, opt, update_param_fn, update_optimizer_fn)

    def split_by_mask(
            self,
            model,
            opt,
            split_indices,
            remain_indices,
            split_n_gaussians: int = 2
    ):
        stds = model.scale[split_indices].repeat(split_n_gaussians, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = model.rotation[split_indices].repeat(split_n_gaussians, 1)
        offsets = quat_apply(rots, samples)

        self.log_particle_change(model, 'Splitted', len(split_indices))

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            repeats = [split_n_gaussians] + [1] * (param.dim() - 1)
            if name == "positions":
                p_split = param[split_indices].repeat(repeats) + offsets  # [2N, 3]
            elif name == "scale":
                scale_act = model.param_act('scale')
                scale_inv_act = model.param_inv_act('scale')

                p_split = scale_inv_act(
                    scale_act(param[split_indices].repeat(repeats)) / (0.8 * split_n_gaussians)
                )
            else:
                p_split = param[split_indices].repeat(repeats)

            return torch.cat([param[remain_indices], p_split])

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            v_split = torch.zeros(
                (split_n_gaussians * len(split_indices), *v.shape[1:]), device=v.device
            )
            return torch.cat([v[remain_indices], v_split])

        self._update_param_with_optimizer(model, opt, update_param_fn, update_optimizer_fn)

    def prune_by_mask(self, model, opt, valid_mask):
        self.log_particle_change(
            model, 'Pruned', model.num_particles - len(valid_mask))

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return param[valid_mask]

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[valid_mask]

        self._update_param_with_optimizer(model, opt, update_param_fn, update_optimizer_fn)

    def reset_density(self, model, opt, value: float):
        if self.verbose:
            print('[Step {}] {}'.format( self._step, model.__class__.__name__),
                  'density reset')
        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            assert name in self.density_params, \
                "wrong parameter passed to update_param_fn"
            return param.clamp(max=value)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(v)

        density_params = list(self.density_params.intersection(set(model.params.keys())))
        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(
            model, opt, update_param_fn, update_optimizer_fn, names=density_params
        )

    @staticmethod
    @torch.no_grad()
    def _update_param_with_optimizer(
            model,
            optimizer: torch.optim.Optimizer,
            update_param_fn: Callable[[str, torch.Tensor], torch.Tensor] | None,
            update_optimizer_fn: Callable[[str, torch.Tensor], torch.Tensor] | None,
            names: Union[list[str], None] = None,
    ) -> None:
        """Update the parameters and the state in the optimizers
        using the provided lambda functions.

        Args:
            optimizer: torch.optim.Optimizer
            update_param_fn: A function that takes the name of the parameter
                and the parameter itself, and returns the new parameter.
            update_optimizer_fn: A function that takes the key of the optimizer state
                and the state value, and returns the new state value.
            names: A list of key names to update. If None, update all. Default: None.
        """
        updated_names = []

        for i, param_group in enumerate(optimizer.param_groups):
            name = param_group["name"]

            if name not in model.params:
                continue

            if (names is None) or (name in names):
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        v = p_state[key]
                        if update_optimizer_fn is not None:
                            p_state[key] = update_optimizer_fn(key, v)

                if update_param_fn is not None:
                    p_new = make_param(update_param_fn(name, p), p)
                    optimizer.param_groups[i]["params"] = [p_new]
                    optimizer.state[p_new] = p_state
                    model.params[name] = p_new

                updated_names.append(name)

        if (update_param_fn is not None) and hasattr(model, 'param_buffers'):
            for name in model.param_buffers.keys():
                if (name not in updated_names) and ((names is None) or (name in names)):
                    p = model.param_buffers[name]
                    p_new = make_param(update_param_fn(name, p), p)
                    setattr(model.param_buffers, name, p_new)
