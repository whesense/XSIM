# Contains modified code from 3DGUT (https://github.com/nv-tlabs/3dgrut,
# Apache-2.0, Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES).
# See THIRD_PARTY_LICENSES.md.

from dataclasses import dataclass

import numpy as np
import torch

from .base import BasicStrategy, check_step_condition
from ..model import GaussianModel


@dataclass
class Eval3DStrategyConfig:
    densify_prune_frequency: int = 100

    densify_start_iter: int = 500
    densify_end_iter: int = 15_000
    densify_grad_thr: float = 0.0002
    densify_size_thr: float = 0.01
    split_count: int = 2
    split_screen_size: float = 0.05
    pause_refine_after_reset: int = 1000
    densify_scale2d_stop_iter: int = 0

    prune_start_iter: int = 500
    prune_end_iter: int = 100_000
    prune_alpha_thr: float = 0.005
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15

    density_reset_freq: int = 3_000
    density_reset_start_iter: int = 0
    density_reset_end_iter: int = 15_000
    density_reset_value: float = 0.01


class Eval3DStrategy(BasicStrategy):
    config: Eval3DStrategyConfig
    uses_render_info: bool = True

    def __init__(
            self,
            scene_extent: float = 1.0,
            config: Eval3DStrategyConfig = Eval3DStrategyConfig(),
            verbose: bool = False
    ):
        super().__init__(
            scene_extent=scene_extent,
            config=Eval3DStrategyConfig(**config) if isinstance(config, dict) else config,
            verbose=verbose
        )
        # Accumulation of the norms of the positions gradients
        self.grad_sum = torch.nn.Buffer(torch.empty([0, 1]), persistent=False)
        self.grad_cnt = torch.nn.Buffer(torch.empty([0, 1]), persistent=False)
        self.radii_buffer = torch.nn.Buffer(torch.empty([0, ]), persistent=False)

        # Parameters related to densification, pruning and reset
        self.relative_size_threshold = self.config.densify_size_thr
        self.prune_density_threshold = self.config.prune_alpha_thr
        self.new_max_density = self.config.density_reset_value
        self.max_density_inv_act = None

    def init(self, model: GaussianModel):
        super().init(model)

        self.max_density_inv_act = model.param_inv_act('density')(
            torch.tensor(self.new_max_density)
        ).item()
        self.reset_state(model)
        return self

    def modify_state(self, op):
        for key in ['grad_sum', 'grad_cnt', 'radii_buffer']:
            setattr(self, key, op(getattr(self, key)))

    def reset_state(self, model: GaussianModel):
        num_particles = model.num_particles
        self.grad_sum = torch.nn.Buffer(torch.zeros(
            (num_particles, 1), dtype=torch.float32, device=model.device), persistent=False)
        self.grad_cnt = torch.nn.Buffer(torch.zeros(
            (num_particles, 1), dtype=torch.int32, device=model.device), persistent=False)
        self.radii_buffer = torch.nn.Buffer(torch.zeros(
            (num_particles, ), dtype=torch.float32, device=model.device
        ), persistent=False)

    @staticmethod
    def compute_grad_value(pos, grads3d, mask, camera, render_info):
        ray_o = camera.world_se3_camera.pose.t.to(grads3d.device)
        dists = (pos[mask] - ray_o).norm(dim=1, keepdim=True)
        return (grads3d[mask] * dists).norm(dim=-1, keepdim=True) * 0.5

    @torch.no_grad()
    @torch.cuda.nvtx.range("update-gradient-buffer")
    def update_gradient_buffer(
            self,
            model: GaussianModel,
            camera,
            render_info
    ):
        # render_info: width, height, radii, local_ids, global_mask
        stop_iter = self.config.densify_scale2d_stop_iter
        update_grads = self.step < self.config.densify_end_iter
        update_radii = (
                (stop_iter > 0) and (self.step < stop_iter + 1)
                and ('radii' in render_info)
        )

        if not (update_grads or update_radii):
            return

        global_mask = (
            render_info['global_mask'] if 'global_mask' in render_info else np.s_[:]
        )
        if global_mask is None:
            return
        radii: torch.Tensor = render_info["radii"]
        radii = radii[0][global_mask] if radii.ndim == 3 else radii[global_mask]
        result_ids = (radii > 0).all(dim=-1).nonzero().flatten()
        radii = radii[result_ids].amax(dim=-1)
        if 'local_ids' in render_info:
            local_ids = render_info['local_ids'][global_mask][result_ids]
        else:
            local_ids = result_ids

        if update_grads:
            params_grad = model.positions.grad
            mask = local_ids
            grad_value = self.compute_grad_value(
                model.positions, params_grad, mask, camera, render_info
            )

            self.grad_sum[mask] += grad_value
            self.grad_cnt[mask] += 1

        if update_radii:
            width, height = render_info['width'], render_info['height']
            self.radii_buffer[local_ids] = torch.maximum(
                self.radii_buffer[local_ids], radii / float(max(width, height)),
            )

    @torch.no_grad()
    def post_backward(
            self,
            model: GaussianModel,
            opt: torch.optim.Optimizer,
            batch,
            render_info
    ) -> bool:
        self.update_gradient_buffer(
            model=model,
            camera=batch.camera,
            render_info=render_info
        )
        return False

    def check_step(self, start_iter, end_iter, freq):
        return check_step_condition(self.step, start_iter, end_iter, freq)

    @torch.no_grad()
    def post_optimizer_step(
        self,
        model: GaussianModel,
        opt: torch.optim.Optimizer,
        batch,
        render_info
    ) -> bool:
        """Callback function to be executed after the `loss.backward()` call."""
        scene_updated = False

        # Nothing to clone, split, prune or reset once the model is empty -- and
        # every one of those would index into zero-row tensors.
        if self.model_is_empty(model):
            super().post_optimizer_step(model, opt, batch, render_info)
            return scene_updated

        if (self.step % self.config.density_reset_freq) > self.config.pause_refine_after_reset:
            reset_buffer = False
            if self.check_step(
                    self.config.densify_start_iter,
                    self.config.densify_end_iter,
                    self.config.densify_prune_frequency
            ):
                scene_updated |= self.densify(model=model, opt=opt)
                reset_buffer = True

            if self.check_step(
                    self.config.prune_start_iter,
                    self.config.prune_end_iter,
                    self.config.densify_prune_frequency
            ):
                scene_updated |= self.prune(model=model, opt=opt)
                reset_buffer = True

            if reset_buffer:
                self.reset_state(model)
                torch.cuda.empty_cache()

        if self.check_step(
                self.config.density_reset_start_iter,
                self.config.density_reset_end_iter,
                self.config.density_reset_freq
        ) and (self.step > 0):
            self.reset_density(model=model, opt=opt, value=self.max_density_inv_act)

        super().post_optimizer_step(model, opt, batch, render_info)
        return scene_updated

    @torch.cuda.nvtx.range("densify")
    def densify(self, model, opt):
        densify_grad_norm = (self.grad_sum / self.grad_cnt.clamp_min(1)).squeeze()
        grad_thr = self.config.densify_grad_thr

        scale_max = model.scale.amax(dim=1)
        grad_mask = densify_grad_norm >= grad_thr
        clone_mask = grad_mask & (scale_max <= self.relative_size_threshold * self.scene_extent)
        split_mask = grad_mask & (scale_max > self.relative_size_threshold * self.scene_extent)
        if self.step < self.config.densify_scale2d_stop_iter:
            split_mask |= self.radii_buffer > self.config.split_screen_size

        clone_indices = clone_mask.nonzero().flatten()
        n_clones = len(clone_indices)
        if n_clones > 0:
            self.clone_by_mask(model, opt, clone_indices)
            self.modify_state(lambda v: torch.cat([v, v[clone_indices]]))

        # split_mask was built at the pre-clone length; cloning grew the model by
        # n_clones. Extend it so the just-cloned particles count as "remain" (they
        # are never split candidates), otherwise the split rebuild drops them.
        split_mask = torch.cat([
            split_mask, torch.zeros(n_clones, dtype=torch.bool, device=split_mask.device)
        ])
        split_indices = split_mask.nonzero().flatten()
        remain_indices = (~split_mask).nonzero().flatten()
        n_split = len(split_indices)
        if n_split > 0:
            self.split_by_mask(model, opt, split_indices, remain_indices, split_n_gaussians=self.config.split_count)
            self.modify_state(
                lambda v: torch.cat((v[remain_indices], v[split_indices].repeat([2] + [1] * (v.dim() - 1))))
            )
        return (n_clones > 0) or (n_split > 0)

    def prune(self, model: GaussianModel, opt):
        if 'lidar_density' not in model.params:
            prune_mask = model.density.view(-1) < self.config.prune_alpha_thr
        else:
            prune_mask = torch.maximum(model.density.view(-1), model.lidar_density.view(-1)) < self.config.prune_alpha_thr

        if self.step > self.config.density_reset_freq:
            is_too_big = model.scale.amax(dim=-1) > self.config.prune_scale3d * self.scene_extent

            if self.step < self.config.densify_scale2d_stop_iter:
                is_too_big |= self.radii_buffer > self.config.prune_scale2d

            prune_mask = prune_mask | is_too_big

        prune_mask = self.modify_prune_mask(prune_mask)
        valid_mask = (~prune_mask).nonzero().flatten()
        self.prune_by_mask(model, opt, valid_mask)
        self.modify_state(lambda x: x[valid_mask])
        return True
