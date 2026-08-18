# Contains modified code from gsplat's MCMC strategy
# (https://github.com/nerfstudio-project/gsplat, Apache-2.0, Copyright 2024-2025
# the Regents of the University of California, Nerfstudio Team and contributors).
# See THIRD_PARTY_LICENSES.md.

import math
from dataclasses import dataclass

import torch

from xsimgs.rendering.mcmc import add_mcmc_noise

from .base import BasicStrategy, check_step_condition
from ..model import GaussianModel


@dataclass
class MCMCConfig:
    reloc_start_iter: int = 500
    reloc_end_iter: int = 20_000
    reloc_freq: int = 100

    spawn_start_iter: int = 500
    spawn_end_iter: int = 20_000
    spawn_freq: int = 100
    spawn_ratio: int = 0.05

    noise_start_iter: int = 0
    noise_end_iter: int = 35000
    noise_freq: int = 1
    noise_lr: float = 5e5

    binom_n_max: int = 51
    density_thr: float = 0.005
    max_num_particles: int = 5_000_000

    k: float = 100.0
    x0: float = 0.995


class MCMCStrategy(BasicStrategy):
    config: MCMCConfig

    """Densification and prunning strategy that follows the paper:

    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`

    MCMC Strategy interprets the training process of placing and optimizing Gaussians
    as a sampling process.

    Specifically, it periodically:
    - Moves "dead" Gaussians (low opacity) to the location of "live" Gaussians (high opacity).
    - Adds covariance dependent noise to the positions of the Gaussians.
    - Introduces new Gaussians sampled based on the opacity distribution.
    """

    def __init__(
            self,
            scene_extent: float = 1.0,
            config: MCMCConfig = MCMCConfig(),
            verbose: bool = False
    ):
        super().__init__(
            scene_extent=scene_extent,
            verbose=verbose,
            config=config if isinstance(config, MCMCConfig) else MCMCConfig(**config)
        )

        self.binoms = torch.nn.Buffer(torch.as_tensor(
            [
                [
                    math.comb(n, k) if k <= n else 0
                    for k in range(self.config.binom_n_max)
                ]
                for n in range(self.config.binom_n_max)
            ],
            dtype=torch.float32
        ))

    def check_step(self, start_iter, end_iter, freq):
        return check_step_condition(self.step, start_iter, end_iter, freq)

    @torch.no_grad()
    def post_optimizer_step(
            self,
            model: GaussianModel,
            opt: torch.optim.Optimizer,
            batch,
            render_info
    ):
        updated = False

        # Relocation and spawning both sample from the particles that are alive;
        # with none left there is nothing to sample from.
        if self.model_is_empty(model):
            super().post_optimizer_step(model, opt, batch, render_info)
            return updated

        # Relocate dead gaussians to the alive areas
        if self.check_step(
                self.config.reloc_start_iter,
                self.config.reloc_end_iter,
                self.config.reloc_freq
        ):
            updated = True
            self.relocate_particles(model, opt)

        # Add new Gaussians if the maximum number has not been reached
        if self.check_step(
                self.config.spawn_start_iter,
                self.config.spawn_end_iter,
                self.config.spawn_freq
        ):
            updated = True
            self.spawn_particles(model, opt)

        if self.verbose and updated:
            print('Number of particles: {:,d}'.format(model.num_particles))

        # Perturb the positions of the Gaussians
        if self.check_step(
                self.config.noise_start_iter,
                self.config.noise_end_iter,
                self.config.noise_freq
        ):
            self.add_noise(model, opt)

        super().post_optimizer_step(model, opt, batch, render_info)
        return True

    @torch.no_grad()
    def relocate_particles(
            self,
            model: GaussianModel,
            opt: torch.optim.Optimizer
    ):
        # Get the per Gaussian densities and scales (after sigmoid)
        densities: torch.Tensor = model.density
        # Find the dead indices
        dead_idxs = torch.where(densities <= self.config.density_thr)[0]
        alive_idxs = torch.where(densities > self.config.density_thr)[0]
        n_dead_gaussians = len(dead_idxs)

        if n_dead_gaussians > 0:
            sampled_idxs, new_densities, new_scales = self.sample_particles(
                model, n_dead_gaussians, alive_idxs
            )
            def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
                if name == "density":
                    param[sampled_idxs] = new_densities
                elif name == "scale":
                    param[sampled_idxs] = new_scales
                param[dead_idxs] = param[sampled_idxs]
                return param

            def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
                v[sampled_idxs] = 0
                return v

            self._update_param_with_optimizer(
                model, opt, update_param_fn, update_optimizer_fn
            )

        self.log_particle_change(model, 'Relocated', n_dead_gaussians)

    @torch.no_grad()
    def spawn_particles(
            self,
            model: GaussianModel,
            opt: torch.optim.Optimizer
    ):
        # Get the current number of gaussians
        num_particles = model.num_particles
        target_num_particles = min(
            self.config.max_num_particles,
            int((1 + self.config.spawn_ratio) * num_particles)
        )
        spawn_cnt = max(0, target_num_particles - num_particles)

        # Reported against the count before spawning, so log it before the update.
        self.log_particle_change(model, 'Added', spawn_cnt)

        if spawn_cnt > 0:
            sampled_idxs, new_densities, new_scales = self.sample_particles(model, spawn_cnt)

            def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
                if name == "density":
                    param[sampled_idxs] = new_densities
                elif name == "scale":
                    param[sampled_idxs] = new_scales
                param_new = torch.cat([param, param[sampled_idxs]])
                return param_new

            def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
                v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
                return torch.cat([v, v_new])

            self._update_param_with_optimizer(
                model, opt, update_param_fn, update_optimizer_fn
            )

    @torch.no_grad()
    def add_noise(
            self,
            model: GaussianModel,
            opt: torch.optim.Optimizer
    ):
        lr = list(filter(
            lambda p: p["name"] == "positions", opt.param_groups)
        )[0]["lr"]

        add_mcmc_noise(
            positions=model.get_param_value('positions'),
            density=model.get_param_value('density'),
            rotation=model.get_param_value('rotation'),
            scale=model.get_param_value('scale'),
            k=self.config.k,
            x0=self.config.x0,
            noise_scale=self.config.noise_lr * lr,
        )

    def sample_particles(
            self,
            model: GaussianModel,
            num_particles: int,
            valid_indices: torch.Tensor = None
    ):
        densities = model.density
        scales = model.scale

        if valid_indices is None:
            valid_indices = torch.arange(
                0, int(densities.shape[0]),
                device=densities.device,
                dtype=torch.int32
            )

        probabilities = densities[valid_indices].flatten()  # ensure its shape is [N,]

        # Sample the locations to which the dead Gaussians
        # will be moved proportional to the opacity of the alive Gaussians
        sampled_idxs = torch.multinomial(probabilities, num_particles, replacement=True)
        sampled_idxs = valid_indices[sampled_idxs]

        ratios = torch.bincount(sampled_idxs)[sampled_idxs] + 1
        ratios = ratios.clamp_(1, self.config.binom_n_max).int()

        from gsplat.relocation import compute_relocation as compute_relocation_tensor

        new_densities, new_scales = compute_relocation_tensor(
            densities[sampled_idxs].contiguous(),
            scales[sampled_idxs].contiguous(),
            ratios.contiguous(),
            self.binoms,
        )
        new_densities = new_densities.clamp(self.config.density_thr, 1.0 - 1e-6)
        new_densities = model.param_inv_act('density')(new_densities)
        new_scales = model.param_inv_act('scale')(new_scales)

        return sampled_idxs, new_densities, new_scales
