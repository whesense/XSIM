import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr
from xsimgs.structures import ROIBox3D

from ..common import OptimizationConfig, init_optimization, SceneNode
from xsim.modeling.gaussian.strategy import BasicStrategy
from xsim.modeling.scene import Scene



def blend(render, alpha, background, additive):
    if additive:
        return render + (1.0 - alpha[..., None]) * background
    else:
        return alpha[..., None] * render + (1.0 - alpha[..., None]) * background


class EnvLightNode(SceneNode):
    @staticmethod
    def create(sim_ds, init, **kwargs):
        return EnvLightNode(roi=sim_ds.roi, **kwargs)

    def __init__(
            self,
            roi: ROIBox3D,
            resolution: int = 128,
            max_resolution: int = 1024,
            increase_resolution_steps: int = 3000,
            randomize_background: bool = False,
            additive: bool = True,
            opt_cfg: OptimizationConfig = None,
            losses=[],
    ):
        super().__init__(roi=roi, losses=losses)
        self.max_resolution = max_resolution
        self.increase_resolution_steps = increase_resolution_steps
        self.opt_cfg = opt_cfg
        self.params = torch.nn.ParameterDict({
            'env': torch.nn.Parameter(
                torch.ones(1, 6, resolution, resolution, 3) * 0.5,
                requires_grad=True
            )
        })
        self.additive = additive
        self.randomize_background = randomize_background
        self.register_load_state_dict_pre_hook(self.load_hook)

    @staticmethod
    def load_hook(module, state_dict, *args, **kwargs):
        resolution = None
        for key in state_dict:
            if key.endswith('env'):
                resolution = state_dict[key].shape[2]

        module.params['env'] = torch.nn.Parameter(
            torch.zeros(1, 6, resolution, resolution, 3,
                        device=module.params['env'].device)
        )

    @property
    def resolution(self):
        return self.params['env'].shape[2]

    def init_optimization(self):
        self.opt, self.schedulers = init_optimization(
            self.opt_cfg, self.params, self.resolve_lr_scale_factor,
            self.__class__.__name__
        )

    @staticmethod
    def upsample_fn(name: str, t: torch.Tensor) -> torch.Tensor:
        t = t.data[0].permute(0, 3, 1, 2)
        t = F.interpolate(t, scale_factor=2, mode='bilinear')
        t = t.permute(0, 2, 3, 1)[None]
        return t

    def optimizer_step(self, scene: Scene):
        super().optimizer_step(scene)

        if (self.step % self.increase_resolution_steps == 0) and (
                self.resolution < self.max_resolution):
            BasicStrategy._update_param_with_optimizer(
                self, self.opt, self.upsample_fn, self.upsample_fn, names=["env"]
            )

    def forward(self, scene: Scene):
        for render in scene.renders:
            if render.result.color is not None:
                # world-space normalized directions cached by the render node
                ray_dirs = render.result.info['ray_directions']
                ray_dirs = ray_dirs.float()[None].contiguous().to(
                    render.result.color.device)
                env = dr.texture(
                    self.params['env'], ray_dirs,
                    filter_mode='linear', boundary_mode='cube'
                )
                if self.randomize_background:
                    env = torch.rand_like(env)
                render.result.color = blend(
                    render.result.color,
                    render.result.alpha,
                    env[0], self.additive)
