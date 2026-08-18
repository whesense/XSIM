from typing import Type, Optional

import torch

from xsimgs.structures import ROIBox3D

from xsim.data.loaders import SensorType
from xsim.modeling.scene import Scene
from ..common import (
    SceneNode,
    BasicScheduler, init_optimization, OptimizationConfig, LossesType
)
from xsim.modeling.gaussian import (
    GaussianModel, ActivatedGaussians, BasicStrategy, GaussianField as GF
)


class GaussianSceneNode(SceneNode):
    model: GaussianModel
    strategy: Optional[BasicStrategy]
    opt: torch.optim.Optimizer = None
    opt_cfg: OptimizationConfig = None
    schedulers: list[BasicScheduler] = []

    def __init__(
            self,
            roi: ROIBox3D,
            model_type: type,
            model_params: dict,
            return_velocity: bool = False,
            strategy_type: Optional[Type[BasicStrategy]] = None,
            strategy_cfg = None,
            opt_cfg: Optional[OptimizationConfig] = None,
            fixed_params: Optional[list[str]] = None,
            losses: LossesType = None
    ):
        super().__init__(roi=roi, losses=losses)
        self.model = model_type(**model_params, noopt_params=fixed_params)
        self.return_velocity = return_velocity

        strategy_cfg = strategy_cfg or {}
        scene_extent = strategy_cfg.pop('scene_extent', 'scene_radius')
        self.strategy = strategy_type(
            scene_extent=self.resolve_lr_scale_factor(scene_extent),
            **strategy_cfg
        ).init(self.model) if strategy_type is not None else None

        self.opt_cfg = opt_cfg or OptimizationConfig()

    @property
    def num_particles(self) -> int:
        return len(next(iter(self.model.params.values())))

    @property
    def is_empty(self) -> bool:
        """True once the model has no particles left.

        Densification can prune a node down to nothing -- a small instanced
        model whose object stops being observed, say. Everything this node does
        then works on zero-row tensors: an empty renderable, a regularizer whose
        mean over no particles is nan, densification with nothing to densify.
        So an empty node renders nothing, regularizes nothing and densifies
        nothing until it is given particles again.
        """
        return self.num_particles == 0

    def params(self):
        return {
            name: self.model.params[name]
            for name in self.model.params
            if isinstance(self.model.params[name], torch.nn.Parameter)
        }

    def init_optimization(self):
        self.opt, self.schedulers = init_optimization(
            self.opt_cfg, self.params(), self.resolve_lr_scale_factor,
            self.__class__.__name__
        )

    @staticmethod
    def color_render(scene: Scene):
        """The render whose gradients drive densification -- the first color
        (camera) render, falling back to the first render if there is none.

        Densification weights position gradients by camera distance
        (``Eval3DStrategy.compute_grad_value``), so it must read a camera view,
        not whichever render (e.g. a lidar sweep) happens to sit at index 0.
        """
        for render in scene.renders:
            if render.sensor_type == SensorType.CAMERA:
                return render
        return scene.renders[0]

    def compute_losses(self, scene: Scene, fwd_result) -> dict[str, torch.Tensor]:
        # An empty node rendered nothing, so ``fwd_result`` is None and the
        # per-particle regularizers have no particles to average over -- which is
        # where a nan enters the total loss and poisons every other node.
        if self.is_empty:
            return {}

        return super().compute_losses(scene, fwd_result)

    def pre_backward(self, scene: Scene, output=None):
        if self.is_empty:
            return

        if self.strategy is not None:
            render = self.color_render(scene)
            self.strategy.pre_backward(
                model=self.model, opt=self.opt,
                batch=render, render_info=render.result.info
            )

    def post_backward(self, scene: Scene, output=None):
        if self.is_empty:
            return

        if self.strategy is not None:
            render = self.color_render(scene)
            self.strategy.post_backward(
                model=self.model, opt=self.opt,
                batch=render, render_info=render.result.info
            )

    def optimizer_step(self, scene: Scene):
        super().optimizer_step(scene)
        self.model.on_optimizer_step(int(self.step))

        if self.strategy is not None:
            render = self.color_render(scene)
            self.strategy.post_optimizer_step(
                model=self.model, opt=self.opt,
                batch=render, render_info=render.result.info
            )

    def forward_model(self, scene: Scene) -> ActivatedGaussians:
        result: ActivatedGaussians = self.model.forward()
        result.data[GF.features] = self.model.compute_colors(result, scene)
        return result

    def forward(self, scene: Scene):
        """No-op for an empty node, otherwise :meth:`forward_renderable`.

        Returning None appends no renderable, which ``GaussianContainer`` already
        reads as "this node occupies no slice of the container".
        """
        if self.is_empty:
            return None

        return self.forward_renderable(scene)

    def forward_renderable(self, scene: Scene) -> ActivatedGaussians:
        """Build this node's renderable, append it to ``scene.renderables`` and
        return it. Subclasses override this rather than :meth:`forward`, so the
        empty-node check stays in one place."""
        result = self.forward_model(scene)

        if self.return_velocity:
            result.data[GF.velocity] = torch.zeros_like(result.positions)
        scene.renderables.append(result)
        return result

    def optimize_inference(self):
        super().optimize_inference()
        self.model = self.model.apply_activations()
        print('Preactivated model with {:,d} particles'.format(self.model.num_particles))

    def container_slice(self, offset: int) -> slice:
        return slice(offset, offset + self.num_particles)

    @torch.no_grad()
    def populate_container(self, container: ActivatedGaussians, offset: int):
        cur_slice = self.container_slice(offset)
        data = container.data
        for key, spec in self.model.specs.items():
            if spec.ret:
                data[key][cur_slice] = getattr(self.model, key)
        data[GF.mask][cur_slice] = True
        if self.model.features_are_static:
            colors = self.model.compute_colors(None, None)
            data[GF.features][cur_slice, :colors.shape[-1]] = colors

    @torch.no_grad()
    def store_to_container(self, scene: Scene, offset: int):
        if self.model.features_are_static:
            return

        cur_slice = self.container_slice(offset)
        colors = self.model.compute_colors(None, scene)
        scene.container.data[GF.features][cur_slice, :colors.shape[-1]] = colors
