from typing import Optional, Type

import torch
from xsimgs.structures import ROIBox3D

from xsim.data import SceneReconstructionDataset
from .gaussian_node import GaussianSceneNode
from xsim.modeling.scene import Scene
from ..common import OptimizationConfig, SceneNode
from ...gaussian import (
    init_from_instances, BasicStrategy, ActivatedGaussians, GaussianField as GF
)

from xsimgs.nodes import rigid_transform_and_put


class RigidSceneNode(GaussianSceneNode):
    @classmethod
    def create(
            cls,
            sim_ds: SceneReconstructionDataset,
            init: dict,

            model_type: type,
            init_configs: list = None,
            model_params: dict = None,
            return_velocity: bool = False,

            strategy_type: Optional[Type[BasicStrategy]] = None,
            strategy_cfg: dict = None,
            opt_cfg = None,
            fixed_params: list[str] = None,
            losses: list[torch.nn.Module] = None
    ):
        if len(init['rigid']) == 0:
            # No rigid objects in this scene, create dummy node
            return SceneNode.create(sim_ds, init)

        model_params_init = init_from_instances(
            model_type, init['rigid'], init_configs or []
        )
        model_params_init.update(model_params or {})

        return cls(
            roi=sim_ds.roi,
            instance_sizes=sim_ds.instance_size,
            model_type=model_type,
            model_params=model_params_init,
            return_velocity=return_velocity,
            strategy_type=strategy_type,
            strategy_cfg=strategy_cfg,
            opt_cfg=opt_cfg,
            fixed_params=fixed_params,
            losses=losses
        )

    def __init__(
            self,
            roi: ROIBox3D,
            instance_sizes: torch.Tensor,
            model_type: type,
            model_params: dict,
            cull_out_of_bounds: bool = True,
            return_velocity: bool = False,
            strategy_type: Optional[Type[BasicStrategy]] = None,
            strategy_cfg = None,
            opt_cfg: Optional[OptimizationConfig] = None,
            fixed_params: Optional[list[str]] = None,
            losses: list[torch.nn.Module] = []
    ):
        super().__init__(
            roi=roi,
            model_type=model_type,
            model_params=model_params,
            return_velocity=return_velocity,
            strategy_type=strategy_type,
            strategy_cfg=strategy_cfg,
            opt_cfg=opt_cfg,
            fixed_params=fixed_params,
            losses=losses
        )
        self.cull_out_of_bounds = cull_out_of_bounds
        self.register_buffer('instance_sizes', instance_sizes)
        if self.cull_out_of_bounds and self.strategy is not None:
            self.strategy.set_prune_callback(self.cull_out_of_box)

    def cull_out_of_box(self):
        inst_ids = self.model.instance_ids.flatten()
        positions = self.model.positions
        size = self.instance_sizes.mul(0.5)[inst_ids]
        out_mask = (positions < -size).any(dim=-1) | (positions > size).any(dim=-1)
        return out_mask

    def rigid_transform(
            self,
            positions: torch.Tensor,
            rotations: torch.Tensor,
            scene: Scene,
            container: Optional[ActivatedGaussians] = None,
            offset: int = 0,
    ):
        objects = scene.objects
        assert objects.mask is not None, "oriented boxes must have mask"

        return rigid_transform_and_put(
            positions=positions,
            rotations=rotations,
            instance_id=self.model.get_param_value("instance_ids"),
            inst_q=objects.pose.q.data,
            inst_t=objects.pose.t,
            inst_v=objects.motion.linear_velocity,
            inst_w=objects.motion.angular_velocity,
            inst_mask=objects.mask,
            buffer_offset=offset,
            out_positions=container.data[GF.positions] if container else None,
            out_rotation=container.data[GF.rotation] if container else None,
            out_velocity=container.data.get(GF.velocity) if container else None,
            out_mask=container.data[GF.mask] if container else None,
        )

    def rigid_transform_train(self, result: ActivatedGaussians, scene: Scene) -> ActivatedGaussians:
        result.positions_local = result.data[GF.positions]
        positions, rotation, velocity, mask = self.rigid_transform(
            result.data[GF.positions], result.data[GF.rotation], scene
        )
        result.data[GF.positions] = positions
        result.data[GF.rotation] = rotation
        if self.return_velocity:
            result.data[GF.velocity] = velocity
        result.data[GF.mask] = mask
        return result

    def forward_renderable(self, scene: Scene):
        result = self.forward_model(scene)
        result = self.rigid_transform_train(result, scene)
        scene.renderables.append(result)
        return result

    @torch.no_grad()
    def store_to_container(self, scene: Scene, offset: int):
        super().store_to_container(scene, offset)

        self.rigid_transform(
            self.model.positions,
            self.model.rotation,
            scene,
            container=scene.container,
            offset=offset,
        )
