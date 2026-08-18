from dataclasses import dataclass, asdict
from typing import Optional, Type

import torch
import torch.nn.functional as F

from xsimgs.structures import ROIBox3D

from xsim.data import SceneReconstructionDataset
from xsim.modeling.scene import Scene
from ..common import OptimizationConfig, SceneNode
from ...gaussian import (
    init_from_instances, BasicStrategy, ActivatedGaussians, GaussianField as GF
)
from .utils import ConditionalDeformNetwork
from .rigid_node import RigidSceneNode



@dataclass
class ConditionalDeformNetworkConfig:
    D: int = 8
    W: int = 256
    input_ch: int = 3
    embed_dim: int = 10
    x_multires: int = 10
    t_multires: int = 10
    deform_quat: bool = True
    deform_scale: bool = True


class DeformableSceneNode(RigidSceneNode):
    @classmethod
    def create(
            cls,
            sim_ds: SceneReconstructionDataset,
            init: dict,

            model_type: type,
            init_configs: list = None,
            model_params: dict = None,
            return_velocity: bool = False,
            deform_after_iteration: int = None,
            deformable_cfg: dict = None,

            strategy_type: Optional[Type[BasicStrategy]] = None,
            strategy_cfg: dict = None,
            opt_cfg = None,
            fixed_params: list[str] = None,
            losses: list[torch.nn.Module] = None
    ):
        if len(init['deformable']) == 0:
            # No deformable objects in this scene, create dummy node
            return SceneNode.create(sim_ds, init)

        model_params_init = init_from_instances(
            model_type, init['deformable'], init_configs or []
        )
        model_params_init.update(model_params or {})

        deformable_cfg = ConditionalDeformNetworkConfig(**deformable_cfg) \
            if deformable_cfg is not None else ConditionalDeformNetworkConfig()

        return cls(
            roi=sim_ds.roi,
            instance_sizes=sim_ds.instance_size,
            deformable_cfg=deformable_cfg,
            deform_after_iteration=deform_after_iteration,
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
            deformable_cfg: ConditionalDeformNetworkConfig,
            deform_after_iteration: int = None,
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
            instance_sizes=instance_sizes,
            cull_out_of_bounds=cull_out_of_bounds,
            model_params=model_params,
            return_velocity=return_velocity,
            strategy_type=strategy_type,
            strategy_cfg=strategy_cfg,
            opt_cfg=opt_cfg,
            fixed_params=fixed_params,
            losses=losses
        )

        self.deform_after_iteration = deform_after_iteration
        self.deform_network = ConditionalDeformNetwork(
            **asdict(deformable_cfg)
            if not isinstance(deformable_cfg, dict)
            else deformable_cfg
        )
        self.inst_embeds = torch.nn.Parameter(
            torch.rand(len(instance_sizes),
                       deformable_cfg.embed_dim,
                       device=instance_sizes.device),
            requires_grad=True
        )

    def params(self):
        return {
            **super().params(),
            'embeddings': self.inst_embeds,
            'network': self.deform_network,
        }

    @property
    def should_deform(self) -> bool:
        return ((self.deform_after_iteration is not None)
                and (self.step > self.deform_after_iteration))

    def deform(
            self,
            result: ActivatedGaussians,
            scene: Scene
    ):
        if not self.should_deform:
            return result

        masked_ids = self.model.get_param_value("instance_ids")
        embeds = self.inst_embeds[masked_ids]
        inst_height = self.instance_sizes[masked_ids, 2:3]
        x = result.positions.detach() / inst_height * 2
        t = scene.time_bounds.normalize_time(scene.world_time).expand_as(inst_height)
        delta_xyz, delta_quat, delta_scale = self.deform_network(x, t, embeds)
        result.data[GF.positions] = result.positions + delta_xyz
        if delta_quat is not None:
            result.data[GF.rotation] = F.normalize(result.rotation + delta_quat, dim=-1)
        if delta_scale is not None:
            # TODO: reparameterize scale so that it does not become negative
            result.data[GF.scale] = result.scale + delta_scale
        return result

    def forward_renderable(self, scene: Scene):
        result = self.deform(self.forward_model(scene), scene)
        result = self.rigid_transform_train(result, scene)
        scene.renderables.append(result)
        return result

    @torch.no_grad()
    def store_to_container(self, scene: Scene, offset: int):
        super(RigidSceneNode, self).store_to_container(scene, offset)

        result = self.deform(self.model.forward(), scene)
        if self.should_deform:
            scene.container.data[GF.scale][self.container_slice(offset)] = result.scale

        self.rigid_transform(
            result.data[GF.positions],
            result.data[GF.rotation],
            scene,
            container=scene.container,
            offset=offset,
        )
