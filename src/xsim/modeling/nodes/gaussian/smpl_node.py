from typing import Optional

import torch
import torch.nn.functional as F
from toast import SE3

from xsimgs.structures import ROIBox3D
from xsimgs.nodes import (
    smpl_lbs_skinning,
    smpl_local_to_global_pose,
    interpolate_smpl_poses,
    valid_instances_indices
)

from xsim.data import SceneReconstructionDataset
from xsim.modeling.scene import Scene
from ..common import OptimizationConfig, SceneNode
from .gaussian_node import  GaussianSceneNode, GF
from .utils.smpl_node_utils import init_smpl_node, SMPLTemplate, smpl_knn


class SMPLSceneNode(GaussianSceneNode):
    @classmethod
    def create(
            cls,
            sim_ds: SceneReconstructionDataset,
            init: dict,

            model_type: type,
            init_configs: list = None,
            model_params: dict = None,

            return_velocity: bool = False,
            use_voxel_deformer: bool = True,

            opt_cfg = None,
            fixed_params: list[str] = None,
            losses: list = None
    ):
        if sim_ds.num_smpl_objects == 0:
            # if no SMPL objects defined, create dummy node
            return SceneNode.create(sim_ds, init)

        template, model_params, joint_quats = init_smpl_node(
            sim_ds, model_type, init_configs, device=sim_ds.device)
        model_params.update(model_params or {})
        instance_ids = torch.tensor(sorted(sim_ds.smpl_object_ids), dtype=torch.int32)

        return cls(
            roi=sim_ds.roi,
            model_type=model_type,
            model_params=model_params,

            template=template,
            joint_quats=joint_quats,
            joint_quats_time=sim_ds.instances.motion.pose_time[instance_ids],
            instance_ids=instance_ids,
            use_voxel_deformer=use_voxel_deformer,
            return_velocity=return_velocity,

            opt_cfg=opt_cfg,
            fixed_params=fixed_params,
            losses=losses
        )

    def __init__(
            self,
            roi: ROIBox3D,
            model_type: type,
            model_params: dict,

            template: SMPLTemplate,
            joint_quats: torch.Tensor,
            joint_quats_time: torch.Tensor,
            instance_ids: torch.Tensor,
            use_voxel_deformer: bool = True,
            return_velocity: bool = False,

            opt_cfg: Optional[OptimizationConfig] = None,
            fixed_params: Optional[list[str]] = None,
            losses = []
    ):
        super().__init__(
            roi=roi,
            model_type=model_type,
            model_params=model_params,
            return_velocity=return_velocity,

            opt_cfg=opt_cfg,
            fixed_params=fixed_params,
            losses=losses
        )
        self.use_voxel_deformer = use_voxel_deformer
        self.template = template
        self.joint_quats = torch.nn.Parameter(joint_quats)
        self.joint_quats_time = torch.nn.Buffer(joint_quats_time)
        self.joints = torch.nn.Buffer(self.template.J_canonical.contiguous())
        self.instance_ids = torch.nn.Buffer(instance_ids)

        a0_inv = SE3.from_matrix(self.template.A0_inv.contiguous())
        self.a0_inv_q = torch.nn.Buffer(a0_inv.q.data.contiguous())
        self.a0_inv_t = torch.nn.Buffer(a0_inv.t.contiguous())

        if not self.use_voxel_deformer:
            self.weights = torch.nn.Buffer(
                self.template.W.detach().permute(0, 2, 1).contiguous()
            )

        # non-persistent so it follows the module onto the GPU via .to()/.cuda()
        # (indexed on-device in the SMPL-KNN loss) and stays out of checkpoints.
        self.register_buffer('nn_ind', None, persistent=False)
        self.update_knn()

    def params(self):
        return {
            # TODO: add joint quats
            'w_dc_vox': self.template,
            **super().params()
        }

    @property
    def canonical_vertices(self):
        return self.model.positions.view(
            len(self.instance_ids), self.template.num_vertices, 3)

    def update_knn(self):
        self.nn_ind = smpl_knn(self.canonical_vertices)

    def forward_renderable(self, scene: Scene):
        assert scene.world_time is not None, "Scene has no world time"
        query_time = scene.world_time

        # print('joint_quats:', self.joint_quats.shape, self.joint_quats.is_contiguous(), self.joint_quats.device)
        # print('joint_quats_time:', self.joint_quats_time.shape, self.joint_quats_time.is_contiguous(), self.joint_quats_time.device)
        # print('root_q:', scene.objects.pose.q.data.shape, scene.objects.pose.q.data.is_contiguous(), scene.objects.pose.q.data.device)
        # print('instance_Ids:', self.instance_ids.shape, self.instance_ids.is_contiguous(), self.instance_ids.dtype, self.instance_ids.device)
        # print('query_time:', query_time.shape, query_time.is_contiguous(), query_time.device)
        # torch.cuda.synchronize()
        smpl_poses = interpolate_smpl_poses(
            smpl_poses=self.joint_quats,
            smpl_poses_time=self.joint_quats_time,
            root_q=scene.objects.pose.q.data,
            instance_ids=self.instance_ids,
            query_time=query_time
        )
        # torch.cuda.synchronize()
        # print('--- smpl_poses:', smpl_poses.shape, smpl_poses)
        cur_ids = valid_instances_indices(self.instance_ids, scene.objects.mask)
        # torch.cuda.synchronize()
        # print('cur_ids:', cur_ids.shape, cur_ids)

        # No human is valid at this time (they enter the sequence later, or have
        # already left). Every kernel below is sized from cur_ids, and launching
        # one with an empty grid raises a CUDA "invalid configuration argument"
        # -- asynchronously, so it surfaces at some unrelated call downstream.
        # Nothing to render either way, so skip the node for this scene.
        if len(cur_ids) == 0:
            return None

        # print('all valid finite:', smpl_poses[cur_ids].isfinite().all())

        global_q, global_t = smpl_local_to_global_pose(
            local_q_ids=cur_ids,
            local_q=smpl_poses,
            joints=self.joints,
            a0_inv_q=self.a0_inv_q,
            a0_inv_t=self.a0_inv_t,
        )
        # torch.cuda.synchronize()

        result = self.forward_model(scene)
        result.cur_ids = cur_ids
        canonical_vertices = result.data[GF.positions].view(len(self.instance_ids), -1, 3)
        device = canonical_vertices.device

        # torch.cuda.synchronize()

        if self.use_voxel_deformer:
            weights = self.template.voxel_deformer(canonical_vertices).contiguous()  # [N, 24, 6890]
        else:
            weights = self.weights

        n = len(result.data[GF.rotation])
        out_mask = torch.zeros(n, dtype=torch.bool, device=device)
        out_positions = torch.zeros(n, 3, device=device)
        out_rotation = torch.zeros(n, 4, device=device)
        out_velocity = torch.zeros(n, 3, device=device) if self.return_velocity else None

        out_positions, out_rotation, out_velocity, out_mask = smpl_lbs_skinning(
            valid_ids=cur_ids,
            global_q=global_q,
            global_t=global_t,
            vertices=canonical_vertices,
            quats=result.data[GF.rotation].view(len(self.instance_ids), -1, 4),
            weights=weights,
            instance_ids=self.instance_ids,
            instance_t=scene.objects.pose.t,
            instance_v=scene.objects.motion.linear_velocity,
            global_offset=0,
            out_positions=out_positions,
            out_rotations=out_rotation,
            out_velocities=out_velocity,
            out_mask=out_mask
        )
        # torch.cuda.synchronize()

        result.data[GF.positions] = out_positions
        result.data[GF.rotation] = out_rotation
        if self.return_velocity:
            result.data[GF.velocity] = out_velocity
        result.data[GF.mask] = out_mask

        scene.renderables.append(result)

        return result

    # TODO: store_to_container, optimize_inference, remap_instances
