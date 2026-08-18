# Contains modified code from OmniRe / drivestudio
# (https://github.com/ziyc/drivestudio, MIT, Copyright (c) 2024 Ziyu Chen).
# See THIRD_PARTY_LICENSES.md.

import torch
import torch.nn as nn

from toast import quat_to_matrix, quat_from_axis_angle

from .smpl_body_model import SMPLBodyModel, kinematic_chain_transforms

from .voxel_deformer import VoxelDeformer


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quat_to_matrix(quat_from_axis_angle(axis_angle))


def get_predefined_human_rest_pose(pose_type):
    # print(f"Using predefined pose: {pose_type}")
    body_pose_t = torch.zeros((1, 69))
    if pose_type.lower() == "da_pose":
        body_pose_t[:, 2] = torch.pi / 6
        body_pose_t[:, 5] = -torch.pi / 6
    elif pose_type.lower() == "a_pose":
        body_pose_t[:, 2] = 0.2
        body_pose_t[:, 5] = -0.2
        body_pose_t[:, 47] = -0.8
        body_pose_t[:, 50] = 0.8
    elif pose_type.lower() == "t_pose":
        pass
    else:
        raise ValueError("Unknown cano_pose: {}".format(pose_type))
    return body_pose_t.reshape(23, 3)


class SMPLTemplate(nn.Module):
    def __init__(self, smpl_model_path, num_human, init_beta, cano_pose_type,
                 voxel_deformer_res=64, use_voxel_deformer=False, is_resume=False):
        super().__init__()
        assert num_human == init_beta.shape[
            0], "num_human should be the same as the number of beta"
        self.num_human = num_human
        self.dim = 24
        # The attribute name is part of the checkpoint contract: the body
        # model's buffers are stored under `_template_layer.*`.
        self._template_layer = SMPLBodyModel(model_path=smpl_model_path)

        init_beta = torch.as_tensor(init_beta, dtype=torch.float32).cpu()
        self.register_buffer("init_beta", init_beta)
        self.cano_pose_type = cano_pose_type
        self.name = "smpl"

        can_pose = get_predefined_human_rest_pose(cano_pose_type)
        can_pose = axis_angle_to_matrix(torch.cat([torch.zeros(1, 3), can_pose], 0))
        self.register_buffer("canonical_pose", can_pose)

        init_smpl_output = self._template_layer(
            betas=init_beta,
            pose_rotations=can_pose[None].repeat(num_human, 1, 1, 1),
        )
        J_canonical = init_smpl_output.canonical_joints
        A0 = init_smpl_output.joint_transforms
        A0_inv = torch.inverse(A0)
        self.register_buffer("A0_inv", A0_inv)
        self.register_buffer("J_canonical", J_canonical)

        v_init = init_smpl_output.vertices  # [B, 6890, 3]
        W_init = self._template_layer.lbs_weights  # [6890, 24]
        self.register_buffer("W", W_init[None].repeat(num_human, 1, 1))

        self.use_voxel_deformer = use_voxel_deformer
        if self.use_voxel_deformer:
            self.voxel_deformer = VoxelDeformer(
                vtx=v_init,
                vtx_features=W_init.unsqueeze(0).repeat(num_human, 1, 1),  # [B, 6890, 24]
                resolution_dhw=[
                    voxel_deformer_res // 4,
                    voxel_deformer_res,
                    voxel_deformer_res,
                ],
                is_resume=is_resume,
                # if is resume, the weight will be randomly initialized to save time
            )

        # * Important, record first joint position,
        # because the global orientation is rotating using this joint position as center,
        # so we can compute the action on later As
        j0_t = init_smpl_output.posed_joints[:, 0]
        self.register_buffer("j0_t", j0_t)
        return

    @property
    def num_vertices(self) -> int:
        return self._template_layer.num_vertices

    def get_init_vf(self):
        init_smpl_output = self._template_layer(
            betas=self.init_beta,
            pose_rotations=self.canonical_pose[None].repeat(self.num_human, 1, 1, 1),
        )
        v_init = init_smpl_output.vertices  # 1,6890,3
        faces = self._template_layer.faces_tensor
        return v_init, faces

    def get_rot_action(self, axis_angle):
        # apply this action to canonical additional bones
        # axis_angle: B,3
        assert axis_angle.ndim == 2 and axis_angle.shape[-1] == 3
        B = len(axis_angle)
        R = axis_angle_to_matrix(axis_angle)  # B,3,3
        I = torch.eye(3).to(R)[None].expand(B, -1, -1)  # B,3,3
        t0 = self.j0_t[None].expand(B, -1)  # B,3
        T = torch.eye(4).to(R)[None].expand(B, -1, -1)  # B,4,4
        T[:, :3, :3] = R
        T[:, :3, 3] = torch.einsum("bij, bj -> bi", I - R, t0)
        return T  # B,4,4

    def forward(self, masked_theta=None, xyz_canonical=None, instances_mask=None):
        # skinning
        if masked_theta is None:
            A = None
        else:
            assert (
                    masked_theta.ndim == 3 and masked_theta.shape[-1] == 4
            ), "pose should have shape Bx24x3, in axis-angle format"
            nB = len(masked_theta)
            with torch.cuda.nvtx.range("kinematic_chain_transforms"):
                _, A = kinematic_chain_transforms(
                    quat_to_matrix(masked_theta),
                    self.J_canonical[instances_mask],
                    self._template_layer.parents,
                )
            with torch.cuda.nvtx.range("einsum_after_rigid_transform"):
                A = torch.einsum("bnij, bnjk->bnik", A,
                                 self.A0_inv[instances_mask])  # B,24,4,4

        if xyz_canonical is None or not self.use_voxel_deformer:
            # forward theta only
            W = self.W
        else:
            with torch.cuda.nvtx.range("voxel_deformer"):
                W = self.voxel_deformer(xyz_canonical)  # B,N,24+K
        W = W[instances_mask]
        return W, A

    def remove_instance(self, ins_id):
        ins_mask = torch.ones(self.num_human, dtype=torch.bool)
        ins_mask[ins_id] = False
        self.J_canonical = self.J_canonical[ins_mask]
        self.W = self.W[ins_mask]
        self.num_human -= 1

        if self.use_voxel_deformer:
            self.voxel_deformer.lbs_voxel_base = self.voxel_deformer.lbs_voxel_base[
                ins_mask]
            self.voxel_deformer.voxel_w_correction = nn.Parameter(
                self.voxel_deformer.voxel_w_correction[ins_mask])
            self.voxel_deformer.offset = self.voxel_deformer.offset[ins_mask]
            self.voxel_deformer.scale = self.voxel_deformer.scale[ins_mask]

    def add_instance(self, ins_id, new_dict):
        voxel_deformer_dict = new_dict["voxel_deformer"]
        self.J_canonical = torch.cat([self.J_canonical, new_dict['J_canonical'][None]],
                                     dim=0)
        self.W = torch.cat([self.W, new_dict['W'][None]], dim=0)
        self.num_human += 1

        if self.use_voxel_deformer:
            self.voxel_deformer.lbs_voxel_base = torch.cat(
                [self.voxel_deformer.lbs_voxel_base,
                 voxel_deformer_dict['lbs_voxel_base'][None]], dim=0)
            self.voxel_deformer.voxel_w_correction = nn.Parameter(torch.cat(
                [self.voxel_deformer.voxel_w_correction,
                 voxel_deformer_dict['voxel_w_correction'][None]], dim=0))
            self.voxel_deformer.offset = torch.cat(
                [self.voxel_deformer.offset, voxel_deformer_dict['offset'][None]], dim=0)
            self.voxel_deformer.scale = torch.cat(
                [self.voxel_deformer.scale, voxel_deformer_dict['scale'][None]], dim=0)
