# Small standalone SMPL body model forward pass, written from the published formulation:
#   Loper et al., "SMPL: A Skinned Multi-Person Linear Model",
#   ACM TOG 34(6) (SIGGRAPH Asia 2015).

from typing import NamedTuple

import torch
import torch.nn as nn

NUM_SMPL_JOINTS = 24


def load_smpl_model_data(model_path: str) -> dict:
    """Load an SMPL parameter dict (the original pickle re-saved via torch.save)
    and convert every array to a dense tensor.

    Uses the standard keys of the SMPL data file: ``v_template`` [V, 3],
    ``shapedirs`` [V, 3, num_shape_components], ``posedirs`` [V, 3, 207],
    ``J_regressor`` (scipy sparse [24, V]), ``weights`` [V, 24],
    ``kintree_table`` [2, 24] and ``f`` [F, 3].
    """
    import numpy as np

    raw = torch.load(model_path, map_location="cpu", weights_only=False)

    def dense_float(value) -> torch.Tensor:
        if not isinstance(value, np.ndarray):
            value = value.toarray()  # scipy sparse matrix (J_regressor)
        return torch.as_tensor(np.asarray(value, dtype=np.float32))

    parents = torch.as_tensor(raw["kintree_table"][0].astype(np.int64))
    parents[0] = -1  # the pelvis is the kinematic root

    return dict(
        v_template=dense_float(raw["v_template"]),
        shapedirs=dense_float(raw["shapedirs"]),
        posedirs=dense_float(raw["posedirs"]),
        J_regressor=dense_float(raw["J_regressor"]),
        lbs_weights=dense_float(raw["weights"]),
        parents=parents,
        faces=torch.as_tensor(raw["f"].astype(np.int64)),
    )


def kinematic_chain_transforms(
        rotations: torch.Tensor,  # [B, J, 3, 3] local joint rotations
        joints: torch.Tensor,  # [B, J, 3] rest-pose joint locations
        parents: torch.Tensor,  # [J] parent index per joint (root = -1)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose local joint rotations along the kinematic tree.

    Each joint's local transform rotates about the joint location, so its
    translation is the bone offset to the parent joint; world transforms are
    the parent-to-child products of these. The returned relative transforms
    ``A_k = G_k(theta) @ G_k(rest)^{-1}`` (with rest transforms being pure
    translations by the joint location) map rest-pose points to posed points,
    which is what LBS consumes.

    Returns (posed_joints [B, J, 3], relative_transforms [B, J, 4, 4]).
    """
    batch_size, num_joints = joints.shape[:2]

    bone_offsets = joints.clone()
    bone_offsets[:, 1:] -= joints[:, parents[1:]]

    local_transforms = torch.zeros(
        batch_size, num_joints, 4, 4, dtype=joints.dtype, device=joints.device)
    local_transforms[..., :3, :3] = rotations
    local_transforms[..., :3, 3] = bone_offsets
    local_transforms[..., 3, 3] = 1.0

    world_transforms = [local_transforms[:, 0]]
    for joint in range(1, num_joints):
        world_transforms.append(
            world_transforms[parents[joint]] @ local_transforms[:, joint])
    world_transforms = torch.stack(world_transforms, dim=1)

    posed_joints = world_transforms[..., :3, 3]

    relative_transforms = world_transforms.clone()
    relative_transforms[..., :3, 3] -= torch.einsum(
        "bjrc, bjc -> bjr", world_transforms[..., :3, :3], joints)

    return posed_joints, relative_transforms


def linear_blend_skinning(
        points: torch.Tensor,  # [B, V, 3] rest-pose points
        skinning_weights: torch.Tensor,  # [B, V, J]
        joint_transforms: torch.Tensor,  # [B, J, 4, 4]
) -> torch.Tensor:
    """Blend per-joint rigid transforms with the skinning weights and apply
    the blended transform to each point."""
    batch_size, num_joints = joint_transforms.shape[:2]

    point_transforms = torch.matmul(
        skinning_weights, joint_transforms.reshape(batch_size, num_joints, 16)
    ).view(batch_size, -1, 4, 4)

    rotated = torch.einsum(
        "bvrc, bvc -> bvr", point_transforms[..., :3, :3], points)
    return rotated + point_transforms[..., :3, 3]


class SMPLForwardResult(NamedTuple):
    vertices: torch.Tensor  # [B, V, 3] posed mesh vertices
    posed_joints: torch.Tensor  # [B, J, 3] joint locations in the given pose
    canonical_joints: torch.Tensor  # [B, J, 3] rest-pose (shaped) joints
    joint_transforms: torch.Tensor  # [B, J, 4, 4] rest-to-posed transforms


class SMPLBodyModel(nn.Module):
    """SMPL forward pass: shape blend offsets, joint regression, pose blend
    offsets, kinematic chain and linear blend skinning.

    Buffers are named after the keys of the SMPL data file, which is also how
    they appear in checkpoint state dicts. Two carry a reshaped layout:
    ``shapedirs`` is truncated to ``num_betas`` components and ``posedirs`` is
    stored flattened as [207, V * 3].
    """

    def __init__(self, model_path: str, num_betas: int = 10):
        super().__init__()
        data = load_smpl_model_data(model_path)
        self.num_betas = num_betas

        num_pose_components = data["posedirs"].shape[-1]
        posedirs = data["posedirs"].reshape(-1, num_pose_components)

        self.register_buffer("v_template", data["v_template"])
        self.register_buffer("shapedirs", data["shapedirs"][..., :num_betas].contiguous())
        self.register_buffer("posedirs", posedirs.T.contiguous())
        self.register_buffer("J_regressor", data["J_regressor"])
        self.register_buffer("lbs_weights", data["lbs_weights"])
        self.register_buffer("parents", data["parents"])
        self.register_buffer("faces_tensor", data["faces"])

    @property
    def num_vertices(self) -> int:
        return self.v_template.shape[0]

    def forward(
            self,
            betas: torch.Tensor,  # [B, num_betas]
            pose_rotations: torch.Tensor,  # [B, 24, 3, 3], joint 0 = global orient
    ) -> SMPLForwardResult:
        batch_size = pose_rotations.shape[0]

        # shape blend offsets and joint regression on the shaped template
        v_shaped = self.v_template + torch.einsum(
            "bs, vcs -> bvc", betas, self.shapedirs)
        canonical_joints = torch.einsum(
            "jv, bvc -> bjc", self.J_regressor, v_shaped)

        # pose blend offsets, driven by the non-root rotations relative to rest
        identity = torch.eye(
            3, dtype=pose_rotations.dtype, device=pose_rotations.device)
        pose_features = (pose_rotations[:, 1:] - identity).reshape(batch_size, -1)
        v_posed = v_shaped + torch.matmul(
            pose_features, self.posedirs).view(batch_size, -1, 3)

        posed_joints, joint_transforms = kinematic_chain_transforms(
            pose_rotations, canonical_joints, self.parents)

        skinning_weights = self.lbs_weights[None].expand(batch_size, -1, -1)
        vertices = linear_blend_skinning(
            v_posed, skinning_weights, joint_transforms)

        return SMPLForwardResult(
            vertices=vertices,
            posed_joints=posed_joints,
            canonical_joints=canonical_joints,
            joint_transforms=joint_transforms,
        )
