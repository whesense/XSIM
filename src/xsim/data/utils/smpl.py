from dataclasses import dataclass

import toast
import torch
from xsimgs.structures import OrientedBoxes


@dataclass
class SMPLData:
    betas: torch.Tensor
    joint_q: torch.Tensor


def get_smpl_instance(
        inst_data: dict,
        world_so3b_cam: toast.Quat
) -> dict[str, torch.Tensor]:
    valid_mask = inst_data['valid_mask']
    cam_indices = inst_data['selected_cam_idx']
    global_orient = toast.Quat.from_matrix(inst_data['smpl']['global_orient'][:, 0]).std()
    body_pose = toast.Quat.from_matrix(inst_data['smpl']['body_pose']).std()

    betas = inst_data['smpl']['betas']
    world_so3_cam = world_so3b_cam[cam_indices, torch.arange(len(cam_indices))]
    world_so3_root = (world_so3_cam @ global_orient).data.view(-1, 1, 4)
    body_pose_full = toast.Quat(torch.cat([world_so3_root, body_pose.data], dim=1))

    return dict(
        mask=valid_mask,
        betas=betas[valid_mask][0],
        pose=body_pose_full.std().data
    )


def process_smpl_instance(
        inst_data: dict,
        inst_boxes: OrientedBoxes,
        world_so3b_cam: toast.Quat,
):
    assert inst_boxes.sensor_link is not None

    lidar_frame_idx = inst_boxes.sensor_link.sample_idx.view(-1)

    inst_smpl = get_smpl_instance(inst_data, world_so3b_cam)
    inst_smpl_mask = inst_smpl['mask'][lidar_frame_idx]
    inst_poses = toast.Quat(inst_smpl['pose'][lidar_frame_idx])

    to_sorted_idxs = lidar_frame_idx.argsort()
    q_t = lidar_frame_idx.float()[to_sorted_idxs].view(1, -1).expand(24, -1)
    seq_quats = inst_poses.data[to_sorted_idxs].permute(1, 0, 2)

    with toast.use_backend('torch'):
        tgt_quats = toast.sample_trajectory(
            query_times=q_t,
            seq_times=q_t,
            seq_quats=seq_quats,
            seq_mask=inst_smpl_mask[to_sorted_idxs].view(1, -1).expand(24, -1),
            extrapolate=False,  # set SMPL pose to the nearest valid
            use_only_valid_keyframes=True,
            return_mask=True
        )[0]
        inv_sort = to_sorted_idxs.argsort()
        tgt_quats = tgt_quats.permute(1, 0, 2)[inv_sort]

    return SMPLData(
        betas=inst_smpl['betas'],
        joint_q=tgt_quats.contiguous()
    )

