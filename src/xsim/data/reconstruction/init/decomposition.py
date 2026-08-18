from collections import defaultdict

import torch
from xsimgs.cameras import RSCamera

from xsimgs.structures import ROIBox3D
from xsim.structures import SceneInitInstance
from xsim.utils.image_utils import grid_sample
from ..dataset import SceneReconstructionDataset
from .init_loader import init_loader_iterator


def extract_scene_instances(
        sim_ds: SceneReconstructionDataset,
        exclude_static_instance_ids: set[int] = None,
        target_device: torch.device | str = None,
        limit_to_roi: bool = True
):
    """
    Prepares scene initialization by:
        1. Coloring LiDAR point clouds by projecting them on closest cameras
        2. By using oriented boxes provided by dataset:
            * Isolates static point cloud
            * Gathers box-local per instance points from all LiDAR sweeps
    """
    if exclude_static_instance_ids is not None:
        excl_ids = torch.as_tensor(list(exclude_static_instance_ids)).to(target_device)
    else:
        excl_ids = None

    instances = defaultdict(list)
    roi: ROIBox3D = sim_ds.roi
    target_device = sim_ds.device if target_device is None else target_device

    for batch in init_loader_iterator(sim_ds, target_device=target_device):
        points = torch.cat([
            sweep.xyz[sweep.mask] for sweep in batch['sweeps']
        ], dim=0)
        if limit_to_roi:
            points = points[roi.check_inside(points)]

        boxes = batch['boxes']
        if excl_ids is not None:
            boxes = boxes[~torch.isin(boxes.object_ids, excl_ids)]

        point_colors = torch.zeros_like(points)
        point_color_norm = torch.zeros_like(points[:, 0])

        for image, camera, ego_mask in zip(
            batch['images'], batch['cameras'], batch['masks']
        ):
            camera: RSCamera
            uvdt, mask = camera.world_to_uv(points)
            uv = uvdt[None, None, :, :2] * 2.0 - 1.0

            sampled_mask = grid_sample(ego_mask, uv, mode='nearest').view(-1)
            mask &= sampled_mask

            sampled_color = grid_sample(image, uv[:, :, mask])[0]
            point_colors[mask] += sampled_color
            point_color_norm[mask] += 1

        point_colors /= point_color_norm.unsqueeze(-1).clamp_min(1e-3)
        point_colors = point_colors.clamp(0, 255).byte()
        visible_mask = point_color_norm > 0

        isect_map = boxes.intersection_map(points)
        if len(boxes) > 0:
            instance_ids = boxes.object_ids[isect_map.clamp(min=0)]
            instance_ids[isect_map == -1] = -1
        else:
            # A frame can hold no boxes at all once the static ones are
            # excluded; then every point is background.
            instance_ids = torch.full_like(isect_map, -1)

        for i, inst_id in enumerate([-1] + boxes.object_ids.tolist()):
            mask = instance_ids == inst_id
            indices = mask.nonzero().flatten()
            if len(indices) == 0:
                continue
            if inst_id == -1:
                inst_points = points[indices]
            else:
                # box for inst_id is boxes[i - 1] (index 0 in the loop is -1)
                inv_pose = boxes[i - 1].motion.pose.inv().view(1)
                inst_points = inv_pose @ points[indices]
            instances[inst_id].append(SceneInitInstance(
                points=inst_points,
                colors=point_colors[indices],
                visibility=visible_mask[indices]
            ))

    return {
        inst_id: SceneInitInstance.cat(instances[inst_id])
        for inst_id in instances
    }
