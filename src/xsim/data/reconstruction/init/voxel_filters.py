import numpy as np
import torch

from xsimgs.structures import ROIBox3D, VoxelGrid3D
from xsim.structures import SceneInitInstance


def voxel_count_filter(
        instance: SceneInitInstance,
        roi: ROIBox3D,
        voxel_size=0.25,
        min_points_per_voxel: int = 2
) -> SceneInitInstance:
    grid = VoxelGrid3D(roi, voxel_size=voxel_size)
    visible_vox_idxs = grid.voxel_coords_offset(instance.points).view(1, -1)
    vox_cnt_sparse = torch.sparse_coo_tensor(
        indices=visible_vox_idxs,
        values=torch.ones_like(visible_vox_idxs[0])
    ).coalesce()
    vox_ids = vox_cnt_sparse.indices().view(-1)
    vox_mask = vox_cnt_sparse.values().view(-1) >= min_points_per_voxel
    mask = vox_mask[torch.searchsorted(vox_ids, visible_vox_idxs.view(-1))]
    return instance[mask]


def voxel_downsample(
        inst: SceneInitInstance,
        roi: ROIBox3D,
        voxel_size: float = 0.1
):
    grid = VoxelGrid3D(roi, voxel_size=voxel_size)
    size=int(np.prod(list(grid.shape)))
    vox_idxs = grid.voxel_coords_offset(inst.points).view(1, -1)
    vox_cnt_sparse = torch.sparse_coo_tensor(
        indices=vox_idxs, values=torch.ones_like(vox_idxs[0])).coalesce()
    vox_points_sparse = torch.sparse_coo_tensor(
        indices=vox_idxs, values=inst.points, size=(size, 3)).coalesce()
    vox_colors_sparse = torch.sparse_coo_tensor(
        indices=vox_idxs, values=inst.colors.float(), size=(size, 3)).coalesce()
    points_new = vox_points_sparse.values() / vox_cnt_sparse.values().view(-1, 1)
    colors_new = (vox_colors_sparse.values() / vox_cnt_sparse.values().view(-1, 1)).byte()
    return points_new, colors_new, vox_cnt_sparse, grid


def inst_z_tile(
        bg: SceneInitInstance,
        lidar_roi: ROIBox3D,
        target_roi: ROIBox3D,
        z_quantile: float = 0.01
):
    # repeat higher part of scene defined by lidar_roi (according to Z axis)
    # up to target_roi

    z_out = bg[bg.points[:, 2] > lidar_roi.vmax[2]]

    z_out_min, z_out_max = torch.quantile(
        z_out.points[:, 2],
        q=torch.tensor([z_quantile, 1 - z_quantile], device=z_out.device),
    )
    z_step = z_out_max - z_out_min
    z_diff = target_roi.vmax[2] - z_out_max
    num_repeats = int((z_diff / z_step).round())

    augs = []
    for i in range(num_repeats):
        newi = z_out.clone()
        newi.points[:, 2] += z_step * (i + 1)
        augs.append(newi)

    return SceneInitInstance.cat([bg] + augs)
