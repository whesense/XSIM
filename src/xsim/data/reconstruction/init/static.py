import torch

from xsimgs.structures import ROIBox3D, OrientedBoxes
from xsim.structures import SceneInitInstance

from .init_config import SceneProcessingConfig
from .voxel_filters import voxel_count_filter, voxel_downsample, inst_z_tile
from .point_sampling import uniform_sample_sphere, far_points_sampling
from xsim.structures.init_instance import knn_avg


def process_background(
        bg: SceneInitInstance,
        cfg: SceneProcessingConfig,
        roi: ROIBox3D,
        lidar_roi: ROIBox3D,
        boxes: OrientedBoxes
):
    if cfg.voxel_count_filter_min_cnt > 1:
        bg = voxel_count_filter(
            bg, roi,
            voxel_size=cfg.voxel_count_filter_size,
            min_points_per_voxel=cfg.voxel_count_filter_min_cnt
        )

    if cfg.background_downsampler == 'voxel':
        visible_bg = bg.visible
        invisible_bg = bg.invisible

        points_vis_dn, colors_vis_dn, voxel_cnt_sparse, grid = voxel_downsample(
            visible_bg, roi,
            voxel_size=cfg.background_downsampler_voxel_size
        )

        if cfg.use_invisible_points:
            points_invis_dn, _, _, _ = voxel_downsample(
                invisible_bg, roi,
                voxel_size=cfg.background_downsampler_voxel_size
            )
            all_boxes = boxes.intersection_map(points_invis_dn) == -1
            points_invis_dn = points_invis_dn[all_boxes]
            # Leave only points in voxels where there is no visible points
            invis_mask = ~torch.isin(grid.voxel_coords_offset(points_invis_dn),
                                     voxel_cnt_sparse.indices()[0])
            points_invis_dn = points_invis_dn[invis_mask]
            colors_invis_dn = knn_avg(
                points_invis_dn, points_vis_dn, colors_vis_dn,
                k=cfg.color_by_k_neighbour
            )
            points_vis_dn = torch.cat([points_vis_dn, points_invis_dn], dim=0)
            colors_vis_dn = torch.cat([colors_vis_dn, colors_invis_dn], dim=0)

        bg = SceneInitInstance(
            points=points_vis_dn, colors=colors_vis_dn,
            visibility=torch.ones_like(points_vis_dn[..., 0], dtype=torch.bool)
        )
    bg = bg.subsample(num_points=cfg.background_random_target)
    bg = bg.color_invisible(k=cfg.color_by_k_neighbour)
    if cfg.background_tile_z_to_roi:
        bg = inst_z_tile(
            bg,
            lidar_roi=lidar_roi,
            target_roi=roi,
            z_quantile=cfg.background_tile_z_quantile
        )

    add_instances = []

    if cfg.num_sphere_samples > 0 or cfg.num_inv_sphere_samples > 0:
        sphere_origin = roi.center.to(bg.points.device).view(1, 3)
        sphere_radius = float(roi.size.max().mul(0.5).mul(cfg.sphere_scale_factor))
        if cfg.num_sphere_samples:
            samples = uniform_sample_sphere(
                cfg.num_sphere_samples,
                device=bg.points.device
            )
            samples = samples * sphere_radius + sphere_origin
            add_instances.append(SceneInitInstance(points=samples))
        if cfg.num_inv_sphere_samples:
            samples = uniform_sample_sphere(
                cfg.num_sphere_samples,
                device=bg.points.device,
                inverse=True
            )
            samples = samples * sphere_radius + sphere_origin
            add_instances.append(SceneInitInstance(points=samples))

    if cfg.num_far_points > 0:
        points = far_points_sampling(
            lidar_roi,
            num_samples=cfg.num_far_points,
            far=cfg.far_points_max_dist,
            device=bg.points.device
        )
        add_instances.append(SceneInitInstance(points=points))

    if len(add_instances) > 0:
        bg = SceneInitInstance.cat([bg] + add_instances, dim=0)

    return bg
