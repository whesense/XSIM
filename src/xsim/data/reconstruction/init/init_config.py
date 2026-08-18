from dataclasses import dataclass
from typing import Literal


@dataclass
class SceneProcessingConfig:
    # If enabled, LiDAR points not contained in scene ROI are discarded
    limit_points_to_roi: bool = False
    # If disabled, LiDAR points not observed by any synchronized camera image are discarded
    # If enabled, they are used, and colored by K nearest neighbors colors
    use_invisible_points: bool = True
    color_by_k_neighbour: int = 3

    # Voxel-based static point cloud filter. Useful to remove noisy points or floaters
    # Divides ROI into voxels with specified size. If voxel has less than specified number
    # of points, they are excluded from static initialization
    voxel_count_filter_size: float = 0.2
    voxel_count_filter_min_cnt: int = 1 # 1 = voxel-based filtering disabled

    # Static initialization downsampling method:
    # * "random": chooses random N points
    # * "voxel": Divides ROI into voxels with given size, aggregates point cloud
    #            so that each voxel have only single point.
    #            Downsamples resulting point up to N points
    background_downsampler: Literal['random'] | Literal['voxel'] = 'voxel'
    background_random_target: int = 800_000
    background_downsampler_voxel_size: float = 0.1
    # If this option is enabled, takes upper part of LiDAR roi, and tiles it vertically
    # up to full scene ROI. Useful to provide initialization points for tall buildings,
    # poorly covered initially by LiDAR
    background_tile_z_to_roi: bool = True
    background_tile_z_quantile: float = 0.01

    # Augments vehicles by using left/right symmetry assumption
    symmetry_augment_axis: int = 1  # Y-axis (longitudinal, along vehicle length)
    symmetry_augment_vehicles: bool = True
    # Maximum number of initialization points for each dynamic instance
    max_points_per_instance: int = 10_000

    # Additionally random sampled points configuration
    sphere_scale_factor: float = 1.1
    num_sphere_samples: int = 0
    num_inv_sphere_samples: int = 0
    num_far_points: int = 0
    far_points_max_dist: float = 1e4

    # Which device to use to perform initialization
    # (also where initialization result is stored)
    target_device: str = 'cuda'
