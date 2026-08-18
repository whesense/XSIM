from dataclasses import dataclass, field
from typing import Literal

from xsimgs.structures import ROIBox3D


@dataclass
class SceneDataOptions:
    # Downscale all images from RGB cameras upon loading.
    # 1 = original resolution, 2 = 2x smaller than original
    image_downscale_factor: float = 1.0
    # Undistort cameras and images upon loading. Only relevant if dataset provides
    # DistortedPerspectiveCamera or FisheyeCamera with non-zero distortion
    undistort_images: bool = True
    # Load segmentation masks
    load_segmentation: bool = True
    image_resample: Literal['nearest', 'bilinear', 'bicubic'] = 'bilinear'

    # Ensure computed scene ROI encompasses min_roi_extent ROI
    min_roi_extent: ROIBox3D = field(default_factory=lambda: ROIBox3D[0:0, 0:0, 0:20])
    # When computing scene ROI from LiDAR points, quantile filter is used to determine
    # ROI extent
    roi_quantile: float = 0.0025


@dataclass
class BatchOptions:
    project_depth_maps: bool = True
    max_lidars_per_batch: int = None



@dataclass
class DynamicFilterOptions:
    # Total minimal travel distance in meters to consider object as dynamic
    # Summed up from deltas between adjacent keyframes
    min_sum_dist: float = 0.2
    # Minimal distance between first and last valid keyframe to consider object as dynamic
    min_total_dist: float = 0.0
    # If min_total_dist > min_sum_dist * min_ratio, object is considered dynamic
    # min_ratio = 0.0 => criteria is not used
    min_ratio: float = 0.0


@dataclass
class ObjectTracksOptions:
    # Check whether object keyframes time form non-decreasing sequence
    check_time: bool = True
    # If object annotations disappear between two valid keyframes,
    # this method interpolate poses and mark these intermediate keyframes as valid
    fix_skips: bool = True
    # As LiDAR sweeps are usually not instantaneous (rolling shutter sensors),
    # individual box time correction is needed to ensure object keyframes time is precise.
    # If correct_time == False, box timestamps as given by scene provider class
    correct_time: bool = True
    # * If time_correction_method == 'points_time', use average time of points that are
    #   inside oriented box. If object is seen at both LiDAR capture start and finish,
    #   resulting time ~ middle of capture.
    # * If time_correction_method == 'box_centers', oriented box centers are projected
    #   onto LiDAR, and projected pixel time is used. If object is seen both at capture
    #   start and finish, resulting time ~ either capture start of finish
    time_correction_method: Literal['points_time'] | Literal['box_centers'] = 'points_time'

    # Additional filtering of dynamic objects to eliminate
    # static objects with annotation jitter
    dynamic_filter: DynamicFilterOptions = field(default_factory=DynamicFilterOptions)


@dataclass
class LoadOptions:
    # Load and cache all images on specified device upon dataset creation
    preload_images: bool = True
    # Device which is used to cache all data
    device: str = 'cpu'
    # Number of CPU workers to load images
    num_workers: int = 8


@dataclass
class SceneReconstructionDatasetConfig:
    scene: SceneDataOptions = field(default_factory=SceneDataOptions)
    #tracks: ObjectTracksOptions = field(default_factory=ObjectTracksOptions)
    dynamic_filter: DynamicFilterOptions = field(default_factory=DynamicFilterOptions)
    load: LoadOptions = field(default_factory=LoadOptions)
    batch: BatchOptions = field(default_factory=BatchOptions)
