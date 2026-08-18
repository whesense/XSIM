import random
from dataclasses import dataclass
from typing import Optional

import torch
from dacite import from_dict
from dacite import Config as DaciteConfig
from toast import SE3
from xsimgs.cameras import RSCamera, rasterize_points_into_depth
from xsimgs.structures import ROIBox3D, OrientedBoxes

from xsim.structures import Sweep
from .dataset_config import SceneReconstructionDatasetConfig
from .utils import fit_camera_time, instance_matrix, dynamic_instances_filter
from ..loaders import SensorLoader, multi_sensor_load_iterator, SensorType
from ..loaders.sensor_iterator import sensor_load_iterator, \
    images_from_iterator, all_sensor_indices
from ..loaders.sensor_loader import ImageIndices, BasicSensorLoadingDataset, \
    MultiLoaderIndices
from ..provider.basic_scene import DatasetScene
from xsim.data.loaders.image_loaders import (
    resize_mask, ImageLoadingDataset, SegmentationLoadingDataset
)
from ..utils.smpl import SMPLData
from ...utils.image_utils import grid_sample



class PreloadedImages(SensorLoader):
    preloaded = True

    def __init__(self, dataset, key: str):
        super().__init__()
        self.dataset = dataset
        self.key = key
        self.field = getattr(self.dataset, self.key)

    def __call__(self, sensor_id: int, sample_idx: int):
        return self.field[sensor_id][sample_idx]



@dataclass
class SensorImage:
    sensor_type: SensorType
    sensor_id: int
    sample_idx: int

    camera: RSCamera
    image: torch.Tensor | Sweep
    gt_mask: torch.Tensor

    info: dict


SensorBatch = list[SensorImage]


class SceneReconstructionDataset:
    # Indices of existing cameras in dataset
    camera_indices: list[int]
    # Camera human-readable captions
    camera_captions: list[str]
    # Order in which cameras should be concatenated for visualization
    camera_vis_order: list[int]
    # Static calibrations between camera and ego-frame.
    # Transformations from camera to ego-frame
    ego_se3_camera: dict[int, SE3]
    # Camera objects. len(cameras[camera_id]) == num images captures with this camera
    cameras: dict[int, RSCamera]

    # Indices of exiting LiDAR sensors
    lidar_indices: list[int]
    # LiDAR human-readable captions
    lidar_captions: list[str]
    # Static calibrations, transformation from LiDAR to ego-frame
    ego_se3_lidar: dict[int, SE3]
    # LiDAR camera objects.
    # len(lidar_cameras[lidar_id]) == num sweeps captured with this LiDAR
    lidar_cameras: dict[int, RSCamera]
    # Sweeps produced by LiDAR-s
    sweeps: dict[int, list[Sweep]]

    # Object annotations
    boxes: list[OrientedBoxes]
    all_boxes: OrientedBoxes
    # SMPL data
    smpl_data: dict[int, SMPLData | None]
    smpl_object_ids: set[int]
    num_smpl_objects: int
    # Current scene ROI in 3D
    roi: ROIBox3D

    # Instance matrix of shape [num_instances, max_num_keyframes]
    instances: OrientedBoxes
    # instance size tensor [num_instances, 3]
    instance_size: torch.Tensor
    static_ids: set[int]
    # Dynamic instances IDs
    dynamic_ids: set[int]
    # Dynamic vehicle IDs
    dynamic_vehicle_ids: set[int]
    # Dynamic human IDs
    dynamic_human_ids: set[int]


    def __init__(
            self,
            scene: DatasetScene,
            config: SceneReconstructionDatasetConfig | dict
    ):
        if not isinstance(config, SceneReconstructionDatasetConfig):
            config = from_dict(
                data_class=SceneReconstructionDatasetConfig,
                data=config,
                config=DaciteConfig(strict=True)
            )

        self.scene = scene
        self.config = config
        self.device = self.config.load.device

        # Sensor rig calibration
        self.camera_indices = scene.cameras.indices
        self.camera_captions = scene.cameras.captions
        self.camera_vis_order = scene.cameras.vis_order
        self.lidar_indices = scene.lidar.indices
        self.lidar_captions = scene.lidar.captions

        self.ego_se3_camera = {
            camera_id: scene.cameras.get_ego_se3_sensor(camera_id)
            for camera_id in self.camera_indices
        }
        self.ego_se3_lidar = {
            lidar_id: scene.lidar.get_ego_se3_sensor(lidar_id)
            for lidar_id in self.lidar_indices
        }
        self.cameras = {
            camera_id: scene.cameras.get_cameras(camera_id).to(self.device)
            for camera_id in self.camera_indices
        }
        self.lidar_cameras = {
            lidar_id: scene.lidar.get_cameras(lidar_id).to(self.device)
            for lidar_id in self.lidar_indices
        }

        self.camera_time_fn, self.lidar_time_fn, self.time_range = fit_camera_time(
            self.cameras, self.lidar_cameras, scene_time_extension=0.2
        )

        self.boxes = [
            scene.get_boxes(i).to(self.device)
            for i in range(scene.num_objects)
        ]
        self.all_boxes = (
            OrientedBoxes.cat(self.boxes, dim=0) if self.boxes
            else OrientedBoxes.zeros(size=[0], device=self.device)
        )

        self.instances, self.instance_size = instance_matrix(
            self.boxes, device=self.device
        )
        self.dynamic_ids = set(dynamic_instances_filter(
            self.instances, self.config.dynamic_filter
        ))
        self.static_ids = set(list(range(scene.num_objects))).difference(self.dynamic_ids)
        self.dynamic_vehicle_ids = set(scene.vehicle_ids).intersection(self.dynamic_ids)
        self.dynamic_human_ids = set(scene.human_ids).intersection(self.dynamic_ids)

        # Only dynamic (articulated) humans go to the SMPL node; static humans
        # are left to the regular gaussian nodes, so drop their SMPL data.
        self.smpl_data = {
            smpl_id: (
                scene.get_smpl_data(smpl_id)
                if smpl_id in self.dynamic_human_ids else None
            )
            for smpl_id in scene.human_ids
        }
        self.smpl_object_ids = {k for k,v in self.smpl_data.items() if v is not None}
        self.num_smpl_objects = len(self.smpl_object_ids)

        self.camera_masks = {
            i: resize_mask(
                scene.cameras.get_mask(i).to(self.device),
                downsample_factor=config.scene.image_downscale_factor
            ) for i in scene.cameras.indices
        }
        self.undistortion_maps = {}
        self.undistortion_maps_cpu = {}
        if self.config.scene.undistort_images:
            for camera_id, camera_caption in zip(self.camera_indices, self.camera_captions):
                cam0 = self.cameras[camera_id][0]
                if not hasattr(cam0, 'undistort') or not hasattr(cam0, 'undistort_map'):
                    print('Camera {} "{}" does not support undistortion, skipping'.format(
                        camera_id, camera_caption
                    ))
                    continue

                cur_map = cam0.undistort_map(
                    width=self.camera_masks[camera_id].shape[1],
                    height=self.camera_masks[camera_id].shape[0],
                ).unsqueeze(0)
                self.undistortion_maps[camera_id] = cur_map
                self.undistortion_maps_cpu[camera_id] = cur_map.cpu()
                self.camera_masks[camera_id] = grid_sample(
                    self.camera_masks[camera_id],
                    cur_map,
                    mode=self.config.scene.image_resample
                )
                self.cameras[camera_id] = self.cameras[camera_id].undistort()

        self.images = None
        self.seg_masks = None

        if self.config.load.preload_images:
            self.images = self.load_images(
                self.camera_loader_dataset(all_sensor_indices(self.scene.cameras)),
                tqdm_desc='Loading images'
            )
            if self.config.scene.load_segmentation:
                self.seg_masks = self.load_images(
                    self.segmentation_loader_dataset(all_sensor_indices(self.scene.cameras)),
                    tqdm_desc='Loading segmentation masks'
                )

        self.sweeps = self.load_images(
            self.lidar_loader_dataset(all_sensor_indices(self.scene.lidar)),
            tqdm_desc='Loading LiDAR'
        )

        for lidar_id in self.lidar_indices:
            if len(self.sweeps[lidar_id]) != len(self.lidar_cameras[lidar_id]):
                raise ValueError(
                    'Mismatch in number of LiDAR sweeps and cameras '
                    'for lidar_id={}'.format(lidar_id))

        self.lidar_roi = scene.get_roi(
            sum([self.sweeps[lidar_id] for lidar_id in self.lidar_indices], start=[]),
            quantile=self.config.scene.roi_quantile
        )
        self.roi = self.lidar_roi | self.config.scene.min_roi_extent

    def camera_loader_dataset(self, indices: ImageIndices):
        if self.images is None:
            return ImageLoadingDataset(
                self.scene.cameras.get_loader(),
                indices=indices,
                downsample_factor=self.config.scene.image_downscale_factor,
                img_remap=self.undistortion_maps_cpu,
                downsample_mode=self.config.scene.image_resample,
                ensure_divisible=1
            )
        else:
            return BasicSensorLoadingDataset(
                PreloadedImages(self, 'images'),
                indices=indices,
            )

    def lidar_loader_dataset(self, indices: ImageIndices):
        return BasicSensorLoadingDataset(
            (
                self.scene.lidar.get_loader()
                if (not hasattr(self, 'sweeps')) or (self.sweeps is None)
                else PreloadedImages(self, 'sweeps')
            ), indices=indices
        )

    def segmentation_loader_dataset(self, indices: ImageIndices):
        if self.seg_masks is None:
            return SegmentationLoadingDataset(
                self.scene.get_segmentation_loader(),
                indices=indices,
                downsample_factor=self.config.scene.image_downscale_factor,
                img_remap=self.undistortion_maps_cpu,
                ensure_divisible=1,
                downsample_mode=self.config.scene.image_resample,
            )
        else:
            return BasicSensorLoadingDataset(
                PreloadedImages(self, 'seg_masks'),
                indices=indices,
            )

    def loader_datasets(self, indices: ImageIndices):
        return {
            'image': self.camera_loader_dataset(indices),
            'lidar': self.lidar_loader_dataset(indices),
            'seg_masks': self.segmentation_loader_dataset(indices),
        }

    def make_sensor_batch(self, iter_batch: dict) -> SensorBatch:
        result = SensorBatch()

        points_lst = []
        points = None

        gather_points = self.config.batch.project_depth_maps
        use_lidar_ids = [v[0][0] for v in iter_batch.get('lidar', [])]
        if use_lidar_ids and self.config.batch.max_lidars_per_batch is not None:
            use_lidar_ids = random.sample(
                use_lidar_ids,
                k=self.config.batch.max_lidars_per_batch
            )
        if 'lidar' in iter_batch:
            for (lidar_id, lidar_sample_idx), sweep in iter_batch['lidar']:
                info = {}
                sweep: Sweep

                if gather_points:
                    points_lst.append(sweep.xyz[sweep.mask])
                if lidar_id in use_lidar_ids:
                    camera = self.lidar_cameras[lidar_id][lidar_sample_idx].to(sweep.device)
                    result.append(SensorImage(
                        sensor_type=SensorType.LIDAR,
                        sensor_id=lidar_id,
                        sample_idx=lidar_sample_idx,
                        image=sweep,
                        gt_mask=sweep.mask,
                        camera=camera,
                        info=info,
                    ))

        if len(points_lst) > 0 and self.config.batch.project_depth_maps:
            points = torch.cat(points_lst, dim=0)

        if 'image' in iter_batch:
            for i, ((camera_id, camera_sample_idx), image) in enumerate(iter_batch['image']):
                camera = self.cameras[camera_id][camera_sample_idx].to(image.device)
                info = {}

                if points is not None:
                    info['depth'] = rasterize_points_into_depth(
                        camera, points, width=image.shape[1], height=image.shape[0],
                    )[0]
                if 'seg_masks' in iter_batch:
                    seg_mask = iter_batch['seg_masks'][i][1]
                    info['sky_mask'] = seg_mask == 2

                result.append(SensorImage(
                    sensor_type=SensorType.CAMERA,
                    sensor_id=camera_id,
                    sample_idx=camera_sample_idx,
                    image=image,
                    gt_mask=self.camera_masks[camera_id].to(image.device),
                    camera=camera,
                    info=info,
                ))

        return result

    def _load_params(
            self,
            num_workers: Optional[int] = None,
            target_device: Optional[torch.device | str] = None,
    ):
        return dict(
            num_workers=(
                num_workers if num_workers is not None else self.config.load.num_workers
            ),
            target_device=(
                target_device if target_device is not None else self.config.load.device
            ),
        )

    def load_iterator(
            self,
            loader_dataset: BasicSensorLoadingDataset,
            num_workers: Optional[int] = None,
            target_device: Optional[torch.device | str] = None,
    ):
        return sensor_load_iterator(
            loader_dataset,
            **self._load_params(num_workers=num_workers, target_device=target_device)
        )

    def multi_loader_iterator(
            self,
            loaders: dict[str, BasicSensorLoadingDataset],
            indices: MultiLoaderIndices,
            infinite_iterator: bool = False,
            shuffle: bool = False,
            num_workers: Optional[int] = None,
            target_device: Optional[torch.device | str] = None,
    ):
        return multi_sensor_load_iterator(
            loaders, indices,
            infinite_iterator=infinite_iterator,
            shuffle=shuffle,
            **self._load_params(num_workers=num_workers, target_device=target_device),
        )

    def load_images(
            self,
            loader_dataset: BasicSensorLoadingDataset,
            num_workers: Optional[int] = None,
            target_device: Optional[torch.device | str] = None,
            tqdm_desc: Optional[str] = 'Loading images'
    ):
        return images_from_iterator(
            self.load_iterator(
                loader_dataset,
                num_workers=num_workers,
                target_device=target_device
            ), total_images=len(loader_dataset),
            tqdm_desc=tqdm_desc
        )
