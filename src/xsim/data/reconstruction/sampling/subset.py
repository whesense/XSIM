from typing import Optional, Literal

import torch

from xsimgs.structures import BoxSensorLink

from xsim.data.loaders import SensorType
from ..dataset import SceneReconstructionDataset
from .error_based_sampler import ErrorBasedImageSampler, ErrorBasedIndicesProvider


class Subset:
    def __init__(
            self,
            dataset: SceneReconstructionDataset,
            camera_ids: dict[int, list[int]],
            lidar_ids: dict[int, list[int]],
            split: str,
            sampler_kwargs: Optional[dict] = None,
    ):
        self.dataset = dataset
        self.camera_ids = camera_ids
        self.lidar_ids = lidar_ids
        self.lidar_ids_tensor = {
            lidar_id: torch.tensor(lidar_ids[lidar_id], dtype=torch.int32)
            for lidar_id in self.lidar_ids
        }
        self.split = split

        self.image_indices = sum([
            [(sensor_id, sample_idx) for sample_idx in camera_ids[sensor_id]]
            for sensor_id in camera_ids.keys()
        ], start=[])
        self.lidar_indices = sum([
            [(sensor_id, sample_idx) for sample_idx in lidar_ids[sensor_id]]
            for sensor_id in lidar_ids.keys()
        ], start=[])
        self.image_keys = BoxSensorLink.cat([
            BoxSensorLink.from_indices(SensorType.CAMERA.value, self.image_indices),
            BoxSensorLink.from_indices(SensorType.LIDAR.value, self.lidar_indices)
        ]).key().flatten()

        self.img_sampler = ErrorBasedImageSampler(
            self.image_indices,
            **(sampler_kwargs or {})
        )

        _, cameras_closest_lidar, _ = dataset.scene.camera_lidar_synchronization()
        # Find closest LiDAR samples from the same subset
        self.lidar_synchronized_indices = []
        for camera_id, sample_idx in self.image_indices:
            closest_lidars = cameras_closest_lidar[camera_id][sample_idx]
            cur_indices = []
            for lidar_id, lidar_sample_idx in closest_lidars:
                if lidar_id in self.lidar_ids and len(self.lidar_ids[lidar_id]) > 0:
                    cur_t = self.lidar_ids_tensor[lidar_id]
                    cur_indices.append((
                        lidar_id,
                        int(cur_t[(cur_t - lidar_sample_idx).abs().argmin()])
                    ))
            self.lidar_synchronized_indices.append(cur_indices)

        self.multi_indices = [
            {'image': [image_index], 'lidar': lidar_indices, 'seg_masks': [image_index]}
            for image_index, lidar_indices in
                zip(self.image_indices, self.lidar_synchronized_indices)
        ]

        self.train_indices_provider = ErrorBasedIndicesProvider(self.multi_indices)

        self.loaders = self.dataset.loader_datasets(None)

    def update_queue(self, n: int = 1):
        for _ in range(n):
            self.train_indices_provider.put(self.img_sampler.propose_sample())

    def eval_lidar_iterator(
            self,
            target_device: Optional[str] = None,
            num_workers: Optional[int] = None,
    ):
        iter = self.dataset.multi_loader_iterator(
            {'lidar': self.loaders['lidar']},
            [{'lidar': [v]} for v in self.lidar_indices],
            infinite_iterator=False,
            shuffle=False,
            target_device=target_device,
            num_workers=num_workers,
        )
        for iter_batch in iter:
            yield self.dataset.make_sensor_batch(iter_batch)

    def eval_iterator(
            self,
            components: Optional[list[str]] = None,
            target_device: Optional[str] = None,
            num_workers: Optional[int] = None,
    ):
        indices = self.multi_indices
        if components:
            indices = [
                {k: v for k, v in cur_index.keys() if k in components}
                for cur_index in indices
            ]

        iter = self.dataset.multi_loader_iterator(
            self.loaders,
            indices,
            infinite_iterator=False,
            shuffle=False,
            target_device=target_device,
            num_workers=num_workers,
        )
        for iter_batch in iter:
            yield self.dataset.make_sensor_batch(iter_batch)

    def train_iterator(
            self,
            target_device: Optional[str] = None,
            num_workers: Optional[int] = None,
    ):
        iter = self.dataset.multi_loader_iterator(
            self.loaders,
            self.train_indices_provider,
            infinite_iterator=True,
            shuffle=True,
            target_device=target_device,
            num_workers=num_workers,
        )
        real_num_workers = (
            num_workers if num_workers is not None else self.dataset.config.load.num_workers
        )
        prefetch_factor = 2
        self.update_queue(n=real_num_workers * (prefetch_factor + 1))

        for iter_batch in iter:
            self.update_queue()
            yield self.dataset.make_sensor_batch(iter_batch)
