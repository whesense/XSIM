from enum import Enum, auto

import torch
from toast import SE3

from xsimgs.cameras import RSCamera
from .sensor_loader import SensorLoader


class SensorType(int, Enum):
    CAMERA = auto()
    LIDAR = auto()


class SensorData:
    @property
    def indices(self) -> list[int]:
        raise NotImplementedError()

    @property
    def captions(self) -> list[str]:
        raise NotImplementedError()

    @property
    def vis_order(self) -> list[int]:
        return self.indices

    @property
    def num_sensors(self) -> int:
        return len(self.indices)

    def num_images(self, sensor_id: int) -> int:
        raise NotImplementedError()

    @property
    def total_num_images(self) -> int:
        return sum([
            self.num_images(sensor_id)
            for sensor_id in self.indices
        ])

    def get_camera(self, sensor_id: int, sample_idx: int) -> RSCamera:
        raise NotImplementedError()

    def get_cameras(self, sensor_id: int) -> RSCamera:
        return RSCamera.stack([
            self.get_camera(sensor_id, i)
            for i in range(self.num_images(sensor_id))
        ], dim=0)

    def get_mask(self, sensor_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_ego_se3_sensor(self, sensor_id: int) -> SE3:
        raise NotImplementedError()

    def get_loader(self) -> SensorLoader:
        pass
