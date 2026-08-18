from copy import deepcopy
from pathlib import Path
from typing import Optional

import pandas as pd
from toast import SE3
import torch
import torchvision as tv

from xsimgs.cameras import DistortedPerspectiveCamera
from xsim.structures import Timestamp
from xsim.data.loaders.sensor_data import SensorData, SensorLoader

from .camera_utils import process_cam_calib, parse_cameras


class WaymoCameraLoader(SensorLoader):
    def __init__(self, img_df: dict[int, pd.DataFrame]):
        self.img_df = img_df

    def __call__(self, sensor_id: int, sample_idx: int) -> torch.Tensor:
        row = self.img_df[sensor_id].iloc[sample_idx]
        img = torch.frombuffer(
            bytearray(row['[CameraImageComponent].image']),
            dtype=torch.uint8
        )
        img = tv.io.decode_jpeg(img, device='cpu')
        return img



class WaymoCameraData(SensorData):
    _camera_captions: dict[int, str] = {
        1: "front_camera",
        2: "front_left_camera",
        3: "front_right_camera",
        4: "left_camera",
        5: "right_camera"
    }
    _default_camera_indices: list[int] = list(range(1, 6))
    _default_vis_camera_order: list[int] = [4, 2, 1, 3, 5]

    def __init__(
            self,
            root_path: Path,
            segment_name: str,
            ref_ts: Timestamp,
            ref_se3_world: SE3,
            use_cameras: Optional[list[int]] = None,
    ):
        self.root_path = root_path
        self.segment_name = segment_name
        self.ref_ts = ref_ts
        self.ref_se3_world = ref_se3_world

        self._camera_indices = (
            use_cameras if use_cameras is not None
            else deepcopy(self._default_camera_indices)
        )
        self._vis_camera_order = (
            deepcopy(self._default_vis_camera_order)
            if use_cameras is None else
            [i for i in self._default_vis_camera_order if i in use_cameras]
        )
        self.camera_calib = process_cam_calib(self._read_data('camera_calibration'))

        img_df = self._read_data('camera_image')
        img_df = img_df.sort_values(
            by=['key.camera_name', 'key.frame_timestamp_micros'])

        self.img_df_slices = {
            camera_idx: img_df[img_df['key.camera_name'] == camera_idx]
            for camera_idx in self.indices
        }
        self.cameras = {
            camera_idx: parse_cameras(
                self.img_df_slices[camera_idx],
                self.camera_calib[camera_idx],
                ref_ts=self.ref_ts,
                ref_se3_world=self.ref_se3_world,
            )
            for camera_idx in self._camera_indices
        }

    def _read_data(self, modality: str) -> pd.DataFrame:
        return pd.read_parquet(
            self.root_path / modality / (self.segment_name + '.parquet'))

    @property
    def indices(self) -> list[int]:
        return deepcopy(self._camera_indices)

    @property
    def captions(self) -> list[str]:
        return [
            self._camera_captions[camera_idx]
            for camera_idx in self.indices
        ]

    @property
    def vis_order(self) -> list[int]:
        return deepcopy(self._vis_camera_order)

    def num_images(self, sensor_id: int) -> int:
        return len(self.cameras[sensor_id])

    def get_camera(self, sensor_id: int, sample_idx: int) -> DistortedPerspectiveCamera:
        return self.cameras[sensor_id][sample_idx]

    def get_cameras(self, sensor_id: int) -> DistortedPerspectiveCamera:
        return self.cameras[sensor_id]

    def get_mask(self, sensor_id: int) -> torch.Tensor:
        calib = self.camera_calib[sensor_id]
        return torch.ones(calib['height'], calib['width'], dtype=torch.bool)

    def get_ego_se3_sensor(self, sensor_id: int) -> SE3:
        return self.camera_calib[sensor_id]['ego_se3_cam']

    def get_loader(self) -> SensorLoader:
        return WaymoCameraLoader(self.img_df_slices)
