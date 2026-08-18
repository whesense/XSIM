from copy import deepcopy
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from toast import SE3
from xsimgs.cameras import RSCamera

from xsim.data.loaders.sensor_data import SensorData, SensorLoader
from xsim.structures import Sweep, Timestamp, SE3Trajectory
from .lidar_utils import process_lidar_calib, load_top_lidar, load_side_lidar


class WaymoLidarLoader(SensorLoader):
    preloaded = True

    def __init__(self, lidar_data):
        self.lidar_data = lidar_data

    def __call__(self, sensor_id: int, sample_idx: int) -> Sweep:
        return self.lidar_data.sweeps[sensor_id][sample_idx]


class WaymoLidarData(SensorData):
    _lidar_captions: dict[int, str] = {
        1: 'lidar_top',
        2: 'lidar_front',
        3: 'lidar_side_left',
        4: 'lidar_side_right',
        5: 'lidar_rear'
    }
    _top_lidar_idx: int = 1
    _side_lidar_indices: list[int] = list(range(2, 6))

    def __init__(
            self,
            root_path: Path,
            segment_name: str,
            scene_cache_path: Path,
            ego_traj: SE3Trajectory,
            ref_ts: Timestamp,
            ref_se3_world: SE3,
            use_lidars: Optional[list[int]] = None
    ):
        self.root_path = root_path
        self.segment_name = segment_name
        self.scene_cache_path = scene_cache_path
        self.use_lidars = use_lidars
        self._lidar_indices = (
            use_lidars if use_lidars is not None
            else [self._top_lidar_idx]
        )
        self.ego_traj = ego_traj
        self.ref_ts = ref_ts
        self.ref_se3_world = ref_se3_world
        self.lidar_calib = process_lidar_calib(self._read_data('lidar_calibration'))

        self.sweeps = {}
        self._lidar_cameras = {}
        self.rng_df = None

        for lidar_idx in self._lidar_indices:
            if lidar_idx == self._top_lidar_idx:
                sweeps, cams = load_top_lidar(self)
            else:
                sweeps, cams = load_side_lidar(self, lidar_idx)
            self.sweeps[lidar_idx] = sweeps
            self._lidar_cameras[lidar_idx] = cams

    def _read_data(self, modality: str) -> pd.DataFrame:
        return pd.read_parquet(
            self.root_path / modality / (self.segment_name + '.parquet'))

    def _load_rng_df(self):
        if self.rng_df is not None:
            return

        print('Loading LiDAR dataframe...')
        rng_df = self._read_data('lidar')
        rng_df = rng_df.sort_values(
            by=['key.laser_name', 'key.frame_timestamp_micros'])
        self.rng_df = {
            lidar_idx: rng_df[rng_df['key.laser_name'] == lidar_idx]
            for lidar_idx in self._lidar_indices
        }

    @property
    def indices(self) -> list[int]:
        return deepcopy(self._lidar_indices)

    @property
    def captions(self) -> list[str]:
        return [self._lidar_captions[lidar_idx] for lidar_idx in self._lidar_indices]

    def num_images(self, sensor_id: int) -> int:
        return len(self.sweeps[sensor_id])

    def get_camera(self, sensor_id: int, sample_idx: int) -> RSCamera:
        return self._lidar_cameras[sensor_id][sample_idx]

    def get_cameras(self, sensor_id: int) -> RSCamera:
        return self._lidar_cameras[sensor_id]

    def get_mask(self, sensor_id: int) -> torch.Tensor:
        return torch.ones(*self.sweeps[sensor_id][0].shape, dtype=torch.bool)

    def get_ego_se3_sensor(self, sensor_id: int) -> SE3:
        return self.lidar_calib[sensor_id]['ego_se3_lidar']

    def get_loader(self) -> SensorLoader:
        return WaymoLidarLoader(self)
