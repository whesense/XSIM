import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from jetcon import read
from toast import SE3

from xsimgs.structures import OrientedBoxes
from xsim.structures import Sweep, SE3Trajectory, Timestamp
from .camera_data import WaymoCameraData

from .annotations import process_boxes
from .enums import BoxType
from .lidar_data import WaymoLidarData
from .utils import df_to_tensor
from ..basic_scene import DatasetScene
from ...loaders import SensorLoader
from ...loaders.image_loaders import PathBasedImageLoader
from ...loaders.sensor_data import SensorType
from ...utils.camera_segmentation import save_segmentation_masks
from ...utils.smpl import process_smpl_instance, SMPLData
from ...utils.timestamp_correction import update_boxes_time, TimeCorrectionMethod


WAYMO_SCENES_FILE = 'configs/data/waymo/scene_names.yaml'

# Short names for the split a root path points at, for scene_alias.
WAYMO_SPLIT_ALIASES = {
    'training': 'train',
    'validation': 'val',
    'testing': 'test',
}


class WaymoSceneProvider(DatasetScene):
    camera: WaymoCameraData
    lidar: WaymoLidarData

    def __init__(
            self,
            root_path: Path | str,
            cache_path: Path | str,
            smpl_data_path: str,
            scene_idx: str | int,
            use_cameras: Optional[list[int]] = None,
            use_lidars: Optional[list[int]] = None,
            ref_sample_idx: int = 0,
            box_time_correction_method: TimeCorrectionMethod = 'box_center_unwrap',
            ignore_segmentation: bool = False
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.cache_path = Path(cache_path)
        self.scene_idx = scene_idx
        self.smpl_data_path = smpl_data_path
        self.ref_sample_idx = ref_sample_idx

        if isinstance(scene_idx, int):
            if (self.smpl_data_path is not None) and ("{" in self.smpl_data_path):
                self.smpl_data_path = self.smpl_data_path.format(scene_idx)
            segment_name = read(WAYMO_SCENES_FILE).train[scene_idx]
        else:
            segment_name = scene_idx

        self.segment_name = segment_name
        self.scene_cache_path = self.cache_path / self.segment_name

        self.ref_ts, self.ref_se3_world, self.ego_traj = self._read_ego_poses()

        # read boxes in ego coordinates
        (
            self.boxes, self.boxes_frame_idx,
            self.unique_obj_ids, self.id_to_type
        ) = process_boxes(
            self._read_data('lidar_box'),
            ego_traj=self.ego_traj,
            ref_ts=self.ref_ts
        )

        self._vehicle_ids = [
            obj_id for obj_id in self.id_to_type
            if self.id_to_type[obj_id] == BoxType.VEHICLE.value
        ]
        human_category = [BoxType.PEDESTRIAN.value, BoxType.CYCLIST.value]
        self._human_ids = [
            obj_id for obj_id in self.id_to_type
            if self.id_to_type[obj_id] in human_category
        ]

        self.sensors[SensorType.CAMERA] = WaymoCameraData(
            self.root_path, self.segment_name,
            ref_ts=self.ref_ts, ref_se3_world=self.ref_se3_world,
            use_cameras=use_cameras
        )
        self.sensors[SensorType.LIDAR] = WaymoLidarData(
            self.root_path, self.segment_name, self.scene_cache_path, self.ego_traj,
            ref_ts=self.ref_ts, ref_se3_world=self.ref_se3_world,
            use_lidars=use_lidars
        )

        update_boxes_time(
            self.sensors,
            self.boxes,
            method=box_time_correction_method
        )

        world_so3b_cam = SE3.stack([
            self.cameras.get_cameras(sensor_id).world_se3_camera.pose.q
            for sensor_id in self.cameras.indices
        ], dim=0).std()

        self.smpl_data = {}

        if self.smpl_data_path is not None:
            with Path(self.smpl_data_path).open('rb') as f:
                smpl_data = pickle.load(f)

            smpl_ids = sorted(list(smpl_data.keys()))

            self.smpl_data = {
                inst_id: process_smpl_instance(
                    smpl_data[inst_id],
                    self.get_boxes(inst_id),
                    world_so3b_cam
                ) for inst_id in smpl_ids
            }

        self.seg_path_fmt = str(
            self.scene_cache_path / 'seg_cam_{sensor_id:}/frame_{sample_idx:03d}.jpg'
        )
        test_path = self.seg_path_fmt.format(
            sensor_id=self.cameras.indices[0], sample_idx=0
        )
        if (not os.path.isfile(test_path)) and (not ignore_segmentation):
            save_segmentation_masks(self.cameras, self.seg_path_fmt)

    def _read_data(self, modality: str) -> pd.DataFrame:
        return pd.read_parquet(
            self.root_path / modality / (self.segment_name + '.parquet'))

    def _read_ego_poses(self) -> tuple[Timestamp, SE3, SE3Trajectory]:
        poses = self._read_data('vehicle_pose')
        world_to_ego_key = '[VehiclePoseComponent].world_from_vehicle.transform'
        world_se3_ego = df_to_tensor(poses, world_to_ego_key).view(-1, 4, 4)
        world_se3_ego = SE3.from_matrix(world_se3_ego)
        ts = Timestamp.from_us(df_to_tensor(poses, 'key.frame_timestamp_micros'))
        ref_ts = ts[self.ref_sample_idx]
        ref_se3_world = world_se3_ego[self.ref_sample_idx].inv()

        ref_se3_ego = ref_se3_world.view(1) @ world_se3_ego
        ts = ts - ref_ts

        return ref_ts, ref_se3_world, SE3Trajectory(ref_se3_ego, ts)

    @property
    def scene_alias(self) -> str:
        split = WAYMO_SPLIT_ALIASES.get(self.root_path.name, self.root_path.name)
        return '{}-{}'.format(split, super().scene_alias)

    @property
    def num_objects(self) -> int:
        return len(self.unique_obj_ids)

    @property
    def vehicle_ids(self) -> list[int]:
        return deepcopy(self._vehicle_ids)

    @property
    def human_ids(self) -> list[int]:
        return deepcopy(self._human_ids)

    def get_boxes(self, object_id: int) -> OrientedBoxes:
        return self.boxes[self.boxes.object_ids == object_id]

    def get_segmentation_loader(self) -> SensorLoader:
        return PathBasedImageLoader(self.seg_path_fmt)

    def get_smpl_data(self, object_id: int) -> SMPLData | None:
        return self.smpl_data.get(object_id, None)
