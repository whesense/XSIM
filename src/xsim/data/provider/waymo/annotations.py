from typing import Optional

import pandas as pd
import torch

from toast import SE3, Quat, LinearMotion
from xsimgs.structures import OrientedBoxes, BoxSensorLink


from xsim.structures import SE3Trajectory
from xsim.structures.timestamp import Timestamp

from .enums import BoxType
from .lidar_data import WaymoLidarData
from .utils import df_to_tensor
from ...loaders.sensor_data import SensorType


def get_boxes_cmp(boxes, cmp='center'):
    BOX_CMP = '[LiDARBoxComponent].{}.{}'
    df_slice = boxes[[BOX_CMP.format(cmp, ax) for ax in ['x', 'y', 'z']]]
    return torch.as_tensor(df_slice.to_numpy().copy()).float()


def process_boxes(
        boxes_df: pd.DataFrame,
        ego_traj: SE3Trajectory,
        ref_ts: Optional[Timestamp] = None
):
    obj_mask = boxes_df['[LiDARBoxComponent].type'] != BoxType.UNKNOWN.value
    obj_mask &= boxes_df['[LiDARBoxComponent].type'] != BoxType.SIGN.value
    boxes_df = boxes_df[obj_mask]

    t = get_boxes_cmp(boxes_df, 'box.center')
    size = get_boxes_cmp(boxes_df, 'box.size')
    heading = df_to_tensor(boxes_df, '[LiDARBoxComponent].box.heading').float()
    heading_axes = torch.as_tensor([0, 0, 1]).float().view(1, 3)
    quats: Quat = Quat.from_axis_angle(heading_axes * heading.view(-1, 1))

    ts = Timestamp.from_us(df_to_tensor(boxes_df, 'key.frame_timestamp_micros'))
    if ref_ts is not None:
        ts = ts - ref_ts

    obj_ids = boxes_df['key.laser_object_id'].unique().tolist()
    unique_obj_ids = {obj_id: i for i, obj_id in enumerate(obj_ids)}
    ids = list(boxes_df['key.laser_object_id'])
    types = list(boxes_df['[LiDARBoxComponent].type'])
    id_to_type = {unique_obj_ids[ids[i]]: types[i] for i in range(len(ids))}

    obj_ids = torch.as_tensor(
        [unique_obj_ids[obj_id] for obj_id in boxes_df['key.laser_object_id']]).int()
    obj_types = torch.as_tensor(boxes_df['[LiDARBoxComponent].type'].tolist()).byte()

    velocity = get_boxes_cmp(boxes_df, 'speed')

    box_frame_idx = torch.searchsorted(
        ego_traj.time.value,
        ts.value
    )
    sensor_link = BoxSensorLink(
        sensor_type=torch.tensor(SensorType.LIDAR.value).broadcast_to((len(box_frame_idx), 1)),
        sensor_id=torch.tensor(WaymoLidarData._top_lidar_idx).broadcast_to(
            (len(box_frame_idx), 1)),
        sample_idx=box_frame_idx.view(-1, 1)
    )
    motion = LinearMotion(
        SE3(quats, t),
        pose_time=ts.sec.value.float().view(-1, 1),
        linear_velocity=velocity,
        angular_velocity=torch.zeros_like(velocity),
    )
    box_poses = ego_traj.pose.float()[box_frame_idx]
    motion = motion.transform(box_poses)

    boxes = OrientedBoxes(
        motion=motion,
        sizes=size,
        category=obj_types,
        object_ids=obj_ids,
        sensor_link=sensor_link
    )

    return boxes, box_frame_idx, unique_obj_ids, id_to_type
