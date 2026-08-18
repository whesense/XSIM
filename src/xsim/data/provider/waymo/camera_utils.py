from typing import Optional

import pandas as pd
import torch

from toast import SE3, Quat, LinearMotion

from xsim.structures.timestamp import Timestamp
from xsimgs.cameras import (
    RollingShutterDirection,
    DistortedPerspectiveCamera,
    RollingShutter
)

from .enums import RollingShutterReadOutDirection as WaymoShutterDirection
from .utils import df_to_tensor


INTRINSIC_KEY = '[CameraCalibrationComponent].intrinsic.{}'
EXTRINSIC_KEY = '[CameraCalibrationComponent].extrinsic.transform'
SHUTTER_KEY = '[CameraCalibrationComponent].rolling_shutter_direction'
VELOCITY_KEY = '[CameraImageComponent].velocity.{}_velocity.{}'
RS_KEY = '[CameraImageComponent].rolling_shutter_params.{}'

WAYMO_CAMERA_ROTATION = Quat(torch.as_tensor([0.5, -0.5, 0.5, -0.5])).double()


def parse_waymo_direction(waymo_dir: int):
    match waymo_dir:
        case WaymoShutterDirection.GLOBAL_SHUTTER.value:
            return RollingShutterDirection.GLOBAL_SHUTTER
        case WaymoShutterDirection.LEFT_TO_RIGHT.value:
            return RollingShutterDirection.LEFT_TO_RIGHT
        case WaymoShutterDirection.RIGHT_TO_LEFT.value:
            return RollingShutterDirection.RIGHT_TO_LEFT
        case WaymoShutterDirection.TOP_TO_BOTTOM.value:
            return RollingShutterDirection.TOP_TO_BOTTOM
        case WaymoShutterDirection.BOTTOM_TO_TOP.value:
            return RollingShutterDirection.BOTTOM_TO_TOP

    return WaymoShutterDirection.GLOBAL_SHUTTER


def process_cam_calib(calib_df: pd.DataFrame) -> dict:
    result = {}
    for _ ,cam_row in calib_df.iterrows():
        camera_name = cam_row['key.camera_name']
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = [
            cam_row[INTRINSIC_KEY.format(v)]
            for v in ['f_u', 'f_v', 'c_u', 'c_v', 'k1', 'k2', 'p1', 'p2', 'k3']
        ]
        extr = cam_row[EXTRINSIC_KEY].copy()
        ego_se3_cam = SE3.from_matrix(torch.as_tensor(extr.reshape(4, 4)))
        ego_se3_cam.q = ego_se3_cam.q @ WAYMO_CAMERA_ROTATION
        width, height = [
            cam_row['[CameraCalibrationComponent].{}'.format(v)]
            for v in ['width', 'height']
        ]
        result[camera_name] = dict(
            intrs = [fx, fy, cx, cy],
            dist = [k1, k2, p1, p2, k3, 0.0, 0.0, 0.0],
            width=width,
            height=height,
            shutter_direction=parse_waymo_direction(cam_row[SHUTTER_KEY]),
            ego_se3_cam=ego_se3_cam
        )

    return result


def parse_cameras(
        df_cam: pd.DataFrame,
        cur_cam_calib: dict,
        ref_ts: Optional[Timestamp] = None,
        ref_se3_world: Optional[SE3] = None
):
    pose_ts = Timestamp.from_sec(df_cam['[CameraImageComponent].pose_timestamp'].array)
    trigger_time = Timestamp.from_sec(df_cam[RS_KEY.format('camera_trigger_time')].array)
    final_time = Timestamp.from_sec(
        df_cam[RS_KEY.format('camera_readout_done_time')].array)

    if ref_ts is not None:
        pose_ts -= ref_ts
        trigger_time -= ref_ts
        final_time -= ref_ts

    ego_se3_camera = cur_cam_calib['ego_se3_cam'].double()
    world_se3_ego = SE3.from_matrix(
        df_to_tensor(df_cam, '[CameraImageComponent].pose.transform').view(-1, 4, 4)
    )
    velocity = torch.stack([
        df_to_tensor(df_cam, VELOCITY_KEY.format('linear', v))
        for v in 'xyz'
    ], dim=1).double()

    angular_velocity = torch.stack([
        df_to_tensor(df_cam, VELOCITY_KEY.format('angular', v))
        for v in 'xyz'
    ], dim=1).double()

    if ref_se3_world is not None:
        world_se3_ego = ref_se3_world.view(1) @ world_se3_ego
        velocity = ref_se3_world.q @ velocity

    lever = torch.cross(angular_velocity, ego_se3_camera.t.view(1, 3), dim=-1)
    velocity = velocity + world_se3_ego.q @ lever

    motion = LinearMotion(
        pose=(world_se3_ego @ ego_se3_camera).std(),
        pose_time=pose_ts.sec.value.unsqueeze(-1),
        linear_velocity=velocity,
        angular_velocity=world_se3_ego.q @ angular_velocity
    )

    exposure = Timestamp.from_sec(df_cam[RS_KEY.format('shutter')].array)
    trigger_time = trigger_time + 0.5 * exposure
    duration = final_time - trigger_time - exposure
    shutter_dir = cur_cam_calib['shutter_direction']

    dist = torch.as_tensor(cur_cam_calib['dist']).view(1, -1).expand(len(motion), -1)
    intrinsics = DistortedPerspectiveCamera.intrinsics_from_sized(
        *cur_cam_calib['intrs'],
        width=cur_cam_calib['width'],
        height=cur_cam_calib['height']
    ).view(1, -1).expand(len(motion), -1)

    new_camera = DistortedPerspectiveCamera(
        world_se3_camera=motion,
        shutter=RollingShutter.from_type(
            trigger_time.sec.value.unsqueeze(-1),
            duration.sec.value.unsqueeze(-1),
            shutter_dir,
            exposure_time=exposure.sec.value.unsqueeze(-1),
        ),
        normalized_intrinsics=intrinsics,
        distortion=dist,
        camera_name=int(df_cam['key.camera_name'].iloc[0]),
    ).float()

    return new_camera
