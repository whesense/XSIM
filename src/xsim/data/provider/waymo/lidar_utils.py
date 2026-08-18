import pandas as pd
import torch
from toast import SE3, Quat, LinearMotion

from xsimgs.cameras import RollingShutter, LidarCamera, SphericalCamera
from xsim.structures import Sweep, SE3Trajectory
from xsim.utils import track


def process_lidar_calib(calib_df):
    TOP_LIDAR_SWEEP = 2650
    TOP_LIDAR_RAYS = 64
    SIDE_LIDAR_SWEEP = 600
    SIDE_LIDAR_RAYS = 200
    EXTR_KEY = '[LiDARCalibrationComponent].extrinsic.transform'

    result = {}
    for _, row in calib_df.iterrows():
        laser_name = row['key.laser_name']
        extrinsic = torch.as_tensor(row[EXTR_KEY].copy()).view(4, 4)
        beam_incl_min = row['[LiDARCalibrationComponent].beam_inclination.min']
        beam_incl_max = row['[LiDARCalibrationComponent].beam_inclination.max']
        beam_incls = row['[LiDARCalibrationComponent].beam_inclination.values']
        beam_incls = torch.as_tensor(beam_incls.copy()) if beam_incls is not None else None

        az_offset = 0.5 * torch.pi - torch.atan2(extrinsic[1, 0], extrinsic[0, 0])
        quat_az = Quat.from_axis_angle(torch.as_tensor(
            [0.0, 0.0, az_offset], dtype=torch.double))
        ego_se3_lidar_raw = SE3.from_matrix(extrinsic)
        ego_se3_lidar = ego_se3_lidar_raw.clone()
        ego_se3_lidar.q = ego_se3_lidar.q @ quat_az

        result[laser_name] = dict(
            laser_name=int(laser_name),
            extrinsic=extrinsic,
            beam_incl_min=beam_incl_min,
            beam_incl_max=beam_incl_max,
            beam_incls=beam_incls,
            width=TOP_LIDAR_SWEEP if laser_name == 1 else SIDE_LIDAR_SWEEP,
            height=TOP_LIDAR_RAYS if laser_name == 1 else SIDE_LIDAR_RAYS,
            ego_se3_lidar=ego_se3_lidar,
            ego_se3_lidar_raw=ego_se3_lidar_raw,
        )
        result[laser_name]['lidar_rays'] = precompute_lidar_rays(result[laser_name])


    return result


def rng_image(row, component='LiDARComponent', return_idx=1):
    key_template = '[{}].range_image_return{}.{}'

    shape = row[key_template.format(component, return_idx, 'shape')].tolist()
    img = row[key_template.format(component, return_idx, 'values')].reshape(*shape)
    return torch.as_tensor(img.copy())

def pose_img_to_se3(pose_img):
    return SE3(Quat.from_euler_angles(pose_img[..., :3]), pose_img[..., 3:])

def lidar_pose_image(df_row) -> SE3:
    return pose_img_to_se3(rng_image(df_row, 'LiDARPoseComponent'))

def lidar_rng_image(df_row) -> torch.Tensor:
    return rng_image(df_row, 'LiDARComponent')


def get_t(points, ref_point_a, ref_point_b):
    d = ref_point_b - ref_point_a
    t = ((points - ref_point_a) * d).sum(dim=-1) / (d * d).sum(dim=-1)
    mask = (t >= 0) & (t <= 1)
    return t.clamp(0, 1), mask


def get_lidar_time(
        poses: torch.Tensor, # [H, W, 3]
        world_se3b_ego: SE3, # [N,]
        frame_time: torch.Tensor, # [N, ]
        frame_idx: int
):
    prev_t, cur_t, next_t = world_se3b_ego[frame_idx - 1: frame_idx + 2].t.float()
    left_t, left_mask = get_t(poses, prev_t, cur_t)
    right_t, right_mask = get_t(poses, cur_t, next_t)

    left_ts = torch.lerp(frame_time[frame_idx - 1], frame_time[frame_idx], left_t)
    right_ts = torch.lerp(frame_time[frame_idx], frame_time[frame_idx + 1], right_t)
    ts = torch.where(left_mask, left_ts, right_ts)
    return ts, left_mask | right_mask





def get_inclinations(cur_lidar_calib):
    if cur_lidar_calib['beam_incls'] is not None:
        return cur_lidar_calib['beam_incls']

    height = cur_lidar_calib['height']
    incl_min, incl_max = cur_lidar_calib['beam_incl_min'], cur_lidar_calib[
        'beam_incl_max']
    return torch.arange(height).add(0.5).mul((incl_max - incl_min) / height).add(incl_min)


def precompute_lidar_rays(cur_lidar_calib: dict):
    # method is adapted from official waymo toolkit
    height, width = cur_lidar_calib['height'], cur_lidar_calib['width']
    inclination = torch.flip(get_inclinations(cur_lidar_calib), dims=[0, ]).float()
    extrinsic = cur_lidar_calib['extrinsic']

    az_correction = torch.atan2(extrinsic[1, 0], extrinsic[0, 0])
    azimuth = torch.arange(width, 0, -1) * (2 * torch.pi / width) - (
                (width + 1) * torch.pi / width + az_correction)
    azimuth_tile = azimuth.view(1, -1).expand(height, -1)
    incl_tile = inclination.view(-1, 1).expand(-1, width)

    cos_azimuth, sin_azimuth = azimuth_tile.cos(), azimuth_tile.sin()
    cos_incl, sin_incl = incl_tile.cos(), incl_tile.sin()

    points_lidar = torch.stack([
        cos_azimuth * cos_incl,
        sin_azimuth * cos_incl,
        sin_incl
    ], dim=-1)
    return points_lidar



def process_top_lidar(
        scene,
        top_lidar_pose_df,
        min_translation_threshold: float = 0.05,
        target_lidar_shutter_ms: float = 100.0,
        max_lidar_shutter_dev_ms: float = 2.0,
        ego_filter_dist: float = 2.0
):
    N_AZIMUTH_OUTLIERS = 24

    cameras: list[LidarCamera | None] = [None] * len(scene.ego_traj)
    sweeps: list[Sweep] = [None] * len(scene.ego_traj)
    sweep_time_valid = torch.zeros(len(scene.ego_traj), dtype=torch.bool)

    poses = []

    for frame_idx in track(range(len(scene.ego_traj)), description='Processing lidar_top'):
        img = rng_image(scene.rng_df[1].iloc[frame_idx])[..., :2]
        ray_lengths, ray_intensity = img.unbind(dim=-1)
        mask = ray_lengths > ego_filter_dist
        # pose_img = world_se3_ego [H, W]
        pose_img = lidar_pose_image(top_lidar_pose_df.iloc[frame_idx]).double()
        poses_ref_se3_ego = scene.ref_se3_world @ pose_img
        poses_ref_se3_lidar = poses_ref_se3_ego @ scene.get_ego_se3_sensor(1).view(1, 1)
        poses.append(poses_ref_se3_lidar)

        points_lidar = scene.lidar_calib[1]['lidar_rays'] * ray_lengths.unsqueeze(-1)
        # raw = calibration from dataset, without azimuth offset correction
        ego_se3_lidar_raw = scene.lidar_calib[1]['ego_se3_lidar_raw'].view(1)
        points_ego = ego_se3_lidar_raw @ points_lidar.double()
        points_world = poses_ref_se3_ego @ points_ego
        points_origins = (poses_ref_se3_ego @ ego_se3_lidar_raw.view(1, 1)).t

        sweep = Sweep(
            xyz=points_world.float(),
            origin=points_origins.float(),
            intensity=ray_intensity.float(),
            mask=mask
        )
        sweeps[frame_idx] = sweep

        # Per-point timestamps for LiDARs are not given in Waymo dataset.
        # Instead, per-point world_se3_ego poses are provided. Based on them, and
        # reference world_se3_ego poses with timestamps we estimate point timestamps
        # This is done via projecting per-point translations onto ego trajectory
        # Method working correctly only when vehicle is moving with > 2.5km/h velocity
        if frame_idx == 0 or frame_idx == len(scene.ego_traj) - 1:
            continue
        prev_pose, cur_pose, next_pose = scene.ego_traj.pose.t[frame_idx - 1: frame_idx + 2]
        prev_small = (prev_pose - cur_pose).norm() < min_translation_threshold
        next_small = (next_pose - cur_pose).norm() < min_translation_threshold
        if prev_small or next_small:
            continue

        time_img, time_mask = get_lidar_time(
            poses_ref_se3_ego.t, scene.ego_traj.pose, scene.ego_traj.time.sec.value,
            frame_idx # current frame to limit search space
        )
        total_mask = mask & time_mask
        # LiDAR beams are azimuthally offset from each other, so first and last N
        # azimuth returns are wrapped and do not satisfy linear shutter condition
        # discard them for fitting shutter parameters as outliers
        total_mask[:, :N_AZIMUTH_OUTLIERS] = False
        total_mask[:, -N_AZIMUTH_OUTLIERS:] = False
        shutter = RollingShutter.from_time_image(
            time_image=time_img.float(),
            mask=total_mask,
            # LiDAR beams are slightly offset in time from each other
            # to restrict rolling shutter estimate to only horizontal direction
            horizontal_only=True
        )
        shutter_dev_ms = abs(shutter.duration[0].item()*1000 - target_lidar_shutter_ms)
        if shutter_dev_ms > max_lidar_shutter_dev_ms:
            continue

        sweep.time = time_img.float()

        valid_poses = poses_ref_se3_lidar[time_mask].std()
        valid_time = time_img[time_mask].view(-1, 1)
        motion: LinearMotion = LinearMotion.fit(valid_poses, valid_time)
        motion.pose = motion.pose.std()

        camera = LidarCamera.create(
            world_se3_camera=motion,
            shutter=shutter,
            beam_angles=scene.lidar_calib[1]['beam_incls'].flip(dims=[0]).float(),
            phi_min=torch.tensor(0),
            phi_range=torch.tensor(2 * torch.pi)
        )
        cameras[frame_idx] = camera.float()
        sweep_time_valid[frame_idx] = True

    num_valid = sweep_time_valid.sum().item()
    print('Successfully estimated LiDAR time: {:d}/{:d}'.format(
        num_valid, len(sweep_time_valid)
    ))


    valid_idxs = sweep_time_valid.nonzero().flatten()
    # process remaining frames with no valid estimation
    for frame_idx in range(len(scene.ego_traj)):
        if sweep_time_valid[frame_idx]:
            continue

        closest_valid_frame: int = valid_idxs[(valid_idxs - frame_idx).abs().argmin()].item()
        closest_shutter = cameras[closest_valid_frame].shutter

        cur_t = scene.ego_traj.time[frame_idx].sec.value.item()
        val_t = scene.ego_traj.time[closest_valid_frame].sec.value.item()
        time_delta = cur_t - val_t
        cur_t = closest_shutter.image_time(sweeps[closest_valid_frame]) + time_delta

        sweeps[frame_idx].time = cur_t.unsqueeze(-1).float()
        shutter = closest_shutter.clone()
        shutter.offset_time += time_delta

        total_mask = sweeps[frame_idx].mask.clone()
        total_mask[:, :N_AZIMUTH_OUTLIERS] = False
        total_mask[:, -N_AZIMUTH_OUTLIERS:] = False

        valid_poses = poses[frame_idx][total_mask].std()
        valid_time = cur_t[total_mask].view(-1, 1).double()
        motion: LinearMotion = LinearMotion.fit(valid_poses, valid_time)
        motion.pose = motion.pose.std()

        camera = LidarCamera.create(
            world_se3_camera=motion,
            shutter=shutter,
            beam_angles=scene.lidar_calib[1]['beam_incls'].flip(dims=[0]).float(),
            phi_min=torch.tensor(0),
            phi_range=torch.tensor(2 * torch.pi),
            camera_name=1
        )
        cameras[frame_idx] = camera.float()

    return sweeps, LidarCamera.stack(cameras, dim=0)


def load_top_lidar(scene):
    sweeps_cache_path = scene.scene_cache_path / 'lidar_top_sweeps.pth'
    cameras_cache_path = scene.scene_cache_path / 'lidar_top_cameras.pth'
    if not sweeps_cache_path.exists() or not cameras_cache_path.exists():
        sweeps_cache_path.parent.mkdir(parents=True, exist_ok=True)

        scene._load_rng_df()
        top_lidar_pose_df = scene._read_data('lidar_pose')

        sweeps, lidar_cameras = process_top_lidar(scene, top_lidar_pose_df)

        torch.save(sweeps, sweeps_cache_path)
        torch.save(lidar_cameras, cameras_cache_path)
    else:
        sweeps = torch.load(sweeps_cache_path, weights_only=False)
        lidar_cameras = torch.load(cameras_cache_path, weights_only=False)

    return sweeps, lidar_cameras


def process_side_lidar(
        rng_df: pd.DataFrame,
        ego_traj: SE3Trajectory,
        lidar_calib: dict
):
    num_frames = len(rng_df)
    ref_se3_ego = ego_traj.pose
    reg_se3_ego_time = ego_traj.time.sec.value

    ego_se3_lidar = lidar_calib['ego_se3_lidar'].view(1)

    ref_se3_lidar = LinearMotion(
        pose=ref_se3_ego @ ego_se3_lidar,
        pose_time=reg_se3_ego_time,
        linear_velocity=torch.zeros(num_frames, 3),
        angular_velocity=torch.zeros(num_frames, 3),
    )
    shutter = RollingShutter(
        offset_time=reg_se3_ego_time.view(-1, 1),
        duration=torch.zeros(num_frames, 2),
    )
    theta_min = lidar_calib['beam_incl_min']
    theta_max = lidar_calib['beam_incl_max']
    limits = torch.tensor([0, 2*torch.pi, theta_max, theta_min - theta_max])
    limits = limits.view(1, 4).expand(num_frames, -1)

    cameras = SphericalCamera(
        world_se3_camera=ref_se3_lidar,
        shutter=shutter,
        limits=limits,
        camera_name=lidar_calib['laser_name']
    ).float()
    imgs = torch.stack([
        rng_image(rng_df.iloc[i])[..., :2] for i in range(num_frames)
    ], dim=0)
    return imgs, cameras


def side_lidar_sweep(img: torch.Tensor, camera: SphericalCamera):
    height, width, _ = img.shape
    ray_o, ray_d = camera.camera_rays(width=width, height=height)
    ray_lengths, ray_intensity = img.unbind(dim=-1)
    mask = ray_lengths > 0
    points = ray_o + ray_d * ray_lengths.unsqueeze(-1)
    sweep = Sweep(
        xyz=points,
        origin=ray_o,
        mask=mask,
        intensity=ray_intensity
    )
    return sweep


def load_side_lidar(scene, lidar_idx: int):
    cur_caption = scene._lidar_captions[lidar_idx]

    sweeps_cache_path = scene.scene_cache_path / '{}_sweeps.pth'.format(cur_caption)
    cameras_cache_path = scene.scene_cache_path / '{}_cameras.pth'.format(cur_caption)

    if not sweeps_cache_path.exists() or not cameras_cache_path.exists():
        print('Cache for "{}" does not exist, processing...'.format(cur_caption))
        sweeps_cache_path.parent.mkdir(parents=True, exist_ok=True)
        scene._load_rng_df()

        imgs, cameras = process_side_lidar(
            scene.rng_df[lidar_idx],
            scene.ego_traj,
            scene.lidar_calib[lidar_idx]
        )
        torch.save(imgs, sweeps_cache_path)
        torch.save(cameras, cameras_cache_path)
    else:
        imgs = torch.load(sweeps_cache_path)
        cameras = torch.load(cameras_cache_path, weights_only=False)

    sweeps = [side_lidar_sweep(img, camera) for img, camera in zip(imgs, cameras)]
    return sweeps, cameras
