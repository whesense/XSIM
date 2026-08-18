from typing import Literal

import numpy as np
import torch

from toast import SE3, LinearMotion, Quat
from xsimgs.cameras import RollingShutter, LidarCamera,rasterize_into_depth_mask
from xsimgs.cameras.torch.spherical import cartesian_to_spherical

from xsim.structures import Sweep, Timestamp
from xsim.structures.timestamp import TimestampScale
from xsim.data.utils.lidar_models import LidarModel


DEGREES_PER_TURN = 360.0


def read_lidar_pcd(
        path,
        base_ts: Timestamp,
        timestamp_field: str,
        timestamp_scale: TimestampScale,
        beam_idx_field: str | None = None,
        intensity_field: str | None = None,
) -> Sweep:
    from pypcd4 import PointCloud

    pcd = PointCloud.from_path(path)
    xyz = torch.as_tensor(pcd[['x', 'y', 'z']].numpy()).view(-1, 3).float()

    # Subtract in double: a nanosecond stamp needs more than float32's ~7 digits,
    # and only becomes a small relative offset once base_ts is removed.
    raw_time = torch.as_tensor(pcd[timestamp_field].numpy()).view(-1).double()
    time = (Timestamp(raw_time, timestamp_scale) - base_ts).sec.value.float()

    beam_idx = None
    if beam_idx_field is not None:
        beam_idx = torch.as_tensor(
            pcd[beam_idx_field].numpy().astype(np.uint8)).view(-1)

    intensity = None
    if intensity_field is not None:
        intensity = torch.as_tensor(pcd[intensity_field].numpy()).view(-1)

    return Sweep(
        xyz=xyz,
        origin=torch.zeros(3),
        intensity=intensity,
        beam_idx=beam_idx,
        time=time,
    )


def sweep_azimuth(xyz: torch.Tensor) -> torch.Tensor:
    """Azimuth of lidar-space points in degrees, in xsimgs' convention.

    That convention runs clockwise, so azimuth grows with time on these
    sensors, which is the direction a range image column index has to grow in.
    """
    azimuth = cartesian_to_spherical(xyz)[..., 0]
    return torch.rad2deg(azimuth) % DEGREES_PER_TURN


def fit_spin_rate(azimuth: torch.Tensor, time: torch.Tensor) -> float:
    """Fit how fast a sweep turns, in degrees per second.

    Azimuth is linear in time within a sweep, so unwrapping it along time and
    fitting a line recovers the rate without assuming a nominal one.
    """
    order = time.argsort()
    y = torch.from_numpy(
        np.unwrap(azimuth[order].double().numpy(), period=DEGREES_PER_TURN)
    ).unsqueeze(1)
    X = torch.stack([
        time[order].double(),
        torch.ones(len(time), dtype=torch.double)
    ], dim=1)

    return float(torch.linalg.lstsq(X, y).solution[0])


def circular_median(angles: torch.Tensor) -> float:
    """Median of angles in degrees, for angles gathered in one narrow arc."""
    radians = torch.deg2rad(angles)
    center = torch.rad2deg(torch.atan2(radians.sin().mean(), radians.cos().mean()))
    centered = (angles - center + DEGREES_PER_TURN / 2) % DEGREES_PER_TURN - DEGREES_PER_TURN / 2

    return float((centered.median() + center) % DEGREES_PER_TURN)


def compute_azimuth_extent(
        sweeps: list[tuple[torch.Tensor, torch.Tensor]],
        spin_rate: float,
        margin: float = 1.0,
) -> tuple[float, float]:
    """Azimuth a lidar's range image starts at, and how much of a turn it spans.

    Feeds ``phi_min`` and ``phi_range`` of an xsimgs spherical camera's limits,
    once converted to radians. Column zero lands on the azimuth a sweep is cut
    at, so the jump from a sweep's last point back to its first sits on the
    image edge, where a shutter linear in image space expects it.

    A lidar that only sweeps an arc gets cropped to it, rather than paying for a
    turn's worth of empty columns. The arc is measured across sweeps and not
    within one: the span a single sweep covers is bounded by where its returns
    happen to fall, so it understates an arc whose edges are pointed at sky,
    while the widest sweep seen approaches the arc the lidar truly scans.

    Args:
        sweeps: Per sweep of a single lidar, its point azimuths in degrees (as
            from ``sweep_azimuth``) and its point times in seconds relative to
            that same sweep. The more sweeps, the tighter the arc is pinned.
        spin_rate: Degrees per second, as from ``fit_spin_rate``.
        margin: Degrees of slack left either side of the arc, so that points on
            its edge do not fall out of the image by rounding alone.

    Returns:
        The azimuth column zero starts at and the azimuth the image spans, both
        in degrees.
    """
    seams = torch.tensor([
        circular_median(
            (azimuth - spin_rate * (time - time.min())) % DEGREES_PER_TURN
        ) for azimuth, time in sweeps
    ], dtype=torch.double)
    seam = circular_median(seams)

    # Sweeps are cut at slightly differing azimuths, so measure each one's arc
    # against the common seam. Quantiles rather than extremes: a stray return
    # carries a time from outside its own sweep.
    starts = (seams - seam + DEGREES_PER_TURN / 2) % DEGREES_PER_TURN - DEGREES_PER_TURN / 2
    spans = torch.tensor([
        spin_rate * float(time.quantile(0.9999) - time.quantile(0.0001))
        for _, time in sweeps
    ], dtype=torch.double)

    arc_start = float(starts.min()) - margin
    arc_end = float((starts + spans).max()) + margin
    if arc_end - arc_start >= DEGREES_PER_TURN:
        return seam, DEGREES_PER_TURN

    return (seam + arc_start) % DEGREES_PER_TURN, arc_end - arc_start


def fit_azimuth_extent(
        raw_sweeps: list[Sweep],
        min_points_for_spin_fit: int = 100,
) -> tuple[float, float]:
    """Fit the azimuth a lidar's range image starts at and the azimuth it spans.

    The pair a lidar's range images rest on, fit from the returns themselves and
    shared by every one of its sweeps (see ``compute_azimuth_extent``). Kept
    apart from ``build_range_sweeps`` so that reading an offset back -- to undo
    the pose spin it produced, say -- costs nothing but the fit.

    Args:
        raw_sweeps: The lidar's unprojected sweeps, in capture order. Per-point
            times need only be consistent within a sweep, since both the spin
            rate and the seam are measured inside one.
        min_points_for_spin_fit: A sweep must have at least this many valid
            returns to join the spin-rate fit; see ``build_range_sweeps``.

    Returns:
        The azimuth column zero starts at and the azimuth the image spans, both
        in degrees.
    """
    valid = [raw_sweep.xyz.norm(dim=-1) > 1e-3 for raw_sweep in raw_sweeps]
    azimuth_time = [
        (sweep_azimuth(raw_sweep.xyz[keep].float()).double(), raw_sweep.time[keep].double())
        for raw_sweep, keep in zip(raw_sweeps, valid)
    ]
    # Azimuth is monotonic in time only within a sweep, so the spin rate is fit
    # per sweep and aggregated; unwrapping across sweep seams would not fit a line.
    spin_rate = float(torch.tensor([
        fit_spin_rate(azimuth, time)
        for azimuth, time in azimuth_time if len(azimuth) >= min_points_for_spin_fit
    ]).median())

    return compute_azimuth_extent(azimuth_time, spin_rate)


def build_range_sweeps(
        raw_sweeps: list[Sweep],
        motion: LinearMotion,
        model: LidarModel,
        min_points_for_spin_fit: int = 100,
) -> tuple[list[Sweep], LidarCamera]:
    """Rasterize a lidar's raw sweeps into range images and build their cameras.

    The azimuth seam and extent are fit once from all the sweeps, so every sweep
    of a lidar shares a column-to-azimuth map and the cameras stack. Per-point
    times must be in the same timeline as ``motion.pose_time`` (scene-relative),
    so that placing points at their capture time via ``motion.pose_at`` and the
    fit rolling shutter agree.

    Args:
        raw_sweeps: The lidar's unprojected sweeps, in capture order.
        motion: The lidar's per-sweep ego motion, batched over sweeps.
        model: The sensor model supplying beam order, elevations and resolution.
        min_points_for_spin_fit: A sweep must have at least this many valid
            returns to be used in the spin-rate (shutter duration) fit. Sweeps
            with fewer points fit a noisy line and are skipped. Lower it for
            sparse or partial-arc sensors whose sweeps carry few returns.

    Returns:
        The rasterized range sweeps and a LidarCamera batched over them.
    """
    azimuth_offset, azimuth_range = fit_azimuth_extent(
        raw_sweeps, min_points_for_spin_fit)

    range_sweeps = []
    cameras = []
    for sweep_idx, raw_sweep in enumerate(raw_sweeps):
        range_sweep = rasterize_range_sweep(
            raw_sweep, motion[sweep_idx], model, azimuth_offset, azimuth_range
        )
        cameras.append(construct_lidar_camera(
            range_sweep, motion[sweep_idx], model, azimuth_offset, azimuth_range
        ))
        range_sweeps.append(range_sweep)

    return range_sweeps, LidarCamera.stack(cameras, dim=0)


def range_sweep_from_raw_sweep(
        raw_sweep: Sweep,
        motion: LinearMotion,
        rng_image: torch.Tensor,
        mask: torch.Tensor,
        az_idx: torch.Tensor,
        beam_idx: torch.Tensor
):
    az_masked = az_idx[mask]
    beam_masked = beam_idx[mask]

    int_image = torch.zeros_like(rng_image, dtype=torch.uint8)
    int_image[beam_masked, az_masked] = raw_sweep.intensity[mask]

    time_image = torch.zeros_like(rng_image)
    time_image[beam_masked, az_masked] = raw_sweep.time[mask]

    mask_image = torch.zeros_like(rng_image, dtype=torch.bool)
    mask_image[beam_masked, az_masked] = True

    xyz_image = torch.zeros(rng_image.shape[0], rng_image.shape[1], 3)
    origin_image = torch.zeros_like(xyz_image)

    point_poses = motion.float().pose_at(
        raw_sweep.time[mask].float().unsqueeze(-1)
    )
    xyz_image[beam_masked, az_masked] = point_poses @ raw_sweep.xyz[mask].float()
    origin_image[beam_masked, az_masked] = point_poses.t

    return Sweep(
        xyz=xyz_image,
        origin=origin_image,
        intensity=int_image.unsqueeze(-1),
        time=time_image.unsqueeze(-1),
        mask=mask_image
    )


def rasterize_range_sweep(
        raw_sweep: Sweep,
        motion: LinearMotion,
        model: LidarModel,
        azimuth_offset: float,
        azimuth_range: float = DEGREES_PER_TURN,
) -> Sweep:
    """Rasterize a raw spinning-lidar sweep into a range image.

    Rows are beams ordered top to bottom (via ``model.beam_remap``), columns are
    azimuth bins measured from ``azimuth_offset`` -- the azimuth the sweep is cut
    at, so its time discontinuity lands on the image edge and the shutter stays
    linear in image space (see ``compute_azimuth_extent``). The image spans
    ``azimuth_range``, so its width matches the camera limits and column x maps
    back to a consistent azimuth; a partial-arc lidar is cropped rather than
    padded to a full turn. Points colliding in a cell are resolved to the
    nearest return, as in ``rasterize_into_depth_mask``.

    Args:
        raw_sweep: A single unprojected sweep in lidar space.
        motion: The lidar's ego motion, used to place points in the reference
            frame at their capture time.
        model: The sensor model supplying beam order and azimuth resolution.
        azimuth_offset: Azimuth of column zero, in degrees.
        azimuth_range: Azimuth the image spans, in degrees (as from
            ``compute_azimuth_extent``); the full turn for a 360 deg lidar.

    Returns:
        The sweep rasterized into ``[num_beams, width]`` images, with
        ``width = round(azimuth_range / azimuth_resolution)``.
    """
    azimuth = sweep_azimuth(raw_sweep.xyz.float())
    distance = raw_sweep.xyz.float().norm(dim=-1)
    width = round(azimuth_range / model.azimuth_resolution)

    row = model.beam_remap[raw_sweep.beam_idx.long()]
    # Match the renderer's column-to-azimuth map. The CUDA lidar camera reads
    # column i at azimuth_offset + (i + 0.5) * azimuth_range / width (its
    # pixel_to_uv is u = (i + 0.5) / width, pixel centres on half-integers), so a
    # point at azimuth a belongs to bin floor(azimuth_fraction * width). Rounding
    # instead would centre bins on integers and shift every point half a column
    # off where it is sampled.
    azimuth_fraction = ((azimuth - azimuth_offset) % DEGREES_PER_TURN) / azimuth_range
    column = (azimuth_fraction * width).floor().long()

    rng_image, mask = rasterize_into_depth_mask(
        column, row, distance,
        width=width,
        height=model.num_beams,
    )

    return range_sweep_from_raw_sweep(
        raw_sweep=raw_sweep,
        motion=motion,
        rng_image=rng_image,
        mask=mask,
        az_idx=column,
        beam_idx=row,
    )


def azimuth_rotation(
        azimuth_offset: float,
        dtype: torch.dtype = torch.double
) -> SE3:
    """The lidar-space remount that moves azimuth zero to ``azimuth_offset``.

    Its sign is negative because xsimgs' azimuth (``atan2(-x, -y) + pi``) runs
    clockwise, opposite to a positive rotation about +Z.

    This is what separates a lidar camera built by ``construct_lidar_camera`` in
    'rotate_pose' mode from the sensor it belongs to, so it is also what carries
    a point from that camera's frame back into the lidar's own.
    """
    return SE3(
        Quat.from_axis_angle(
            torch.tensor([0.0, 0.0, -azimuth_offset], dtype=dtype).deg2rad()),
        torch.zeros(3, dtype=dtype),
    )


def rotate_motion_azimuth(
        motion: LinearMotion,
        azimuth_offset: float
) -> LinearMotion:
    """Spin a lidar's pose about its own Z axis so azimuth zero moves to ``azimuth_offset``.

    This is the pose-side equivalent of a non-zero ``phi_min``: a camera built
    on the returned motion with ``phi_min = 0`` casts the same world rays as one
    built on ``motion`` with ``phi_min = azimuth_offset``.

    The rotation (``azimuth_rotation``, which carries the sign convention) is
    applied on the right, in lidar space, so it is a fixed remount of the sensor
    rather than a change of the world frame.

    Both velocities are left untouched, and stay exact rather than approximate.
    They live in the world frame, where ``pose_at`` integrates them on the left
    (``q(t) = exp(w dt) q0``, ``t(t) = t0 + v dt``), so a constant right-hand
    factor passes straight through: ``pose_at(t) @ R == rotated.pose_at(t)``.
    (This holds because the factor is a pure rotation. A remount with a non-zero
    lever arm would move the origin the linear velocity describes, and would
    need correcting.)

    Args:
        motion: The lidar's ego motion, ``world_se3_lidar``.
        azimuth_offset: Azimuth to place at column zero, in degrees.

    Returns:
        The motion with its pose rotated; velocities shared with the input.
    """
    lidar_se3_rotated = azimuth_rotation(azimuth_offset, motion.pose.dtype)

    return LinearMotion(
        pose=(motion.pose @ lidar_se3_rotated).std(),
        pose_time=motion.pose_time,
        linear_velocity=motion.linear_velocity,
        angular_velocity=motion.angular_velocity,
    )


def construct_lidar_camera(
        range_sweep: Sweep,
        motion: LinearMotion,
        model: LidarModel,
        azimuth_offset: float,
        azimuth_range: float,
        inlier_time_threshold: float = 0.01,
        offset_mode: Literal['rotate_pose', 'az_limits'] = 'rotate_pose'
) -> LidarCamera:
    """Build a LidarCamera whose rolling shutter matches a range image's times.

    The shutter is fit from the range image's own per-pixel times, so rendering
    reproduces when each column was captured. Points whose time is far from that
    linear fit -- the few the beam azimuth offsets strand on the wrong side of
    the seam -- are held out of the fit so they cannot skew it.

    Args:
        range_sweep: A rasterized sweep, as from ``rasterize_range_sweep``.
        motion: The lidar's ego motion.
        model: The sensor model supplying beam elevations.
        azimuth_offset: Azimuth of column zero, in degrees (the camera phi_min).
        azimuth_range: Azimuth the image spans, in degrees (the camera phi_range).
        inlier_time_threshold: Seconds; pixels whose time deviates from the
            linear shutter by more than this are excluded from the fit.
        offset_mode: How the azimuth offset reaches the camera. 'az_limits'
            passes it as ``phi_min``; 'rotate_pose' instead spins the pose about
            the lidar's Z axis and leaves ``phi_min`` at zero (see
            ``rotate_motion_azimuth``). The two are equivalent in the camera
            model, so prefer 'rotate_pose' while renderers are unproven on a
            non-zero ``phi_min``.

    Returns:
        A LidarCamera carrying the fit shutter, beam elevations and azimuth
        limits, sized to the range image.
    """
    assert offset_mode in ['rotate_pose', 'az_limits']
    time_image = range_sweep.time
    coarse_shutter = RollingShutter.from_time_image(
        time_image, mask=range_sweep.mask, horizontal_only=True
    )
    predicted_time = coarse_shutter.image_time(time_image)
    inlier = range_sweep.mask & (
        (predicted_time - time_image).abs().squeeze(-1) < inlier_time_threshold
    )
    shutter = RollingShutter.from_time_image(
        time_image, mask=inlier, horizontal_only=True
    )
    az_offset = torch.deg2rad(torch.tensor(azimuth_offset))
    az_range = torch.deg2rad(torch.tensor(azimuth_range))
    if offset_mode == 'rotate_pose':
        motion = rotate_motion_azimuth(motion, azimuth_offset)
        az_offset = torch.zeros_like(az_offset)

    return LidarCamera.create(
        world_se3_camera=motion,
        shutter=shutter,
        beam_angles=torch.deg2rad(model.beam_elevations),
        phi_min=az_offset,
        phi_range=az_range,
    ).float()
