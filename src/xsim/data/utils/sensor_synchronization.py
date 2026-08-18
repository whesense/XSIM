import torch
from torch.nn.functional import normalize

from ..loaders.sensor_data import SensorData


def sensor_sample_times(sensor: SensorData, sensor_id: int) -> torch.Tensor:
    """Capture time of every sample of a sensor, in scene-relative seconds.

    Reads ``pose_time``, which is the middle of the sample's shutter for both
    cameras and lidars, so times of sensors with very different capture
    durations stay comparable.
    """
    return sensor.get_cameras(sensor_id).world_se3_camera.pose_time.view(-1)


def closest_sample_indices(
        query_times: torch.Tensor,
        reference_times: torch.Tensor
) -> list[int]:
    """For each query time, the index of the nearest of ``reference_times``."""
    return (
        query_times.view(-1, 1) - reference_times.view(1, -1)
    ).abs().argmin(dim=-1).tolist()


def transpose_sensor_matches(
        sensor_ids: list[int],
        query_times: dict[int, torch.Tensor],
        reference_times: dict[int, torch.Tensor]
) -> list[list[tuple[int, int]]]:
    """For each query, the nearest sample of every sensor in ``sensor_ids``.

    Both sides are keyed per sensor, because a lidar reaches each camera's
    direction at its own instant, so the timeline a match runs against differs
    per pair. Matching runs once per sensor over all queries, then the result
    is transposed into the per-query lists the synchronization API returns.
    """
    matches = {
        sensor_id: closest_sample_indices(
            query_times[sensor_id], reference_times[sensor_id]
        )
        for sensor_id in sensor_ids
    }
    num_queries = len(query_times[sensor_ids[0]]) if sensor_ids else 0

    return [
        [(sensor_id, matches[sensor_id][sample_idx]) for sensor_id in sensor_ids]
        for sample_idx in range(num_queries)
    ]


def wrap_azimuth_fraction(u: torch.Tensor, phi_range: torch.Tensor) -> torch.Tensor:
    """Wrap a spherical camera's horizontal uv into its arc.

    ``camera_to_uv`` divides a ``[0, 2pi)`` azimuth by the arc without treating
    it as periodic, so a direction past the seam lands outside ``[0, 1]`` even
    when the arc covers it; shifting by whole turns puts it back. What remains
    outside is a bearing a partial-arc lidar genuinely never scans, and is
    clamped to whichever end of the arc it is nearer -- the closest that sensor
    ever looks that way.
    """
    turns = 2.0 * torch.pi / phi_range
    u = u % turns
    nearer_end = (u - 1.0) < (turns - u)

    return torch.where(
        u <= 1.0, u, torch.where(nearer_end, torch.ones_like(u), torch.zeros_like(u))
    )


def camera_principal_directions(cameras: SensorData) -> dict[int, torch.Tensor]:
    """Unit principal ray of every camera, in ego coordinates.

    Taken through ``uv_to_camera`` at the image centre rather than assumed to be
    an axis, so each camera's own intrinsics decide where it looks.
    """
    directions = {}
    for camera_id in cameras.indices:
        camera = cameras.get_camera(camera_id, 0)
        forward = camera.uv_to_camera(
            torch.tensor([[0.5, 0.5]], dtype=camera.dtype),
            torch.ones(1, dtype=camera.dtype)
        )[0]
        ego_se3_camera = cameras.get_ego_se3_sensor(camera_id)
        directions[camera_id] = normalize(
            ego_se3_camera.q.to(forward.dtype) @ forward, dim=-1
        )

    return directions


def principal_ray_scan_times(
        cameras: SensorData,
        lidar: SensorData
) -> dict[int, dict[int, torch.Tensor]]:
    """When each sweep of each lidar scans each camera's principal ray.

    The camera's viewing direction is rotated into lidar coordinates through the
    rig calibration, turned into a column by the lidar camera model, and read
    back as a time through that sweep's shutter.

    Returns:
        ``[lidar_id][camera_id]`` to a tensor of absolute times, one per sweep
        of that lidar.
    """
    directions = camera_principal_directions(cameras)
    scan_times = {}

    for lidar_id in lidar.indices:
        lidar_cameras = lidar.get_cameras(lidar_id)
        ego_se3_lidar = lidar.get_ego_se3_sensor(lidar_id)
        scan_times[lidar_id] = {}

        for camera_id, ego_direction in directions.items():
            direction = ego_se3_lidar.q.inv().to(
                ego_direction.dtype
            ) @ ego_direction
            direction = direction.to(
                dtype=lidar_cameras.dtype, device=lidar_cameras.device
            ).view(1, 3).expand(len(lidar_cameras), 3)

            uv = lidar_cameras.camera_to_uv(direction)
            u = wrap_azimuth_fraction(uv[..., 0], lidar_cameras.phi_range)
            uv = torch.stack([u, uv[..., 1]], dim=-1).unsqueeze(-2)
            scan_times[lidar_id][camera_id] = (
                lidar_cameras.shutter.uv_time(uv).view(-1)
            )

    return scan_times


def lidar_synchronized_groups(
        lidar: SensorData,
        tolerance: float
) -> list[tuple[int, ...]]:
    """Lidars that fire together, which must be consumed as one sample.

    Two lidars belong to the same group when they hold the same number of
    samples and no pair of those samples is further apart than ``tolerance``
    seconds.
    """
    times = {
        lidar_id: sensor_sample_times(lidar, lidar_id)
        for lidar_id in lidar.indices
    }
    groups = []
    for lidar_id, lidar_times in times.items():
        for group in groups:
            reference = times[group[0]]
            if len(reference) == len(lidar_times) and float(
                (reference - lidar_times).abs().max()
            ) <= tolerance:
                group.append(lidar_id)
                break
        else:
            groups.append([lidar_id])

    return [tuple(group) for group in groups]


def camera_lidar_synchronization(
        cameras: SensorData,
        lidar: SensorData,
        lidar_sync_tolerance: float = 0.005
):
    """Pair camera and lidar samples by when each observed the same bearing.

    Matching is on time, but not on the sweep's timestamp: a sweep meets a
    camera at the instant its spin crosses that camera's principal ray (see
    ``principal_ray_scan_times``).

    Args:
        lidar_sync_tolerance: Seconds two lidars' samples may differ by and
            still count as firing together.

    Returns:
        The ``(lidar_closest_cameras, cameras_closest_lidar,
        lidar_synchronized_groups)`` triple the ``DatasetScene`` API expects.
    """
    camera_times = {
        camera_id: sensor_sample_times(cameras, camera_id)
        for camera_id in cameras.indices
    }
    scan_times = principal_ray_scan_times(cameras, lidar)

    lidar_closest_cameras = {
        lidar_id: transpose_sensor_matches(
            cameras.indices,
            query_times=scan_times[lidar_id],
            reference_times=camera_times
        )
        for lidar_id in lidar.indices
    }
    cameras_closest_lidar = {
        camera_id: transpose_sensor_matches(
            lidar.indices,
            query_times={
                lidar_id: camera_times[camera_id] for lidar_id in lidar.indices
            },
            reference_times={
                lidar_id: scan_times[lidar_id][camera_id]
                for lidar_id in lidar.indices
            }
        )
        for camera_id in cameras.indices
    }

    return (
        lidar_closest_cameras,
        cameras_closest_lidar,
        lidar_synchronized_groups(lidar, lidar_sync_tolerance)
    )
