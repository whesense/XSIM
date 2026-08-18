from typing import NamedTuple
import torch

from xsimgs.cameras import RSCamera
from xsimgs.structures import OrientedBoxes
from xsim.structures.time_range import TimeRange

from .dataset_config import DynamicFilterOptions

CameraTimeFunction = NamedTuple('CameraTimeFunction', [('freq', float), ('t0', float)])


def estimate_camera_time_fn(
        cameras: RSCamera,
):
    tmin, tmax = cameras.shutter.time_range
    mtx = torch.stack([
        torch.arange(len(tmin), device=cameras.device),
        torch.ones(len(tmin), device=cameras.device)
    ], dim=1)
    freq, t_offset = torch.linalg.lstsq(mtx, tmin.view(-1, 1)).solution.view(-1).tolist()
    return freq, t_offset, tmin.amin().item(), tmax.amax().item()


def fit_camera_time(
        cameras: dict[int, RSCamera],
        lidar: dict[int, RSCamera],
        scene_time_extension: float = 0.2
) -> tuple[dict[int, CameraTimeFunction], dict[int, CameraTimeFunction], TimeRange]:
    cam_times: dict[int, CameraTimeFunction] = {}
    lidar_times: dict[int, CameraTimeFunction] = {}

    min_t = 0.0
    max_t = 0.0

    for camera_id in cameras:
        freq, t_offset, cur_min_t, cur_max_t = estimate_camera_time_fn(cameras[camera_id])
        min_t = min(min_t, cur_min_t)
        max_t = max(max_t, cur_max_t)
        cam_times[camera_id] = CameraTimeFunction(freq, t_offset)

    for lidar_id in lidar:
        freq, t_offset, cur_min_t, cur_max_t = estimate_camera_time_fn(lidar[lidar_id])
        min_t = min(min_t, cur_min_t)
        max_t = max(max_t, cur_max_t)
        lidar_times[lidar_id] = CameraTimeFunction(freq, t_offset)

    time_bounds = TimeRange(
    	min_t - scene_time_extension, max_t + scene_time_extension
    )

    return cam_times, lidar_times, time_bounds


def instance_matrix(
        boxes: list[OrientedBoxes],
        device: torch.device | str = None,
) -> tuple[OrientedBoxes, torch.Tensor]:
    if len(boxes) == 0:
        # A fully static scene has no objects; return empty, well-formed
        # containers so everything downstream simply iterates over nothing.
        device = device or 'cpu'
        instances = OrientedBoxes.zeros(size=[0, 0], device=device)
        instances.mask = torch.zeros(0, 0, dtype=torch.bool, device=device)
        return instances, torch.zeros(0, 3, device=device)

    device = boxes[0].device
    for b in boxes:
        b.mask = torch.ones(len(b), dtype=torch.bool, device=device)
    instance_sizes = torch.stack([
        boxes[i].sizes.amax(dim=0)
        for i in range(len(boxes))
    ], dim=0)
    max_num_keyframes = max([len(b) for b in boxes])

    instances = OrientedBoxes.zeros(size=[len(boxes), max_num_keyframes], device=device)
    for inst_id, inst_boxes in enumerate(boxes):
        instances[inst_id, :len(inst_boxes)] = inst_boxes

    return instances, instance_sizes


def dynamic_instances_filter(
        instances: OrientedBoxes,
        cfg: DynamicFilterOptions
) -> list[int]:
    result = []
    for obj_id in range(len(instances)):
        cur_boxes = instances[obj_id]
        cur_boxes = cur_boxes[cur_boxes.mask]
        centers = cur_boxes.centers
        sum_traveled = float(torch.norm(centers[1:] - centers[:-1], dim=-1).sum())
        total_travelled = (centers[-1] - centers[0]).norm().item()
        ratio = total_travelled / (sum_traveled + 1e-7)
        if (cfg.min_ratio is not None) and (ratio < cfg.min_ratio):
            continue

        if (sum_traveled > cfg.min_sum_dist) and (total_travelled > cfg.min_total_dist):
            result.append(obj_id)

    return result

