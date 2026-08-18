from typing import Literal

import torch
from xsimgs.cameras import SphericalCameraTorch
from xsimgs.structures import OrientedBoxes


from ..loaders.sensor_data import SensorData, SensorType


TimeCorrectionMethod = (
    Literal['points_time'] | Literal['box_center']
    | Literal['box_center_unwrap'] | Literal['disable']
)

# Extra sweep-phase margin added to the box angular extent when deciding
# whether a box may straddle the azimuth wrap-around boundary
BOUNDARY_PHASE_MARGIN = 0.02

# Cost of moving a box time by a whole sweep period during unwrapping.
# Strictly a tie-breaker: must stay orders of magnitude below real phase
# jumps (~1.0), otherwise long shifted runs accumulate more penalty than
# the single jump they avoid and the track breaks mid-way
SHIFT_PENALTY = 1e-4


def reduce_stats(
        num_boxes, isect_mask, mask, cur_sweep, stat: str
):
    result = torch.zeros(num_boxes)
    result.scatter_reduce_(
        dim=0, index=isect_mask[mask].long(),
        src=cur_sweep.time[mask].view(-1).float(),
        reduce=stat,
        include_self=False
    )
    return result


def update_boxes_time_by_points(
        sensors: dict[SensorType, SensorData],
        boxes: OrientedBoxes,
):
    assert boxes.sensor_link is not None, "Boxes must be associated with sensors"
    assert (boxes.sensor_link.sensor_type == SensorType.LIDAR.value).all().item(),\
        "Only LiDAR associated boxes are supported now"

    sensor_data = sensors[SensorType.LIDAR]
    sensor_loader = sensor_data.get_loader()

    for group_indices, box_group in boxes.sample_group_iterator():
        sensor_id = int(box_group.sensor_link.sensor_id[0])
        sample_idx = int(box_group.sensor_link.sample_idx[0])
        cur_sweep = sensor_loader(sensor_id=sensor_id, sample_idx=sample_idx)
        tmin, tmax = cur_sweep.time.amin().item(), cur_sweep.time.amax().item()
        isect_mask = box_group.intersection_map(cur_sweep.xyz)
        mask = isect_mask != -1

        box_time_min = reduce_stats(
            len(box_group), isect_mask, mask, cur_sweep, stat='min')
        box_time_max = reduce_stats(
            len(box_group), isect_mask, mask, cur_sweep, stat='max')
        box_time_mean = reduce_stats(
            len(box_group), isect_mask, mask, cur_sweep, stat='mean')
        # If box is seen across the azimuth boundary we have no reliable method to estimate
        # which boundary was used during annotation. In such cases leave timestamp as is
        valid_update_mask = (box_time_min != 0)
        valid_update_mask &= (box_time_max - box_time_min) < (0.5 * (tmax - tmin))
        box_time_new = torch.where(valid_update_mask, box_time_mean, box_group.timestamps)

        boxes.motion.pose_time[group_indices] = box_time_new.view(-1, 1)


def update_boxes_time_by_centers(
        sensors: dict[SensorType, SensorData],
        boxes: OrientedBoxes,
        unwrap: bool = False,
):
    # estimate boxes timestamps by projecting its 3D center onto associated camera
    # uses projection time as estimation
    # if unwrap=True, objects seen near azimuth discontinuity boundary on
    # spherical rolling shutter cameras, get special treatment - their phase is recovered
    # via dynamic programming to minimize phase jumps

    num_boxes = len(boxes)
    device = boxes.motion.pose_time.device
    proj_time = torch.zeros(num_boxes, device=device)
    period = torch.zeros(num_boxes, device=device)
    sweep_start = torch.zeros(num_boxes, device=device)
    phase = torch.full((num_boxes,), 0.5, device=device)
    alt = torch.zeros(num_boxes, device=device)
    valid = torch.zeros(num_boxes, dtype=torch.bool, device=device)
    sensor_time = torch.zeros(num_boxes, device=device)

    for cur_mask, cur_boxes in boxes.sample_group_iterator():
        sensor_type = SensorType(int(cur_boxes.sensor_link.sensor_type[0]))
        sensor_id = int(cur_boxes.sensor_link.sensor_id[0])
        sample_idx = int(cur_boxes.sensor_link.sample_idx[0])
        cur_camera = sensors[sensor_type].get_camera(
            sensor_id=sensor_id, sample_idx=sample_idx
        )
        # Common time axis for ordering track observations across sensors
        sensor_time[cur_mask] = float(cur_camera.world_se3_camera.pose_time.item())

        uvdt, proj_mask = cur_camera.world_to_uv(cur_boxes.centers)
        update_indices = cur_mask[proj_mask]

        proj_time[update_indices] = uvdt[proj_mask, 3]
        valid[update_indices] = True

        tmin, tmax = [t.item() for t in cur_camera.shutter.time_range]
        cur_period = float(tmax - tmin)
        if cur_period <= 0:
            # Global shutter camera, no special treatment required
            continue
        period[cur_mask] = cur_period
        sweep_start[cur_mask] = float(tmin)

        # Capture time normalized by rolling shutter capture interval; valid for any RS camera
        cur_phase = (uvdt[..., 3] - tmin) / cur_period
        phase[update_indices] = cur_phase[proj_mask]

        if not isinstance(cur_camera, SphericalCameraTorch):
            # Wrap-around ambiguity only exists for spherical cameras
            continue
        phi_range = float(cur_camera.phi_range)
        wrap = 2 * torch.pi / phi_range

        # The box is ambiguous if its azimuth extent wraps, and can be seen from the
        # opposite side of spherical image
        # az_angular_size = object azimuth extent normalized by azimuth range + eps
        bev_radius = 0.5 * cur_boxes.sizes[..., :2].norm(dim=-1)
        az_angular_size = torch.atan2(bev_radius, uvdt[..., 2]) / phi_range
        az_angular_size = az_angular_size + BOUNDARY_PHASE_MARGIN

        cur_alt = torch.where(
            cur_phase < 0.5,
            torch.where(cur_phase <= 1.0 - wrap + az_angular_size, wrap, 0.0),
            torch.where(cur_phase >= wrap - az_angular_size, -wrap, 0.0),
        )
        alt[update_indices] = cur_alt[proj_mask]

    shift = unwrap_shifts(boxes, phase, alt, valid, sensor_time) if unwrap \
        else torch.zeros_like(proj_time)
    new_time = proj_time + shift * period

    if unwrap:
        # Detected before filling: a filled box carries no measurement of its
        # own, so its alt stays 0 and would read as an anchor the track has not
        # got.
        unplaceable = anchorless_tracks(boxes, alt, valid)

        # Boxes whose projection failed (non-converged solve at the azimuth
        # boundary, center outside the beam fan) would keep the raw frame
        # timestamp, inconsistent with corrected neighbors
        fill_time, filled = fill_missing(
            boxes, phase + shift, valid, sweep_start, period, sensor_time)
        new_time = torch.where(filled, fill_time, new_time)
        valid = valid | filled

        # A track that never reaches an unambiguous frame cannot be placed by
        # unwrapping: both branches are equally smooth, so whichever the
        # tie-break picks is a guess worth a whole revolution, applied to the
        # whole track. The sensor's own reference time is the honest answer --
        # wrong by at most half a period, and not pretending otherwise.
        for track in unplaceable:
            new_time[track] = sensor_time[track]
            valid[track] = True

    boxes.motion.pose_time[valid] = new_time[valid].view(-1, 1)


def fill_missing(
        boxes: OrientedBoxes,
        psi: torch.Tensor,
        valid: torch.Tensor,
        sweep_start: torch.Tensor,
        period: torch.Tensor,
        sensor_time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Interpolate the unwrapped phase of boxes without a valid center
    projection from valid boxes of the same track, and convert it back to
    a timestamp within their own sweep. Returns (times, filled_mask).
    """
    time = torch.zeros_like(psi)
    filled = torch.zeros_like(valid)

    for obj_id in boxes.object_ids.unique().tolist():
        track = (boxes.object_ids == obj_id).nonzero().view(-1)
        missing = track[~valid[track] & (period[track] > 0)]
        anchors = track[valid[track]]
        if len(missing) == 0 or len(anchors) == 0:
            continue
        anchors = anchors[sensor_time[anchors].argsort()]
        anchor_t, anchor_psi = sensor_time[anchors], psi[anchors]
        for i in missing.tolist():
            j = int(torch.searchsorted(anchor_t, sensor_time[i]))
            if j == 0:
                cur_psi = anchor_psi[0]
            elif j == len(anchors):
                cur_psi = anchor_psi[-1]
            else:
                w = ((sensor_time[i] - anchor_t[j - 1])
                     / (anchor_t[j] - anchor_t[j - 1]).clamp(min=1e-6))
                cur_psi = torch.lerp(anchor_psi[j - 1], anchor_psi[j], w)
            time[i] = sweep_start[i] + cur_psi * period[i]
            filled[i] = True
    return time, filled


def anchorless_tracks(
        boxes: OrientedBoxes,
        alt: torch.Tensor,
        valid: torch.Tensor
) -> list[torch.Tensor]:
    """Tracks every frame of which sits at the sweep boundary.

    Unwrapping makes such a track self-consistent but cannot place it: anchored
    all-early and all-late are equally smooth, so the branch comes from a
    tie-break rather than from evidence, and whichever it picks is applied to
    the whole track. Correcting these is worse than not correcting them, so
    they are detected here and left to the sensor reference time.

    Membership is decided on the track's valid boxes, but every box of the
    track is returned: once the track is known to be unplaceable, its filled-in
    boxes carry the same guess and must be treated the same way.

    Returns:
        Box indices of each anchorless track, one tensor per track.
    """
    tracks = []
    for obj_id in boxes.object_ids.unique().tolist():
        in_track = (boxes.object_ids == obj_id).view(-1)
        measured = (in_track & valid).nonzero().view(-1)
        if len(measured) >= 2 and not (alt[measured] == 0).any():
            tracks.append(in_track.nonzero().view(-1))

    return tracks


def unwrap_shifts(
        boxes: OrientedBoxes,
        phase: torch.Tensor,
        alt: torch.Tensor,
        valid: torch.Tensor,
        sensor_time: torch.Tensor
) -> torch.Tensor:
    """
    Resolve the sweep wrap-around ambiguity per track. A box whose azimuth
    extent reaches the rescanned direction may anchor to either capture;
    `alt` holds the signed phase offset of the alternate capture (one full
    revolution away, 0 for unambiguous boxes). Track observations are
    ordered by their sensor reference time, so boxes from sensors with
    different capture intervals combine correctly. Returns per-box phase
    shift making the phase sequence along each track smooth.
    """
    assert boxes.object_ids is not None, \
        "Boxes must have object_ids for phase unwrapping"
    shifts = torch.zeros(len(boxes), device=phase.device)

    for obj_id in boxes.object_ids.unique().tolist():
        track = ((boxes.object_ids == obj_id) & valid).nonzero().view(-1)
        if len(track) < 2 or not (alt[track] != 0).any():
            continue
        track = track[sensor_time[track].argsort()]
        track_shifts = unwrap_track(
            phase[track].tolist(),
            alt[track].tolist(),
            sensor_time[track].tolist()
        )
        shifts[track] = torch.as_tensor(
            track_shifts, dtype=shifts.dtype, device=shifts.device)
    return shifts


def unwrap_track(
        phases: list[float],
        alts: list[float],
        times: list[float]
) -> list[float]:
    """
    Viterbi over per-box time branches: an ambiguous box may keep its
    projected phase or take its alternate capture one revolution away.
    Picks the branch combination minimizing total phase variation along
    the track; ties resolve to the raw projection.
    """
    candidates = [
        [p, p + a] if a != 0 else [p]
        for p, a in zip(phases, alts)
    ]

    costs = [SHIFT_PENALTY * abs(c - phases[0]) for c in candidates[0]]
    backptr = []
    for i in range(1, len(candidates)):
        # Near-simultaneous observations (e.g. two lidars in the same
        # frame) must agree on the branch: tiny gap => huge variation cost
        gap = max(times[i] - times[i - 1], 1e-3)
        step_costs, step_ptrs = [], []
        for cand in candidates[i]:
            penalty = SHIFT_PENALTY * abs(cand - phases[i])
            trans = [
                cost + abs(cand - prev) / gap + penalty
                for cost, prev in zip(costs, candidates[i - 1])
            ]
            best = trans.index(min(trans))
            step_costs.append(trans[best])
            step_ptrs.append(best)
        costs = step_costs
        backptr.append(step_ptrs)

    best = costs.index(min(costs))
    shifts = [0.0] * len(candidates)
    for i in range(len(candidates) - 1, -1, -1):
        shifts[i] = candidates[i][best] - phases[i]
        if i > 0:
            best = backptr[i - 1][best]
    return shifts


def update_boxes_time(
        sensors: dict[SensorType, SensorData],
        boxes: OrientedBoxes,
        method: TimeCorrectionMethod = 'points_time'
):
    assert method in ['points_time', 'box_center', 'box_center_unwrap', 'disable']
    if method == 'points_time':
        update_boxes_time_by_points(sensors, boxes)
    if method == 'box_center':
        update_boxes_time_by_centers(sensors, boxes)
    if method == 'box_center_unwrap':
        update_boxes_time_by_centers(sensors, boxes, unwrap=True)

