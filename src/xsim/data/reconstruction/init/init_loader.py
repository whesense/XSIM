import torch
from toast import LinearMotion, SE3, sample_trajectory, use_backend

from xsim.utils import track

from xsimgs.structures import OrientedBoxes

from xsim.data.reconstruction import SceneReconstructionDataset
from xsim.data.loaders.sensor_iterator import sensor_indices


# Seconds a track may be carried past its keyframes before the object counts as
# gone. Matches TrajectoryNode, so init and training agree on which objects exist.
MAX_EXTRAPOLATION = 0.5



def init_loader_indices(sim_ds: SceneReconstructionDataset):
    (
        lidar_closest_cameras,
        cameras_closest_lidar,
        lidar_synchronized_groups
    ) = sim_ds.scene.camera_lidar_synchronization()

    indices = []

    for lidar_ids in lidar_synchronized_groups:
        lidar_indices = list(zip(*[
            sensor_indices(sim_ds.scene.lidar, sensor_ids=[lidar_id, ])
            for lidar_id in lidar_ids
        ]))

        camera_indices = [
            lidar_closest_cameras[lidar_ids[0]][i]
            for i in range(sim_ds.scene.lidar.num_images(lidar_ids[0]))
        ]

        cur_indices = [
            {'lidar': lidar_ids, 'image': camera_ids}
            for lidar_ids, camera_ids in zip(lidar_indices, camera_indices)
        ]
        indices += cur_indices

    return indices


def sample_instance_boxes(
        sim_ds: SceneReconstructionDataset,
        query_time: torch.Tensor,
        max_extrapolation: float = MAX_EXTRAPOLATION
) -> OrientedBoxes:
    instances: OrientedBoxes = sim_ds.instances
    keyframe_time = instances.motion.pose_time.squeeze(-1)
    query_time = query_time.to(keyframe_time.device).reshape(-1, 1).expand(
        len(instances), 1
    )

    # Only the torch backend implements sample_trajectory outside CUDA.
    with use_backend('torch'):
        out_q, out_t, out_mask, out_v, out_w, out_indices = sample_trajectory(
            query_times=query_time,
            seq_times=keyframe_time,
            seq_quats=instances.motion.pose.q.std().data,
            seq_t=instances.motion.pose.t,
            seq_mask=instances.mask,
            use_only_valid_keyframes=True,
            extrapolate=True,
            extrapolation_window=1,
            mask_mode='and',
            return_velocities=True,
            return_mask=True,
            return_indices=True
        )

    # Extrapolation itself is unbounded, so how far the query sits outside the
    # bracketing keyframes is what decides whether the object is still there.
    bracket_start = keyframe_time.take_along_dim(
        out_indices[:, 0, 0].long().view(-1, 1), dim=1
    ).view(-1)
    bracket_end = keyframe_time.take_along_dim(
        out_indices[:, 0, 1].long().view(-1, 1), dim=1
    ).view(-1)
    query = query_time[:, 0]
    too_far = (
        (query < bracket_start - max_extrapolation) |
        (bracket_end + max_extrapolation < query)
    )

    return OrientedBoxes(
        motion=LinearMotion(
            pose=SE3(out_q[:, 0], out_t[:, 0]),
            pose_time=query_time[:, :1],
            linear_velocity=out_v[:, 0],
            angular_velocity=out_w[:, 0],
        ),
        sizes=sim_ds.instance_size.to(out_t.device),
        object_ids=instances.object_ids[:, 0],
        mask=out_mask[:, 0] & ~too_far,
    )


def observed_instance_times(
        sim_ds: SceneReconstructionDataset,
        batch,
        boxes: OrientedBoxes
) -> torch.Tensor:
    """When each instance was actually swept, not when its sweep began.

    A spinning lidar reaches an object at one instant of its revolution, so
    objects of one sweep are seen up to a full period apart. Projecting a box
    centre reads back the time of the column it lands in; passing the box's
    velocity lets the projection solve for where a moving object is at that
    instant instead of assuming it stood still. Times are averaged over the
    lidars that see the object; instances no lidar sees keep their own
    ``pose_time``.

    Returns:
        One time per instance as ``[num_instances, 1]``, in the row order of
        ``boxes``.
    """
    centers = boxes.centers
    velocity = boxes.motion.linear_velocity
    total = torch.zeros(len(boxes), device=centers.device, dtype=centers.dtype)
    count = torch.zeros_like(total)

    for sensor_id, sample_idx in [x[0] for x in batch['lidar']]:
        camera = sim_ds.lidar_cameras[sensor_id][sample_idx]
        uvdt, seen = camera.world_to_uv(
            centers.to(camera.device),
            points_velocity=velocity.to(camera.device)
        )
        seen = seen.to(total.device) & boxes.mask
        total[seen] += uvdt[..., 3].to(total.device, total.dtype)[seen]
        count[seen] += 1

    return torch.where(
        count > 0, total / count.clamp_min(1), boxes.motion.pose_time.view(-1)
    ).view(-1, 1)


def interpolate_boxes(
        sim_ds: SceneReconstructionDataset,
        batch,
        max_extrapolation: float = MAX_EXTRAPOLATION
) -> OrientedBoxes:
    """Boxes of the objects a batch's sweeps saw, each at the instant it was seen.

    The sweep's own time places the boxes well enough to project them, the
    projection reveals when the spin actually reached each object, and the
    segment velocity carries each box from one to the other. Without that step
    an object swept early or late in the revolution sits up to half a period
    away from the points it is supposed to contain.
    """
    # A fully static scene has no instances to sample -- skip the trajectory
    # solve and return no boxes, so every point falls to the background.
    if len(sim_ds.instances) == 0:
        return sim_ds.all_boxes[:0].contiguous()

    frame_time = batch_lidar_time(sim_ds, batch)
    boxes = sample_instance_boxes(sim_ds, frame_time, max_extrapolation)
    boxes = boxes.at_time(observed_instance_times(sim_ds, batch, boxes))

    return boxes[boxes.mask].contiguous()


def batch_lidar_time(sim_ds: SceneReconstructionDataset, batch) -> torch.Tensor:
    """Capture time of a batch's sweeps, averaged over its synchronized lidars."""
    return torch.stack([
        sim_ds.lidar_cameras[sensor_id][sample_idx].world_se3_camera.pose_time.view(())
        for sensor_id, sample_idx in [x[0] for x in batch['lidar']]
    ]).mean()


def init_loader_iterator(
        sim_ds: SceneReconstructionDataset,
        target_device: str | torch.device = 'cuda'
):
    indices = init_loader_indices(sim_ds)
    it = sim_ds.multi_loader_iterator(
        loaders={
            'image': sim_ds.camera_loader_dataset(None),
            'lidar': sim_ds.lidar_loader_dataset(None),
        },
        indices=indices,
        infinite_iterator=False,
        shuffle=False,
        target_device=target_device
    )
    for batch in track(it, description='Extracting instance points', total=len(indices)):
        images = [x[1] for x in batch['image']]
        cameras = [
            sim_ds.cameras[x[0][0]][x[0][1]].to(target_device)
            for x in batch['image']
        ]
        sweeps = [x[1] for x in batch['lidar']]
        ego_masks = [
            sim_ds.camera_masks[x[0][0]].to(target_device)
            for x in batch['image']
        ]

        # Boxes are sampled from the tracks rather than looked up by sensor key:
        # annotations can be sparser than the lidar, and a sweep without boxes
        # would push its dynamic objects into the background.
        boxes = interpolate_boxes(sim_ds, batch).to(target_device)

        yield {
            'images': images,
            'cameras': cameras,
            'camera_ids': [x[0][0] for x in batch['image']],
            'sweeps': sweeps,
            'boxes': boxes,
            'masks': ego_masks
        }
