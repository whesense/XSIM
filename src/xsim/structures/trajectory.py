from typing import Optional, Literal

import torch
from toast import DataTensor, SE3, LinearMotion, sample_trajectory

from .timestamp import Timestamp


class SE3Trajectory(DataTensor):
    _buffers = ["pose", "time"]
    _optional_buffers = ["mask"]

    pose: SE3
    time: Timestamp
    mask: Optional[torch.Tensor] = None

    def __init__(
            self,
            pose: SE3,
            time: Timestamp,
            mask: Optional[torch.Tensor] = None
    ):
        self.pose = pose  # [..., keyframes]
        self.time = time  # [..., keyframes]
        self.mask = mask  # [..., keyframes]

    def __mul__(self, right: SE3) -> "SE3Trajectory":
        pass

    def __rmul__(self, left: SE3) -> "SE3Trajectory":
        pass

    def sample(
            self,
            query_time: torch.Tensor,  # [..., queries]
            use_only_valid_keyframes: bool = True,
            extrapolate: bool = True,
            extrapolate_window: int = 1,
            mask_mode: Literal['and'] | Literal['or'] = 'and',
    ) -> tuple[LinearMotion, torch.Tensor | None]:
        out_q, out_t, out_mask, out_v, out_w, _  = sample_trajectory(
            query_times=query_time,
            seq_times=self.time.sec.value.float(),
            seq_quats=self.pose.q.data,
            seq_t=self.pose.t,
            seq_mask=self.mask,
            use_only_valid_keyframes=use_only_valid_keyframes,
            extrapolate=extrapolate,
            extrapolation_window=extrapolate_window,
            mask_mode=mask_mode,
            return_velocities=True,
            return_mask=self.mask is not None
        )
        return LinearMotion(
            pose=SE3(out_q, out_t),
            pose_time=query_time.broadcast_to(*self.shape, query_time.shape[-1]),
            linear_velocity=out_v,
            angular_velocity=out_w,
        ), out_mask

