from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from toast import DataTensor, SE3
from xsimgs.structures import ROIBox3D


class Sweep(DataTensor):
    _buffers = ["xyz", "origin"]
    _optional_buffers = ['intensity', 'colors', 'beam_idx', 'time', 'mask']

    xyz: torch.Tensor
    origin: torch.Tensor
    intensity: torch.Tensor = None
    colors: torch.Tensor = None
    lidar_idx: torch.Tensor = None
    beam_idx: torch.Tensor = None
    time: torch.Tensor = None
    mask: torch.Tensor = None

    def __init__(
            self,
            xyz: torch.Tensor,
            origin: torch.Tensor,
            intensity: Optional[torch.Tensor] = None,
            colors: Optional[torch.Tensor] = None,
            beam_idx: Optional[torch.Tensor] = None,
            time: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ):
        if origin.shape != xyz.shape:
            origin = origin.broadcast_to(xyz.shape)

        super().__init__(
            xyz, origin, intensity, colors, beam_idx, time, mask)

    @property
    def masked(self) -> "Sweep":
        return self[self.mask] if self.mask is not None else self

    @staticmethod
    def zeros(*shape, device: str = 'cpu'):
        return Sweep(
            xyz=torch.zeros(*shape, 3, device=device),
            origin=torch.zeros(*shape, 3, device=device),
            intensity=torch.zeros(*shape, 1, device=device),
            time=torch.zeros(*shape, 1, device=device),
        )

    def inside_roi(self, roi: ROIBox3D):
        return self if roi is None else self[roi.check_inside(self.xyz)]

    def outside_roi(self, roi: ROIBox3D):
        return self if roi is None else self[~roi.check_inside(self.xyz)]

    @property
    def ray_dirs(self):
        return F.normalize(self.xyz - self.origin, dim=-1)

    @property
    def ray_lengths(self):
        return (self.xyz - self.origin).norm(dim=-1)

    @property
    def origins(self):
        return self.origin

    def transform_(self, se3: SE3):
        dtype, device = self.dtype, self.device
        se3 = se3.to(dtype=dtype, device=device)

        self.xyz = (se3 @ (self.xyz.view(-1, 3))).view_as(self.xyz)
        self.origin = (se3 @ self.origin.view(-1, 3)).view_as(self.origin)
        return self

    def transform(self, se3: SE3):
        return self.clone().transform_(se3)

    def to_ply(self, output_path: Path | str):
        from plytorch import PointCloud
        colors = self.colors
        if colors is not None and colors.dtype == torch.float32:
            colors = colors.clamp(0, 1).mul(255)
        PointCloud(
            points=self.xyz,
            colors=colors.byte() if colors is not None else None
        ).save(str(output_path))
