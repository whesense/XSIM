from typing import Optional

from chamferdist.chamfer import knn_points, knn_gather
from toast import DataTensor
import torch


def knn_avg(queries_points, gt_points, gt_values, k: int = 1):
    _, idxs, _ = knn_points(queries_points[None], gt_points[None], K=k)
    return knn_gather(gt_values[None], idxs)[0].float().mean(dim=1).byte()


class SceneInitInstance(DataTensor):
    _buffers = ['points', 'colors', 'visibility']

    points: torch.Tensor
    colors: torch.Tensor
    visibility: torch.Tensor

    def __init__(
            self,
            points: torch.Tensor,
            colors: Optional[torch.Tensor] = None,
            visibility: Optional[torch.Tensor] = None
    ):
        self.points = points

        if colors is None:
            colors = torch.rand(
                len(self.points), 3, device=points.device
            ).mul(255).byte()

        if visibility is None:
            visibility = torch.ones(
                len(self.points), device=points.device, dtype=torch.bool
            )

        self.colors = colors
        self.visibility = visibility

    @property
    def visible(self):
        return self[self.visibility]

    @property
    def invisible(self):
        return self[~self.visibility]

    def subsample(self, num_points: int):
        p = self.points
        if len(p) < num_points:
            return self
        return self[torch.randperm(len(p), device=p.device)[:num_points]]

    def flip_aug(self, axis: int = 1, do: bool = True):
        if not do: return self
        new_points = self.points.clone()
        new_points[:, axis] *= -1
        return self.cat([
            self,
            SceneInitInstance(
                points=new_points,
                colors=self.colors,
                visibility=self.visibility
            )
        ])

    def to_ply(self, path: str):
        from plytorch import PointCloud
        PointCloud(points=self.points, colors=self.colors).save(path)

    def color_invisible(self, k: int = 1) -> "SceneInitInstance":
        v = self.visible
        points_gt, colors_gt = v.points, v.colors
        points_q = self.points[~self.visibility]

        if len(points_q) == 0:
            return self
        if len(points_gt) == 0:
            self.colors = torch.rand_like(
                self.colors, dtype=torch.float32
            ).mul(255).byte()
            return self

        self.colors[~self.visibility] = knn_avg(
            points_q, points_gt, colors_gt, k=k)
        return self
