import torch

from xsim.data import SensorType
from xsim.data.reconstruction.dataset import SensorImage
from xsim.metrics import Metric
from xsim.modeling import RenderInfo
from xsim.modeling.scene import Scene
from xsim.structures.init_instance import knn_points


def knn_dist(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    idxs = knn_points(pred.unsqueeze(0), gt.unsqueeze(0)).idx[0, :, 0]
    dists = (gt[idxs] - pred).norm(dim=-1).pow(2)
    return dists.mean()


def chamferdist(pred, gt):
    return (knn_dist(pred, gt) + knn_dist(gt, pred)).item()


class ChamferDist(Metric):
    sensor_types = [SensorType.LIDAR]
    round_digits = 2

    def get_sample(self, render: RenderInfo, batch: SensorImage):
        pred = render.result.info.get('points')
        gt = batch.image.masked.xyz
        return pred, gt

    def update(self, scene: Scene):
        for batch, render in zip(scene.batch, scene.renders):
            if batch.sensor_type not in self.sensor_types:
                continue
            pred, gt = self.get_sample(render, batch)
            if pred is None or len(pred) == 0 or len(gt) == 0:
                continue
            self.accumulate(batch.sensor_id, self.forward(pred, gt))

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        return chamferdist(pred, gt)
