from collections import defaultdict
import torch

from xsim.data import SensorType
from xsim.data.reconstruction.dataset import SensorImage
from xsim.modeling import RenderInfo
from xsim.modeling.scene import Scene


class Metric(torch.nn.Module):
    sensor_types = [SensorType.CAMERA]
    round_digits = 4

    total_error: float
    total_cnt: int
    per_sensor_accum: dict[int, tuple[float, int]]

    def __init__(self):
        super().__init__()
        self.reset()

    def value(self, sensor_id: int | None = None):
        if sensor_id is None:
            return self.total_error / self.total_cnt
        error, cnt = self.per_sensor_accum[sensor_id]
        return error / cnt

    def sensor_ids(self):
        return sorted(self.per_sensor_accum)

    def format(self, value: float) -> str:
        return ('{:.'+str(self.round_digits)+'f}').format(float(value))

    def format_value(self, sensor_id: int | None = None) -> str:
        return self.format(self.value(sensor_id))

    def reset(self):
        self.total_error = 0
        self.total_cnt = 0
        self.per_sensor_accum = defaultdict(lambda: (0, 0))

    def accumulate(self, sensor_id: int, error):
        self.total_error += error
        self.total_cnt += 1

        cur_error, cur_cnt = self.per_sensor_accum[sensor_id]
        self.per_sensor_accum[sensor_id] = (cur_error + error, cur_cnt + 1)

    def get_sample(self, render: RenderInfo, batch: SensorImage):
        rgb = render.result.color.clamp(0.0, 1.0)
        gt = (batch.image / 255.0).clamp(0.0, 1.0)
        valid_mask = batch.gt_mask  # .unsqueeze(-1)
        return rgb, gt, valid_mask

    def update(self, scene: Scene):
        for batch, render in zip(scene.batch, scene.renders):
            if batch.sensor_type not in self.sensor_types:
                continue

            self.accumulate(
                batch.sensor_id,
                self.forward(*self.get_sample(render, batch))
            )

    def forward(
            self,
            render: torch.Tensor,
            target: torch.Tensor,
            target_mask: torch.Tensor
    ) -> float:
        pass


class DummyMetric(Metric):
    def forward(
            self,
            render: torch.Tensor,
            target: torch.Tensor,
            target_mask: torch.Tensor
    ) -> float:
        return 1.0
