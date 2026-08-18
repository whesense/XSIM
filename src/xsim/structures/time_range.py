import torch

from .timestamp import Timestamp


class TimeRange:
    tmin: float
    tmax: float

    def __init__(
            self,
            tmin: float | torch.Tensor | Timestamp,
            tmax: float | torch.Tensor | Timestamp,
    ):
        if isinstance(tmin, Timestamp):
            tmin = tmin.sec.value
        if isinstance(tmax, Timestamp):
            tmax = tmax.sec.value
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.duration = self.tmax - self.tmin

    @property
    def min(self) -> float: return self.tmin
    @property
    def max(self) -> float: return self.tmax

    def normalize_time(self, t: torch.Tensor | Timestamp) -> torch.Tensor:
        if isinstance(t, Timestamp):
            return Timestamp.from_sec((t.sec.value - self.tmin) / self.duration)
        else:
            return (t - self.tmin) / self.duration

