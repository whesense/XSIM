from enum import Enum

import torch
from toast import DataTensor


class TimestampScale(float, Enum):
    SECOND = 1.0
    MILLISECOND = 1e3
    MICROSECOND = 1e6
    NANOSECOND = 1e9
    

class Timestamp(DataTensor):
    _buffers = ["value"]
    _non_tensor_data = ["scale"]
    
    value: torch.Tensor
    scale: TimestampScale
    
    def __init__(self, value: torch.Tensor, scale: TimestampScale):
        self.value = (
            torch.as_tensor(value) 
            if not isinstance(value, torch.Tensor) 
            else value
        )
        self.scale = scale
        
    @staticmethod
    def from_sec(value: torch.Tensor):
        return Timestamp(value, TimestampScale.SECOND)
    
    @staticmethod
    def from_ms(value: torch.Tensor):
        return Timestamp(value, TimestampScale.MILLISECOND)
    
    @staticmethod
    def from_us(value: torch.Tensor):
        return Timestamp(value, TimestampScale.MICROSECOND)
    
    @staticmethod
    def from_ns(value: torch.Tensor):
        return Timestamp(value, TimestampScale.NANOSECOND)
    
    def convert(self, new_scale: TimestampScale):
        if self.scale == new_scale:
            return self

        scale_coef = new_scale.value / self.scale.value
        return Timestamp(
            self.value.double() * scale_coef,
            new_scale
        )
    
    @property
    def sec(self):
        return self.convert(TimestampScale.SECOND)
    
    @property
    def ms(self):
        return self.convert(TimestampScale.MILLISECOND)
    
    @property
    def us(self):
        return self.convert(TimestampScale.MICROSECOND)
    
    @property
    def ns(self):
        return self.convert(TimestampScale.NANOSECOND)
    
    @property
    def range(self):
        return Timestamp(
            self.value.amax() - self.value.amin(),
            self.scale
        )

    def __neg__(self) -> "Timestamp":
        return Timestamp(-self.value, self.scale)

    def _binary_op(self, other, fn):
        if isinstance(other, Timestamp):
            other = other.convert(self.scale).value

        return Timestamp(
            value=fn(self.value, other),
            scale=self.scale
        )

    def __add__(self, other) -> "Timestamp":
        return self._binary_op(other, lambda x, y: x + y)

    def __radd__(self, other) -> "Timestamp":
        return self._binary_op(other, lambda x, y: x + y)

    def __sub__(self, other) -> "Timestamp":
        return self._binary_op(other, lambda x, y: x - y)

    def __truediv__(self, other):
        return self._binary_op(other, lambda x, y: x / y)

    def __mul__(self, other) -> "Timestamp":
        return self._binary_op(other, lambda x, y: x * y)

    def __rmul__(self, other) -> "Timestamp":
        return self._binary_op(other, lambda x, y: x * y)

    def __le__(self, other):
        return self._binary_op(other, lambda x, y: x <= y).value

    def __ge__(self, other):
        return self._binary_op(other, lambda x, y: x >= y).value

    def __lt__(self, other):
        return self._binary_op(other, lambda x, y: x < y).value

    def __gt__(self, other):
        return self._binary_op(other, lambda x, y: x > y).value

    def __eq__(self, other):
        return self._binary_op(other, lambda x, y: x == y).value

    def __ne__(self, other):
        return self._binary_op(other, lambda x, y: x != y).value
