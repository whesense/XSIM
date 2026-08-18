from dataclasses import dataclass
from typing import Union, Callable

import torch

from .activations import identity

SizeT = Union[int, Callable[["GaussianModel"], int]]


@dataclass(frozen=True)
class ParamSpec:
    size: SizeT
    # activation function that is applied on forward
    act: Callable = identity
    # inverse activation function. typically used at initialization
    inv_act: Callable = identity
    # whether forward method should put this parameter into activated container
    ret: bool = True
    # If set, this parameter won't be included in optimizable params
    # (always will be in param_buffers)
    non_optimizable: bool = False

    def resolve_size(self, model) -> int:
        return self.size if isinstance(self.size, int) else self.size(model)


class SpecMetaClass(type(torch.nn.Module)):
    """Metaclass merging every `param_specs` along the MRO into `_param_specs`.

    Base classes contribute first, subclasses override/extend,
    so a subclass only needs to declare the params it introduces.
    """
    def __new__(mcs, name, bases, ns):
        cls = super().__new__(mcs, name, bases, ns)
        merged: dict[str, ParamSpec] = {}
        for class_t in reversed(cls.__mro__):
            merged.update(vars(class_t).get("param_specs", {}))
        cls._param_specs = merged
        return cls
