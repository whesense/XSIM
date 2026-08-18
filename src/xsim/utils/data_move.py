from collections.abc import Mapping
import torch

BUILTIN_TYPES = [int, float, str, bool, complex, bytes]

def torch_move(data, device: torch.device | str = 'cpu', non_blocking: bool = False):
    if type(data) in BUILTIN_TYPES:
        return data

    if hasattr(data, 'to'):
        return data.to(device, non_blocking=non_blocking)

    if isinstance(data, Mapping):
        return {k: torch_move(v, device) for k, v in data.items()}

    if isinstance(data, list) | isinstance(data, tuple):
        return data.__class__([torch_move(e, device) for e in data])

    return data


def get_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype)

    return dtype
