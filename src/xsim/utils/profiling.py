import contextlib

import torch
from torch.profiler import record_function

_enabled = False
_NULL = contextlib.nullcontext()


def enabled() -> bool:
    return _enabled


def set_enabled(flag: bool) -> None:
    global _enabled
    _enabled = flag


def annotate(name: str):
    """Context manager labelling a code region (no-op unless profiling is on)."""
    if not _enabled:
        return _NULL
    return _annotate(name)


@contextlib.contextmanager
def _annotate(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        with record_function(name):
            yield
    finally:
        torch.cuda.nvtx.range_pop()


def _emit_instant(label: str) -> None:
    torch.cuda.nvtx.mark(label)
    with record_function(label):  # zero-length slice -> visible in torch.profiler
        pass


def mark_backward(outputs, name: str) -> None:
    """Emit ``{name}.backward`` when the gradient of ``outputs`` is computed.

    ``outputs`` may be a tensor, an ``ActivatedGaussians`` (``.data`` dict),
    or a (nested) list/tuple/dict of those. Hooks are attached to freshly
    created tensors, so they do not accumulate across iterations.
    """
    if not _enabled:
        return
    label = '{}.backward'.format(name)
    for tensor in _grad_tensors(outputs):
        tensor.register_hook(lambda grad, label=label: (_emit_instant(label), grad)[1])
        return  # one marker per node is enough to locate its backward


def _grad_tensors(obj, out=None):
    if out is None:
        out = []
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad:
            out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _grad_tensors(item, out)
    elif isinstance(obj, dict):
        for item in obj.values():
            _grad_tensors(item, out)
    elif hasattr(obj, 'data') and isinstance(getattr(obj, 'data'), dict):
        # ActivatedGaussians and similar param containers
        _grad_tensors(obj.data, out)
    return out
