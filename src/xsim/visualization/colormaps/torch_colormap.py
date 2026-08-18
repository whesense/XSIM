from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .segmentation_palettes import (
    ADE20K_COLORS, NUSC_COLORS32, NUSC_COLORS16,
)


@dataclass
class TorchCMap:
    data: torch.Tensor
    bad: torch.Tensor
    size: int = None

    def __post_init__(self):
        self.size = len(self.data)

    def to(self, device: torch.device | str):
        return TorchCMap(data=self.data.to(device), bad=self.bad.to(device))


def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.0031308,
        f * 12.92,
        torch.pow(torch.clamp(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
    )


def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.04045,
        f / 12.92,
        torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4),
    )


class Colormap:
    _cmaps: dict[tuple[str, torch.device], TorchCMap] = {}

    def __init__(self, cmap: str):
        self.cmap = cmap

    @staticmethod
    def map_data(
            cmap, data,
            normalize: bool = False,
            q_min: float = None,
            q_max: float = None,
            vmin: float = None,
            vmax: float = None,
            mode: str = 'linear_srgb'
    ):
        data = data.clone()  # don't mutate the caller's tensor
        mask = ~torch.isfinite(data)
        data[mask] = 0

        if normalize:
            values = data[~mask]
            if vmin is None:
                vmin = (
                    torch.min(values) if q_min is None
                    else torch.kthvalue(values, int(q_min * len(values))).values
                )
            if vmax is None:
                vmax = (
                    torch.max(values) if q_max is None
                    else torch.kthvalue(values, int(q_max * len(values))).values
                )
            data = (data - vmin) / (vmax - vmin)

        data = data.mul(cmap.size - 1).clamp(0, cmap.size - 1)

        if mode == 'nearest':
            data = cmap.data[data.long()]
        else:
            cmap_data = _srgb_to_rgb(cmap.data) if mode == 'linear_srgb' else cmap.data

            alpha = data - data.floor()
            lower_idx = data.floor().int()
            higher_idx = (lower_idx + 1).clamp_max_(len(cmap_data) - 1)
            data = torch.lerp(
                cmap_data[lower_idx], cmap_data[higher_idx], alpha.unsqueeze(-1)
            )

            if mode == 'linear_srgb':
                data = _rgb_to_srgb(data)

        return data, mask

    def __call__(self, data, normalize: bool = False,
                 q_min: float = None, q_max: float = None,
                 vmin: float = None, vmax: float = None,
                 mode: str = 'linear_srgb', byte: bool = False
                 ):
        cmap_range = self._get_data(self.cmap, data.device)
        result, mask = self.map_data(
            cmap_range, data, normalize, q_min, q_max, vmin=vmin, vmax=vmax
        )
        result[mask] = cmap_range.bad
        if byte:
            result = result.mul(255).byte()
        return result

    @staticmethod
    def _generate_cmap(cmap: str, nsteps) -> TorchCMap:
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        bad = cmap.get_bad()
        steps = np.arange(nsteps) / (nsteps - 1.0)
        return TorchCMap(
            data=torch.as_tensor(cmap(steps)[:, :3].astype(np.float32)),
            bad=torch.as_tensor(bad).float().view(-1)[:3]
        )

    @classmethod
    def _get_data(cls,
                  cmap: str | ListedColormap,
                  device: torch.device = 'cpu',
                  nsteps=256):
        key = (cmap, device) if isinstance(cmap, str) else (cmap.name, device)
        if key not in cls._cmaps:
            cls._cmaps[key] = cls._generate_cmap(cmap, nsteps).to(device)

        return cls._cmaps[key]


class SegmentationColormap(Colormap):
    @staticmethod
    def map_data(cmap, data, normalize: bool = False,
                 q_min: float = None, q_max: float = None, **kwargs):
        # Segmentation gathers colors from the palette by integer label; range
        # kwargs (vmin/vmax/mode) passed by Colormap.__call__ are ignored.
        return cmap.data[data.long()], torch.zeros_like(data, dtype=torch.bool)

    @staticmethod
    def _generate_cmap(cmap, nsteps):
        if cmap == 'ade20k':
            return TorchCMap(data=ADE20K_COLORS, bad=torch.zeros(3, dtype=torch.uint8))
        elif cmap == 'nuscenes':
            return TorchCMap(data=NUSC_COLORS32, bad=torch.zeros(3, dtype=torch.uint8))
        elif cmap == 'nusc16':
            return TorchCMap(data=NUSC_COLORS16, bad=torch.zeros(3, dtype=torch.uint8))
        else:
            raise ValueError(
                'Unknown segmentation cmap. Supported: ade20k, nuscenes, nusc16'
            )


Viridis = Colormap('viridis')
Plasma = Colormap('plasma')
Inferno = Colormap('inferno')
Magma = Colormap('magma')
Jet = Colormap('jet')
Turbo = Colormap('turbo')
Rainbow = Colormap('rainbow')


def get_cmap(cmap: str) -> Colormap:
    """Return an object with a matplotlib-colormap-like interface that takes a
    torch.Tensor as input.

    Much faster than casting a torch.Tensor to numpy, applying a matplotlib cmap
    and casting back, especially for tensors on the GPU.

    The resulting object's ``__call__`` maps a tensor of 0..1 values to colors.
    Tensors of arbitrary shape are supported; the output adds a trailing dim of
    size 3. E.g. an ``NxMxK`` input yields an ``NxMxKx3`` float32 output with
    colors in 0..1 (or 0..255 uint8 when ``byte=True``).

    Parameters
    ----------
    cmap: str
        Name of the matplotlib colormap.
    """
    return Colormap(cmap)


def get_seg_cmap(cmap: str = 'ade20k') -> SegmentationColormap:
    return SegmentationColormap(cmap)
