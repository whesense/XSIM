import warnings

import torch
import torch.nn.functional as F


def apply_map_to_image(
    src: torch.Tensor,
    fn,
    *args,
    mode: str = "bilinear",
    align_corners: bool = False,
    channels_first: bool = False,
    mask_epsilon: float = 1e-3,
    **kwargs,
):
    img = src
    if len(img.shape) not in [2, 3]:
        raise ValueError('Image must have either [H, W], or [H, W, C] '
                         'or [C, H, W] (if channels_first=True) shape')
    if len(img.shape) == 2:
        img = img.unsqueeze(-1)
    if not channels_first:
        img = img.permute(2, 0, 1)

    # add batch dimension
    img = img.unsqueeze(0)

    # cast_to_float = ((src.dtype != torch.float32) and src.is_cuda) or (src.dtype == torch.bool)
    # if cast_to_float:
    # cast to float since interpolation and grid_sample
    # have no implementation for non-float dtype
    # img = img.float()
    cast_to_float = False

    if mode == 'nearest':
        align_corners = None

    with warnings.catch_warnings(action="ignore"):
        try:
            img = fn(img, *args, **kwargs, mode=mode, align_corners=align_corners)[0]
        except NotImplementedError:
            cast_to_float = True
            img = img.float()
            img = fn(img, *args, **kwargs, mode=mode, align_corners=align_corners)[0]

    if cast_to_float:
        # revert dtype back
        if src.dtype == torch.bool:
            img = img > (1.0 - mask_epsilon)
        if src.dtype == torch.uint8:
            img = img.clamp(0, 255).byte()
        else:
            img = img.to(dtype=src.dtype)

    if not channels_first:
        img = img.permute(1, 2, 0)

    if len(src.shape) == 2:
        img = img[..., 0]

    return img


def grid_sample(
        src: torch.Tensor,
        grid: torch.Tensor,
        mode: str = "bilinear",
        align_corners: bool = False,
        channels_first: bool = False,
        mask_epsilon: float = 1e-3,
):
    return apply_map_to_image(
        src, F.grid_sample, grid,
        mode=mode, align_corners=align_corners,
        channels_first=channels_first,
        mask_epsilon=mask_epsilon,
    )


def interpolate(
        src: torch.Tensor,
        target_size: tuple[int, int],
        mode: str = "bilinear",
        antialias: bool = False,
        align_corners: bool = False,
        channels_first: bool = False,
        mask_epsilon: float = 1e-3,
):
    return apply_map_to_image(
        src, F.interpolate, target_size,
        mode=mode, align_corners=align_corners,
        channels_first=channels_first,
        mask_epsilon=mask_epsilon,
        antialias=antialias,
    )

