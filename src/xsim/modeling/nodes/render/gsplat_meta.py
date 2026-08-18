import math
from typing import Literal, Optional

import torch

from xsimgs._C import mask_radii


def prepare_meta(
        means: torch.Tensor,  # [..., N, 3]
        quats: torch.Tensor,  # [..., N, 4]
        scales: torch.Tensor,  # [..., N, 3]
        opacities: torch.Tensor,  # [..., N]

        viewmats: torch.Tensor,  # [..., C, 4, 4]
        Ks: torch.Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
        tile_size: int = 16,
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
        with_ut: bool = False,
        radial_coeffs: Optional[torch.Tensor] = None,  # [..., C, 6] or [..., C, 4]
        tangential_coeffs: Optional[torch.Tensor] = None,  # [..., C, 2]
        thin_prism_coeffs: Optional[torch.Tensor] = None,  # [..., C, 4]
        # rolling shutter
        rolling_shutter: "RollingShutterType" = None,
        viewmats_rs: Optional[torch.Tensor] = None,  # [..., C, 4, 4]
        packed_mask: torch.Tensor | None = None,
):
    from gsplat import (
        fully_fused_projection,
        fully_fused_projection_with_ut,
        isect_tiles,
        isect_offset_encode,
        RollingShutterType
    )
    if rolling_shutter is None:
        rolling_shutter = RollingShutterType.GLOBAL

    if with_ut:
        proj_results = fully_fused_projection_with_ut(
            means,
            quats,
            scales,
            opacities,  # use opacities to compute a tighter bound for radii.
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            calc_compensations=(rasterize_mode == "antialiased"),
            camera_model=camera_model,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            rolling_shutter=rolling_shutter,
            viewmats_rs=viewmats_rs,
        )
    else:
        # Project Gaussians to 2D. Directly passing in {quats, scales} is faster
        # than precomputing covars.
        proj_results = fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            packed=False,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            sparse_grad=False,
            calc_compensations=(rasterize_mode == "antialiased"),
            camera_model=camera_model,
            opacities=opacities,  # use opacities to compute a tighter bound for radii.
        )

    radii, means2d, depths, conics, compensations = proj_results
    if packed_mask is not None:
        mask_radii(packed_mask, radii)

    meta = {
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": opacities,
    }

    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        segmented=False,
        packed=False,
        n_images=1,
        image_ids=None,
        gaussian_ids=None,
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(tile_height, tile_width)

    meta.update({
        "n_cameras": 1,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tile_size": tile_size,
        "width": width,
        "height": height,

        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
    })

    return meta
