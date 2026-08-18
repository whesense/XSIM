import torch

from xsim.utils import interpolate
from xsimgs.nodes import image_to_chw_rgb_byte

PNG_COMPRESSION_LEVEL = 1



def image_grid(
        images: dict[int, torch.Tensor],
        layout: list,
        downscale: float = 1.0,
) -> torch.Tensor:
    """Arrange rendered images into a (possibly ragged) grid.

    Args:
        images: ``sensor_id -> [H, W, 3]`` float32 CUDA image (values in 0..1).
        layout: grid of sensor ids describing placement. A flat ``list[int]`` is
            a single row (horizontal concat); a nested ``list[list[int]]`` is a
            2D grid. Any entry that is ``None`` (or an id missing from
            ``images``) becomes a black cell. Rows may be ragged.
        downscale: divide each image's height/width by this factor before
            stacking (done in float, antialiased).

    Cells of differing size are placed top-left; each row is as tall as its
    tallest cell and each column as wide as its widest cell, so rows and columns
    stay aligned. Padding is left black.

    Returns:
        ``[3, H, W]`` uint8 CUDA tensor
    """

    # Normalize to a list of rows (a flat list is one horizontal row).
    if len(layout) > 0 and not isinstance(layout[0], (list, tuple)):
        rows = [list(layout)]
    else:
        rows = [list(row) for row in layout]
    ncols = max((len(r) for r in rows), default=0)

    # Convert every referenced image to [3, h, w] uint8, downscaling first.
    cells: dict[tuple[int, int], torch.Tensor] = {}
    for i, row in enumerate(rows):
        for j, sid in enumerate(row):
            if sid is None or sid not in images:
                continue
            img = images[sid]
            if downscale != 1.0:
                h, w = img.shape[0], img.shape[1]
                size = (max(1, round(h / downscale)), max(1, round(w / downscale)))
                img = interpolate(img, size, antialias=True)
            cells[(i, j)] = image_to_chw_rgb_byte(img)

    if not cells:
        raise ValueError('image_grid: no images to render for the given layout')

    # Per-row height / per-column width taken over the populated cells; empty
    # rows/columns collapse to zero and are skipped.
    row_h = [0] * len(rows)
    col_w = [0] * ncols
    for (i, j), c in cells.items():
        row_h[i] = max(row_h[i], c.shape[1])
        col_w[j] = max(col_w[j], c.shape[2])

    # Prefix offsets of each row/column in the assembled canvas.
    row_y, y = [0] * len(rows), 0
    for i in range(len(rows)):
        row_y[i] = y
        y += row_h[i]
    col_x, x = [0] * ncols, 0
    for j in range(ncols):
        col_x[j] = x
        x += col_w[j]

    # One allocation, then copy each cell straight into its slice.
    device = next(iter(cells.values())).device
    grid = torch.zeros((3, y, x), dtype=torch.uint8, device=device)
    for (i, j), c in cells.items():
        grid[:, row_y[i]:row_y[i] + c.shape[1], col_x[j]:col_x[j] + c.shape[2]] = c
    return grid




def encode_jpeg(image: torch.Tensor, quality: int = 95) -> torch.Tensor:
    """GPU-encode a ``[3, H, W]`` uint8 tensor to JPEG bytes (1D uint8 tensor)."""
    from torchvision.io import encode_jpeg as tv_encode_jpeg
    return tv_encode_jpeg(image, quality=quality)


def encode_png(image: torch.Tensor,
               compression_level: int = PNG_COMPRESSION_LEVEL) -> torch.Tensor:
    """Encode a ``[3, H, W]`` uint8 tensor to PNG bytes (1D uint8 tensor)."""
    from torchvision.io import encode_png as tv_encode_png
    return tv_encode_png(image.cpu().contiguous(), compression_level)


def chw_byte_image(image: torch.Tensor) -> torch.Tensor:
    """Put an image in the ``[3, H, W]`` uint8 layout the encoders take.

    Accepts ``[H, W, 3]`` or ``[3, H, W]``, uint8 or float (values in 0..1).
    """
    if image.dtype != torch.uint8:
        return image_to_chw_rgb_byte(image)

    if image.shape[-1] in (1, 3):
        # ground-truth images come in as [H, W, 3] bytes
        return image.permute(2, 0, 1)

    return image


def write_image(path: str, image: torch.Tensor,
                compression_level: int = PNG_COMPRESSION_LEVEL):
    """Write an image as PNG.

    Accepts ``[H, W, 3]`` or ``[3, H, W]``, uint8 or float (values in 0..1).
    """
    encoded = encode_png(chw_byte_image(image), compression_level)
    with open(path, 'wb') as f:
        f.write(memoryview(encoded.numpy()))
