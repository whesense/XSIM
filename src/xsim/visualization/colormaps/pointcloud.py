import torch

from .torch_colormap import get_cmap


def export_pcd_colored_by_z(
        points: torch.Tensor,
        path: str,
        cmap: str = 'turbo',
        q: float = 0.01
):
    from plytorch import PointCloud

    cmap_obj = get_cmap(cmap)
    colors = cmap_obj(points[:, 2], normalize=True, byte=True, q_min=q, q_max=1 - q)
    PointCloud(points=points, colors=colors).save(path)
