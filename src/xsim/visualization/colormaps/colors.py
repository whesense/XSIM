import matplotlib.colors as mcolors
import torch


def get_color(color: str, device: torch.device | str = 'cpu') -> torch.Tensor:
    return torch.as_tensor(mcolors.to_rgb(color), device=device).mul(255).byte()


def color_tensor(
        numels: int,
        color: str,
        device: torch.device | str = 'cpu'
) -> torch.Tensor:
    return get_color(color, device=device).view(1, 3).expand(numels, -1)


def color_by_mask(mask: torch.Tensor, color_true: str, color_false: str) -> torch.Tensor:
    color_true = color_tensor(
        mask.numel(), color_true, device=mask.device
    ).view(*mask.shape, 3)
    color_false = color_tensor(
        mask.numel(), color_false, device=mask.device
    ).view(*mask.shape, 3)
    return torch.where(mask.view(-1, 1), color_true, color_false)
