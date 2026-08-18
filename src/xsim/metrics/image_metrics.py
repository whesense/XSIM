import torch
import torch.nn.functional as F

from pytorch_msssim import SSIM as SSIMModule
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


from .metric import Metric


class L1ImageMetric(Metric):
    def forward(self,
                render: torch.Tensor,
                target: torch.Tensor,
                target_mask: torch.Tensor
                ):
        return (render - target).mul(target_mask.unsqueeze(-1)).abs().mean().item()


class PSNR(Metric):
    round_digits = 2

    def __init__(self, base: int = 10):
        super().__init__()
        self.base = base

    def forward(
            self,
            render: torch.Tensor,
            target: torch.Tensor,
            target_mask: torch.Tensor
    ):
        target_mask = target_mask.unsqueeze(-1)
        render_masked = render * target_mask
        gt_masked = target * target_mask
        mse_value = F.mse_loss(render_masked, gt_masked, reduction='sum')
        mse_value = mse_value / (3 * target_mask.sum())
        return -self.base * mse_value.log10().item()


class SSIM(Metric):
    def __init__(self):
        super().__init__()
        self.ssim = SSIMModule(data_range=1.0, size_average=True, channel=3)

    def forward(
            self,
            render: torch.Tensor,
            target: torch.Tensor,
            target_mask: torch.Tensor
    ):
        target_mask = target_mask.unsqueeze(-1)
        gt = (target * target_mask).permute(2, 0, 1)[None]
        render = (render * target_mask).permute(2, 0, 1)[None]
        return self.ssim(gt, render)


class LPIPS(Metric):
    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).eval()

    def forward(
            self,
            render: torch.Tensor,
            target: torch.Tensor,
            target_mask: torch.Tensor
    ):
        target_mask = target_mask.unsqueeze(-1)
        self.lpips.eval()

        gt = (target * target_mask).permute(2, 0, 1)[None]
        render = (render * target_mask).permute(2, 0, 1)[None]
        return self.lpips(render, gt)
