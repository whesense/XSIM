import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM

from xsim.data.loaders import SensorType
from xsim.modeling.scene import Scene


class RenderLoss(torch.nn.Module):
    """
    Base class for losses comparing a rendered quantity to its ground truth.
    """

    render_key = 'color'
    sensor_types = [SensorType.CAMERA]

    def __init__(self, sensor_types: list = None):
        super().__init__()
        if sensor_types is not None:
            self.sensor_types = [
                SensorType[s.upper()] if isinstance(s, str) else s
                for s in sensor_types
            ]

    def gt(self, sensor_image):
        raise NotImplementedError

    def loss_fn(self, gt, render, mask, render_info):
        raise NotImplementedError

    def forward(self, node, scene: Scene, result) -> torch.Tensor:
        loss = 0
        for i, render in enumerate(scene.renders):
            if render.sensor_type not in self.sensor_types:
                continue
            render_v = getattr(render.result, self.render_key)
            if render_v is None:
                continue
            sensor_image = scene.batch[i]
            gt = self.gt(sensor_image)
            if gt is None:
                continue
            loss = loss + self.loss_fn(gt, render_v, sensor_image.gt_mask, render)
        return loss


class ImageLoss(RenderLoss):
    render_key = 'color'

    def gt(self, sensor_image):
        # camera images are loaded as uint8 [0, 255]; the renderer is float [0, 1]
        return sensor_image.image.float() / 255.0


class L1ImageLoss(ImageLoss):
    def loss_fn(self, gt, render, mask, render_info):
        return (gt - render).abs().mul(mask.unsqueeze(-1)).mean()


class ClippingAwareL1ImageLoss(ImageLoss):
    """L1 image loss that ignores pixels clipped to white in the GT when the
    prediction already exceeds them (avoids penalizing overexposed regions)."""

    def __init__(self, gt_value_clip: int = 254, sensor_types: list = None):
        super().__init__(sensor_types)
        self.gt_value_clip = gt_value_clip / 255

    def loss_fn(self, gt, render, mask, render_info):
        m = mask.unsqueeze(-1)
        pred_masked = render * m
        gt_masked = gt * m
        nonclip_mask = ~((gt_masked > self.gt_value_clip) & (pred_masked >= gt_masked))
        return (pred_masked - gt_masked).abs()[nonclip_mask].sum() / (3 * mask.sum())


class SSIMImageLoss(ImageLoss):
    def __init__(self, sensor_types: list = None):
        super().__init__(sensor_types)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    def loss_fn(self, gt, render, mask, render_info):
        m = mask[..., None]
        gt = (gt * m).permute(2, 0, 1)[None]
        render = (render.clamp(0, 1) * m).permute(2, 0, 1)[None]
        return 1 - self.ssim(gt, render)


class OpacityLoss(RenderLoss):
    """Supervise rendered alpha against the sky mask (sky -> alpha 0)."""

    render_key = 'alpha'

    def __init__(self, eps: float = 1e-6, sensor_types: list = None):
        super().__init__(sensor_types)
        self.eps = eps

    def gt(self, sensor_image):
        return sensor_image.info.get('sky_mask')

    def loss_fn(self, gt, render, mask, render_info):
        render = render.reshape(gt.shape).clamp(self.eps, 1 - self.eps)
        return F.binary_cross_entropy(render, 1.0 - gt.float())


class SkyOnlyOpacityLoss(OpacityLoss):
    """Penalize non-zero alpha only at sky pixels."""

    def loss_fn(self, gt, render, mask, render_info):
        render = render.reshape(gt.shape)
        return render[gt > 0.5].abs().mean()


class SkyLogitLoss(RenderLoss):
    """
    Supervise the rendered sky map against the segmentation sky mask.
    """

    render_key = 'sky_map'

    def __init__(self, eps: float = 1e-6, sensor_types: list = None):
        super().__init__(sensor_types)
        self.eps = eps

    def gt(self, sensor_image):
        return sensor_image.info.get('sky_mask')

    def loss_fn(self, gt, render, mask, render_info):
        render = render.reshape(gt.shape).clamp(self.eps, 1 - self.eps)
        return F.binary_cross_entropy(render, gt.float())


class L1DepthLoss(RenderLoss):
    """
    L1 between rendered depth and the lidar-projected depth map.
    """

    render_key = 'depth'
    min_depth = 0.01
    max_depth = 80.0

    def gt(self, sensor_image):
        return sensor_image.info.get('depth')

    def loss_fn(self, gt, render, mask, render_info):
        render = render.reshape(gt.shape)
        depth_mask = (gt > self.min_depth) & (gt < self.max_depth) & (render > 1e-4)
        return (render[depth_mask] - gt[depth_mask]).abs().mean()


class LidarDepthLoss(RenderLoss):
    """
    L1 between rendered depth and lidar ray lengths.
    """

    render_key = 'depth'
    sensor_types = [SensorType.LIDAR]
    min_depth = 0.01

    def gt(self, sensor_image):
        return sensor_image.image.ray_lengths

    def loss_fn(self, gt, render, mask, render_info):
        render = render.reshape(gt.shape)
        valid = mask & (gt > self.min_depth) & (render > 1e-4)
        return (render[valid] - gt[valid]).abs().mean()
