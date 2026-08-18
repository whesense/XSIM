import torch

from xsim.modeling.scene import Scene
from xsim.modeling.gaussian import ActivatedGaussians


class RegularizationLoss(torch.nn.Module):
    """Base class for losses computed from a gaussian node's forward output.

    ``forward`` receives the owning node, the scene, and the node's activated
    gaussians ``gs`` (the node's forward output). ``gs`` may be ``None`` when the
    node produced nothing this step, in which case the loss is zero.
    """

    def forward(self, node, scene: Scene, gs: ActivatedGaussians) -> torch.Tensor:
        raise NotImplementedError


class LidarDensityRegLoss(RegularizationLoss):
    """L1 between the optical density and the lidar density."""

    def forward(self, node, scene: Scene, gs: ActivatedGaussians) -> torch.Tensor:
        if gs is None:
            return 0
        return (gs.density.view(-1) - gs.lidar_density.view(-1)).abs().mean()


class SharpShapeRegLoss(RegularizationLoss):
    """Force gaussians to a more spherical form when the scale aspect ratio
    exceeds ``max_scale_ratio``."""

    def __init__(self, max_scale_ratio: float = 10.0):
        super().__init__()
        self.max_scale_ratio = max_scale_ratio

    def forward(self, node, scene: Scene, gs: ActivatedGaussians) -> torch.Tensor:
        if gs is None:
            return 0
        max_scale = gs.scale.amax(dim=-1)
        min_scale = gs.scale.amin(dim=-1)
        ratio = (max_scale / min_scale).clamp_min(self.max_scale_ratio)
        return (ratio - self.max_scale_ratio).mean()


class FlattenRegLoss(RegularizationLoss):
    """Force gaussians to be flat (smallest scale -> 0)."""

    def __init__(self, max_scale: float = 30.0):
        super().__init__()
        self.max_scale = max_scale

    def forward(self, node, scene: Scene, gs: ActivatedGaussians) -> torch.Tensor:
        if gs is None:
            return 0
        return gs.scale.amin(dim=-1).clamp(0, self.max_scale).mean()


class SparseRegLoss(RegularizationLoss):
    """Push densities toward 0 or 1 (binary-entropy), over rendered particles."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.m1eps = 1 - eps
        self.loss = torch.nn.BCELoss()

    def forward(self, node, scene: Scene, gs: ActivatedGaussians) -> torch.Tensor:
        if gs is None:
            return 0
        radii = node.strategy.radii_buffer
        density = gs.density[radii > 0].clamp(self.eps, self.m1eps)
        return self.loss(density, density)


class MaxSquaredScaleRegLoss(RegularizationLoss):
    """Regularize the maximum scale of each particle."""

    def forward(self, node, scene: Scene, gs: ActivatedGaussians) -> torch.Tensor:
        if gs is None:
            return 0
        max_scale = gs.scale.amax(dim=-1)
        return (max_scale * max_scale).mean()


class OutOfBoundLoss(RegularizationLoss):
    """Penalize instanced particles that fall outside their object box."""

    def forward(self, node, scene: Scene, gs: ActivatedGaussians) -> torch.Tensor:
        if gs is None:
            return 0
        inst_ids = node.model.get_param_value("instance_ids").flatten()
        half_sizes = node.instance_sizes.mul(0.5)[inst_ids]
        return torch.relu(gs.positions_local.abs() - half_sizes).mean()


class DensityRegLoss(RegularizationLoss):
    """L1 on the density."""

    def forward(self, node, scene: Scene, gs: ActivatedGaussians) -> torch.Tensor:
        if gs is None:
            return 0
        return gs.density.abs().mean()


class ScaleRegLoss(RegularizationLoss):
    """L1 on the scale."""

    def forward(self, node, scene: Scene, gs: ActivatedGaussians) -> torch.Tensor:
        if gs is None:
            return 0
        return gs.scale.abs().mean()
