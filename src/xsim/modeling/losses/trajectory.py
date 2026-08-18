import torch

from xsim.modeling.scene import Scene


class PoseAdjustmentPenalty(torch.nn.Module):
    """
    L1 penalty on a TrajectoryNode's pose adjustments.
    """

    def __init__(self, q_weight: float = 1.0, t_weight: float = 1.0):
        super().__init__()
        self.q_weight = q_weight
        self.t_weight = t_weight

    def forward(self, node, scene: Scene, output) -> torch.Tensor:
        if not node.optimize:
            return node.instance_time.new_zeros(())

        q_adj = node.instance_q_adj   # [objects, keyframes, 4]
        t_adj = node.instance_t_adj   # [objects, keyframes, 3]
        num_objects, num_keyframes = q_adj.shape[:2]

        # keyframes touched by gradient this step, restricted to active objects
        idx = output['out_indices'].long().clamp(0, num_keyframes - 1)
        idx = idx.reshape(num_objects, -1)
        touched = torch.zeros(
            num_objects, num_keyframes, dtype=torch.bool, device=q_adj.device
        )
        touched.scatter_(1, idx, torch.ones_like(idx, dtype=torch.bool))
        touched &= output['mask'].view(-1, 1)

        if not touched.any():
            return q_adj.new_zeros(())

        loss_q = q_adj[touched].abs().mean()
        loss_t = t_adj[touched].abs().mean()
        return self.q_weight * loss_q + self.t_weight * loss_t
