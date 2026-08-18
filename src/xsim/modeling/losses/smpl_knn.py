import torch
from chamferdist.chamfer import knn_points, knn_gather
from xsim.modeling.nodes import GaussianSceneNode
from xsim.modeling.nodes.gaussian.smpl_node import SMPLSceneNode


class VoxelDeformerRegLoss(torch.nn.Module):
    def __init__(
            self,
            lambda_std_w: float = 0.6,
            lambda_std_w_rest: float = 0.5,
            lambda_w_norm: float = 0.6,
            lambda_w_rest_norm: float = 0.3
    ):
        super().__init__()
        self.lambda_std_w = lambda_std_w
        self.lambda_std_w_rest = lambda_std_w_rest
        self.lambda_w_norm = lambda_w_norm
        self.lambda_w_rest_norm = lambda_w_rest_norm

    def forward(self, node, scene, gs):
        if (not node.template.use_voxel_deformer) or (gs is None):
            return 0

        w_std = node.template.voxel_deformer.get_tv("dc")
        w_rest_std = node.template.voxel_deformer.get_tv("rest")
        w_norm = node.template.voxel_deformer.get_mag("dc")
        w_rest_norm = node.template.voxel_deformer.get_mag("rest")

        return self.lambda_std_w * w_std + \
            self.lambda_std_w_rest * w_rest_std + \
            self.lambda_w_norm * w_norm + \
            self.lambda_w_rest_norm * w_rest_norm


class KNNRegLoss(torch.nn.Module):
    def __init__(
            self,
            x: float = 0,
            q: float = 1,
            s: float = 1,
            d: float = 1,
            albedo: float = 1,
            sh: float = 1
    ):
        super().__init__()
        self.weights = dict(
            positions=x,
            rotation=q,
            scale=s,
            density=d,
            features_albedo=albedo,
            features_specular=sh
        )

    @staticmethod
    def compute_reg(node: SMPLSceneNode, param, inst_mask, nn_ind):
        num_verts = node.template.num_vertices
        p = node.model.params[param]
        p = p.view(-1, num_verts, p.shape[-1])[inst_mask]
        p = node.model.param_act(param)(p)
        return knn_gather(p, nn_ind).std(dim=2).mean()

    def forward(self, node, scene, gs):
        if gs is None:
            return 0

        mask = gs.cur_ids
        nn_ind_cur = node.nn_ind[mask]
        loss = 0

        for param, weight in self.weights.items():
            if weight > 0:
                loss += weight * self.compute_reg(node, param, mask, nn_ind_cur)
        return loss
