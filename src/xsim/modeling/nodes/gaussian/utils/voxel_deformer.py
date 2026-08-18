# Contains modified code from OmniRe / drivestudio
# (https://github.com/ziyc/drivestudio, MIT, Copyright (c) 2024 Ziyu Chen).
# See THIRD_PARTY_LICENSES.md.

import torch
import torch.nn as nn
import torch.nn.functional as F

from chamferdist.chamfer import knn_points, knn_gather


class VoxelDeformer(nn.Module):
    def __init__(
        self,
        vtx,
        vtx_features,
        resolution_dhw=[8, 32, 32],
        short_dim_dhw=0,  # 0 is d, corresponding to z
        long_dim_dhw=1,
        is_resume=False
    ) -> None:
        super().__init__()
        # vtx B,N,3, vtx_features: B,N,J
        # d-z h-y w-x; human is facing z; dog is facing x, z is upward, should compress on y
        B = vtx.shape[0]
        assert vtx.shape[0] == vtx_features.shape[0], "Batch size mismatch"

        # * Prepare Grid
        self.resolution_dhw = resolution_dhw
        device = vtx.device
        d, h, w = self.resolution_dhw

        self.register_buffer(
            "ratio",
            torch.Tensor(
                [self.resolution_dhw[long_dim_dhw] / self.resolution_dhw[short_dim_dhw]]
            ).squeeze(),
        )
        self.ratio_dim = -1 - short_dim_dhw
        x_range = (
            (torch.linspace(-1, 1, steps=w, device=device))
            .view(1, 1, 1, w)
            .expand(1, d, h, w)
        )
        y_range = (
            (torch.linspace(-1, 1, steps=h, device=device))
            .view(1, 1, h, 1)
            .expand(1, d, h, w)
        )
        z_range = (
            (torch.linspace(-1, 1, steps=d, device=device))
            .view(1, d, 1, 1)
            .expand(1, d, h, w)
        )
        grid = (
            torch.cat((x_range, y_range, z_range), dim=0)
            .reshape(1, 3, -1)
            .permute(0, 2, 1)
        )
        grid = grid.expand(B, -1, -1)

        gt_bbox_min = (vtx.min(dim=1).values).to(device)
        gt_bbox_max = (vtx.max(dim=1).values).to(device)
        offset = (gt_bbox_min + gt_bbox_max) * 0.5
        self.register_buffer(
            "global_scale", torch.Tensor([1.2]).squeeze()
        )  # from Fast-SNARF
        scale = (
            (gt_bbox_max - gt_bbox_min).max(dim=-1).values / 2 * self.global_scale
        ).unsqueeze(-1)

        corner = torch.ones_like(offset) * scale
        corner[:, self.ratio_dim] /= self.ratio
        min_vert = (offset - corner).reshape(-1, 1, 3)
        max_vert = (offset + corner).reshape(-1, 1, 3)
        self.bbox = torch.cat([min_vert, max_vert], dim=1)

        self.register_buffer("scale", scale.unsqueeze(1)) # [B, 1, 1]
        self.register_buffer("offset", offset.unsqueeze(1)) # [B, 1, 3]

        grid_denorm = self.denormalize(
            grid
        )  # grid_denorm is in the same scale as the canonical body

        if not is_resume:
            weights = (
                self._query_weights_smpl(
                    grid_denorm,
                    smpl_verts=vtx.detach().clone(),
                    smpl_weights=vtx_features.detach().clone(),
                )
                .detach()
                .clone()
            )
        else:
            # random initialization
            weights = torch.randn(
                B, vtx_features.shape[-1], *resolution_dhw
            ).to(device)

        self.register_buffer("lbs_voxel_base", weights.detach())
        self.register_buffer("grid_denorm", grid_denorm)

        self.num_bones = vtx_features.shape[-1]

    def enable_voxel_correction(self):
        voxel_w_correction = torch.zeros_like(self.lbs_voxel_base).contiguous()
        self.voxel_w_correction = nn.Parameter(voxel_w_correction)

    def enable_additional_correction(self, additional_channels, std=1e-4):
        additional_correction = (
            torch.ones(
                self.lbs_voxel_base.shape[0],
                additional_channels,
                *self.lbs_voxel_base.shape[2:]
            )
            * std
        )
        self.additional_correction = nn.Parameter(additional_correction)

    @property
    def get_voxel_weight(self):
        w = self.lbs_voxel_base
        if hasattr(self, "voxel_w_correction"):
            w = w + self.voxel_w_correction
        if hasattr(self, "additional_correction"):
            w = torch.cat([w, self.additional_correction], dim=1)
        return w

    def get_tv(self, name="dc"):
        if name == "dc":
            if not hasattr(self, "voxel_w_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.voxel_w_correction
        elif name == "rest":
            if not hasattr(self, "additional_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.additional_correction
        tv_x = torch.abs(d[:, :, 1:, :, :] - d[:, :, :-1, :, :]).mean()
        tv_y = torch.abs(d[:, :, :, 1:, :] - d[:, :, :, :-1, :]).mean()
        tv_z = torch.abs(d[:, :, :, :, 1:] - d[:, :, :, :, :-1]).mean()
        return (tv_x + tv_y + tv_z) / 3.0

    def get_mag(self, name="dc"):
        if name == "dc":
            if not hasattr(self, "voxel_w_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.voxel_w_correction
        elif name == "rest":
            if not hasattr(self, "additional_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.additional_correction
        return torch.norm(d, dim=1).mean()

    def forward(self, xc, mode="bilinear"):
        # xs: [B, N, 3]
        shape = xc.shape  # ..., 3
        # xc = xc.reshape(1, -1, 3)
        w = F.grid_sample(
            self.get_voxel_weight,
            self.normalize(xc)[:, :, None, None],  # [B, N, 3, 1, 1] ?
            align_corners=True,
            mode=mode,
            padding_mode="border",
        )
        # print('w after grid sample:', w.shape)
        w = w.squeeze(3, 4)
        # print('w after squeeze:', w.shape)
        # w = w.permute(0, 2, 1)
        # print('w after permute:', w.shape)
        # w = w.reshape(*shape[:-1], -1)
        # print('w after reshape:', w.shape)
        # # * the w may have more channels
        return w

    def normalize(self, x):
        x_normalized = x.clone()
        x_normalized -= self.offset
        x_normalized /= self.scale
        x_normalized[..., self.ratio_dim] *= self.ratio
        return x_normalized

    def denormalize(self, x):
        x_denormalized = x.clone()
        x_denormalized[..., self.ratio_dim] /= self.ratio
        x_denormalized *= self.scale
        x_denormalized += self.offset
        return x_denormalized

    @torch.no_grad()
    def get_weights_and_dist(self, x, smpl_verts, smpl_weights):
        dist, idx, _ = knn_points(x, smpl_verts, K=30)  # [B, N, 30]
        dist = dist.sqrt().clamp_(0.0001, 1.0)
        expanded_smpl_weights = smpl_weights.unsqueeze(2).expand(-1, -1, idx.shape[2], -1)
        # expanded_smpl_weights: [B, N, 30, J]
        weights = expanded_smpl_weights.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, -1, expanded_smpl_weights.shape[-1]))
        # weights: [B, N, 30, J]
        return weights, 1.0 / dist

    @torch.no_grad()
    def get_weights(self, x, smpl_verts, smpl_weights):
        # Reduce cuda peak memory usage by splitting operation in batches
        x = x.cuda()
        smpl_verts = smpl_verts.cuda()
        smpl_weights = smpl_weights.cuda()

        B = x.shape[0]
        results = []

        for i in range(B):
            weights, ws = self.get_weights_and_dist(
                x[i:i+1], smpl_verts[i:i+1], smpl_weights[i:i+1]
            )
            ws /= ws.sum(-1, keepdim=True)
            weights *= ws[..., None]
            weights = weights.sum(-2)
            results.append(weights)

        return torch.cat(results, dim=0)

    # @torch.compile(dynamic=False)
    @torch.no_grad()
    def _query_weights_smpl(self, x, smpl_verts, smpl_weights):
        # adapted from https://github.com/jby1993/SelfReconCode/blob/main/model/Deformer.py
        weights = self.get_weights(x, smpl_verts, smpl_weights)

        b = x.shape[0]
        c = smpl_weights.shape[-1]
        d, h, w = self.resolution_dhw
        weights = weights.permute(0, 2, 1).reshape(b, c, d, h, w)
        for _ in range(30):
            mean = weights[:, :, 2:, 1:-1, 1:-1].clone()
            mean += weights[:, :, :-2, 1:-1, 1:-1]
            mean += weights[:, :, 1:-1, 2:, 1:-1]
            mean += weights[:, :, 1:-1, :-2, 1:-1]
            mean += weights[:, :, 1:-1, 1:-1, 2:]
            mean += weights[:, :, 1:-1, 1:-1, :-2]
            mean *= (0.3 / 6.0)

            weights[:, :, 1:-1, 1:-1, 1:-1] *= 0.7
            weights[:, :, 1:-1, 1:-1, 1:-1] += mean
            weights /= weights.sum(1, keepdim=True)
        return weights
