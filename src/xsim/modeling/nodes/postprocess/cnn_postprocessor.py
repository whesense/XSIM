# Contains modified code from SplatAD (https://github.com/carlinds/splatad) /
# neurad-studio (https://github.com/georghess/neurad-studio, Apache-2.0,
# Copyright 2024 the authors of NeuRAD and contributors).
# See THIRD_PARTY_LICENSES.md.

import torch
import torch.nn as nn
from xsimgs.structures import ROIBox3D

from xsim.data import SceneReconstructionDataset, SensorType
from xsim.modeling.scene import Scene
from xsim.utils import get_torch_dtype

from ..common import (
    OptimizationConfig, ParamOptConfig, init_optimization, SceneNode)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        if in_dim != dim:
            self.res_branch = nn.Conv2d(in_dim, dim, kernel_size=1)
        else:
            self.res_branch = nn.Identity()
        self.main_branch = nn.Identity()
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_activation(self.res_branch(x) + self.main_branch(x))


class BasicBlock(ResidualBlock):
    def __init__(
            self,
            in_dim: int,
            dim: int,
            kernel_size: int,
            padding: int,
            use_bn: bool = False
    ):
        super().__init__(in_dim, dim)
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(dim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(dim) if use_bn else nn.Identity(),
        )


class RGBDecoderCNN(torch.nn.Module):
    def __init__(
        self,
        in_dim=6,
        out_dim=6,
        skip_dim=3,
        weight_init_scale=1e-2,
        hidden_dim=32,
        kernel_size=3,
        num_hidden_blocks=1,
    ):
        super().__init__()
        last_layer = torch.nn.Conv2d(hidden_dim, out_dim, 1)
        last_layer.weight.data *= weight_init_scale
        layers = [BasicBlock(
            in_dim, hidden_dim, kernel_size, padding=kernel_size // 2, use_bn=False
        )]
        for _ in range(num_hidden_blocks):
            layers.append(BasicBlock(
                hidden_dim, hidden_dim, kernel_size,
                padding=kernel_size // 2, use_bn=False
            ))
        layers.append(last_layer)
        self.net = torch.nn.Sequential(*layers)
        self.skip_dim = skip_dim
        self.out_dim = out_dim

    def forward(self, features: torch.Tensor, ray_dirs: torch.Tensor):
        albedo, spec = features.split(
            [self.skip_dim, features.shape[-1] - self.skip_dim], dim=-1)

        spec = torch.cat([spec, ray_dirs], dim=-1).unsqueeze(0)
        spec = spec.permute(0, 3, 1, 2)
        spec = self.net(spec)
        spec = spec.permute(0, 2, 3, 1)

        return albedo * (1 + spec[..., :3]) + spec[..., 3:]



class CNNPostprocessorNode(SceneNode):
    @classmethod
    def params_from_dict_and_cfg(cls, weights, node_cfg):
        params = SceneNode.params_from_dict_and_cfg(weights, node_cfg)
        params['num_cameras'] = weights['camera_embedding.weight'].shape[0]
        params['camera_names'] = weights['camera_names']
        return params

    @staticmethod
    def create(
            sim_ds: SceneReconstructionDataset,
            init,
            feature_dim: int = 13,
            appearance_dim: int = 8,
            dtype: str = 'float',
            opt_cfg: OptimizationConfig = None
    ):
        print('num_cameras:', len(sim_ds.camera_indices))
        print('feature_dim:', feature_dim, 'app_dim:', appearance_dim)

        return CNNPostprocessorNode(
            roi=sim_ds.roi,
            num_cameras=len(sim_ds.camera_indices),
            camera_names=sim_ds.camera_indices,
            feature_dim=feature_dim,
            appearance_dim=appearance_dim,
            dtype=dtype,
            opt_cfg=opt_cfg
        )

    def __init__(
            self,
            num_cameras: int,
            camera_names: list[int] | torch.Tensor,
            feature_dim: int = 13,
            appearance_dim: int = 8,
            hidden_dim: int = 32,
            kernel_size: int = 3,
            num_hidden_blocks: int = 1,
            dtype: str = 'float',
            roi: ROIBox3D = None,
            opt_cfg: OptimizationConfig = None
    ):
        super().__init__(roi=roi, losses=[])
        self.opt_cfg = opt_cfg
        self.rgb_decoder = RGBDecoderCNN(
            feature_dim + appearance_dim + 3,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_hidden_blocks=num_hidden_blocks,
        )
        self.dtype = get_torch_dtype(dtype)
        if self.dtype != torch.float32:
            self.rgb_decoder = self.rgb_decoder.to(dtype=self.dtype)

        self.camera_embedding = torch.nn.Embedding(num_cameras, appearance_dim)
        self.camera_names = torch.nn.Buffer(torch.as_tensor(camera_names).view(-1))

    def params(self):
        return {'rgb_decoder': self.rgb_decoder,
                'camera_embedding': self.camera_embedding}

    def init_optimization(self):
        opt_cfg = self.opt_cfg or OptimizationConfig(
            params={name: ParamOptConfig(lr=1e-3) for name in self.params()})
        self.opt, self.schedulers = init_optimization(
            opt_cfg, self.params(), self.resolve_lr_scale_factor,
            self.__class__.__name__)

    def camera_idx(self, render):
        cam_idx = (self.camera_names == render.sensor_id).nonzero().view(-1)
        cam_idx = cam_idx.to(self.camera_embedding.weight.device)
        return cam_idx

    def forward(self, scene: Scene):
        for render in scene.renders:
            if render.sensor_type == SensorType.CAMERA:
                if render.camera.camera_name is None:
                    render.result.color = None
                    continue

                cam_embedding = self.camera_embedding(self.camera_idx(render))
                cam_embedding = cam_embedding.view(1, 1, -1)
                cam_embedding = cam_embedding.expand(render.height, render.width, -1)
                features = torch.cat([render.result.color, cam_embedding], dim=-1)
                # world-space normalized directions cached by the render node
                ray_dirs = render.result.info['ray_directions']
                with torch.autocast(device_type='cuda', dtype=self.dtype):
                    color = self.rgb_decoder.forward(features, ray_dirs)
                render.result.color = color[0].float()

    def optimize_inference(self):
        self.rgb_decoder = torch.compile(self.rgb_decoder, fullgraph=True)
