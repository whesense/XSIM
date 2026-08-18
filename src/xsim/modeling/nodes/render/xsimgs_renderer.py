from typing import Optional

import torch

from xsim.data import SceneReconstructionDataset

from xsimgs.structures import ROIBox3D
from xsimgs.rendering import ut_gs_render

from xsim.data.loaders import SensorType
from xsim.modeling.scene import Scene
from xsim.modeling.gaussian import GaussianField as GF, LidarDensity

from ..common import LossesType
from .base_render_node import BaseRenderNode


class XSIMGSRenderNode(BaseRenderNode):
    @staticmethod
    def create(
            sim_ds: SceneReconstructionDataset,
            init: dict,
            render_features: dict[SensorType, bool] | bool = True,
            convert_depth: bool = True,
            expected_depth: bool = False,
            rays_from_sweep: bool = True,
            camera_depth_max_density: bool = True,
            use_phm: bool = False,
            losses: LossesType = None
    ):
        return XSIMGSRenderNode(
            roi=sim_ds.roi,
            render_features=render_features,
            convert_depth=convert_depth,
            expected_depth=expected_depth,
            rays_from_sweep=rays_from_sweep,
            camera_depth_max_density=camera_depth_max_density,
            losses=losses or [],
        )

    def __init__(
            self,
            roi: ROIBox3D,
            render_features: dict[SensorType, bool] | bool = True,
            convert_depth: bool = True,
            expected_depth: bool = False,
            rays_from_sweep: bool = True,
            camera_depth_max_density: bool = True,
            use_phm: bool = False,
            losses: LossesType = None,
    ):
        super().__init__(
            roi=roi, render_features=render_features,
            convert_depth=convert_depth, expected_depth=expected_depth,
            rays_from_sweep=rays_from_sweep, losses=losses,
        )
        self.use_phm = use_phm
        self.camera_depth_max_density = camera_depth_max_density

        self.tile_sizes = {
            SensorType.CAMERA: (16, 16),
            SensorType.LIDAR: (32, 8)
        }

    def rasterize(self, render, points, data, colors, ray_o, ray_d, ray_time_delta,
                  opacity=None):
        tile_size_x, tile_size_y = self.tile_sizes[render.sensor_type]

        if opacity is None:
            opacity = self.sensor_density(data, render.sensor_type)

        return ut_gs_render(
            camera=render.camera,
            points=points,
            scale=data[GF.scale],
            rotation=data[GF.rotation],
            opacity=opacity,
            width=render.width,
            height=render.height,
            colors=colors,
            points_velocity=data.get(GF.velocity),
            packed_particle_mask=data.get(GF.mask),
            tile_size_x=tile_size_x,
            tile_size_y=tile_size_y,
            ray_o=ray_o,
            ray_d=ray_d,
            ray_time_delta=ray_time_delta,
            use_phm=self.use_phm,
        )

    def depth_density(self, data: dict, sensor_type: SensorType) -> Optional[torch.Tensor]:
        if self.training:
            return None

        if not (self.camera_depth_max_density and sensor_type == SensorType.CAMERA):
            return None

        lidar_density = data.get(LidarDensity.field_name)
        if lidar_density is None:
            return None

        return torch.maximum(data[GF.density], lidar_density)

    def forward(self, scene: Scene):
        container = scene.container
        data = container.data

        positions = data[GF.positions]
        velocity = data.get(GF.velocity)
        features = data.get(GF.features)

        for i, render in enumerate(scene.renders):
            camera = render.camera

            # advance positions from scene.world_time to this camera's pose_time
            points = positions
            if velocity is not None and scene.world_time is not None:
                pose_time = camera.world_se3_camera.pose_time.reshape(-1)[0]
                points = positions + velocity * (pose_time - scene.world_time)

            colors = (
                features
                if self.should_render_features(render.sensor_type)
                else None
            )

            # splat the per-particle sky probability as one extra color channel,
            # split back out of the raster below
            sky = self.sky_channel(data, render.sensor_type)
            if sky is not None:
                colors = sky if colors is None else torch.cat([colors, sky], dim=-1)

            # precomputed rays (cameras always; lidar when rays_from_sweep) --
            # None lets ut_gs_render fall back to the camera model
            ray_o, ray_d, ray_time_delta = self.render_rays(render, scene, i)

            rgb, depth, alpha, meta = self.rasterize(
                render, points, data, colors, ray_o, ray_d, ray_time_delta
            )

            # Second, geometry-only pass: same particles and rays, opacity taken
            # from the max of the two densities. Its own alpha comes with it, so
            # the expected-depth division stays consistent with this depth.
            depth_alpha = alpha
            depth_opacity = self.depth_density(data, render.sensor_type)
            if depth_opacity is not None:
                _, depth, depth_alpha, _ = self.rasterize(
                    render, points, data, None, ray_o, ray_d, ray_time_delta,
                    opacity=depth_opacity,
                )

            sky_map = None
            if sky is not None:
                sky_map = rgb[..., -1]
                rgb = rgb[..., :-1]

            self.finalize_render(render, rgb, depth, alpha, meta,
                                 ray_o=ray_o, ray_d=ray_d, sky_map=sky_map,
                                 depth_alpha=depth_alpha)
