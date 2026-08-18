from typing import Optional

import torch
from xsimgs.structures import ROIBox3D
from xsimgs.cameras.torch import PerspectiveCameraTorch

from xsim.data.loaders import SensorType
from xsim.modeling.gaussian import GaussianField as GF, LidarDensity, SkyLogit

from ..common import SceneNode, LossesType


# Camera projection models whose rendered depth is planar (camera-plane) rather
# than euclidean ray length after conversion -- distorted-perspective/fisheye
# subclass PerspectiveCameraTorch; spherical/lidar do not.
PERSPECTIVE_CAMERAS = (PerspectiveCameraTorch,)


class BaseRenderNode(SceneNode):
    # Alpha under which a ray is treated as having accumulated nothing, so its
    # expected depth is reported as zero rather than as a ratio of two sums that
    # are both approaching it.
    min_expected_alpha: float = 1e-3

    # Rendered sky probability above which a camera pixel counts as sky and its
    # depth is masked out.
    sky_threshold: float = 0.5

    def __init__(
            self,
            roi: ROIBox3D,
            render_features: dict[SensorType, bool] | bool = True,
            convert_depth: bool = True,
            expected_depth: bool = False,
            rays_from_sweep: bool = True,
            losses: LossesType = None,
    ):
        super().__init__(roi=roi, losses=losses)
        # normalize to a {SensorType: bool} mapping, casting str keys to enum
        if isinstance(render_features, bool):
            render_features = {st: render_features for st in SensorType}
        self.render_features = {
            (SensorType[k.upper()] if isinstance(k, str) else k): v
            for k, v in render_features.items()
        }
        # When True, convert rendered euclidean ray-length depth to planar depth
        # for perspective-family cameras.
        self.convert_depth = convert_depth
        # When True, divide the accumulated depth by the accumulated alpha.
        self.expected_depth = expected_depth
        # When True, lidar renders take ray origins / directions / time delta
        # from the batch sweep instead of the (idealized) lidar camera model.
        self.rays_from_sweep = rays_from_sweep

    def render_rays(self, render, scene, index):
        if (self.rays_from_sweep and render.sensor_type == SensorType.LIDAR
                and index < len(scene.batch)):
            return self.sweep_rays(render, scene.batch[index].image)

        if render.sensor_type == SensorType.CAMERA:
            ray_o, ray_d = render.camera.camera_rays(
                width=render.width, height=render.height,
                return_origins=True, world_space=True, normalized_rays=True,
            )
            return ray_o, ray_d, None

        return None, None, None

    def sweep_rays(self, render, sweep):
        camera = render.camera
        ray_o, ray_d = camera.camera_rays(
            width=render.width, height=render.height,
            return_origins=True, world_space=True, normalized_rays=True,
        )
        # Not sweep.mask: a cell can be masked off (ground filtering) while
        # still holding a real measurement, and that measurement is usable here.
        measured = sweep.ray_dirs.norm(dim=-1, keepdim=True) > 0
        ray_o = torch.where(measured, sweep.origin, ray_o)
        ray_d = torch.where(measured, sweep.ray_dirs, ray_d)

        time_delta = None
        if sweep.time is not None:
            # Providers differ on whether per-pixel time carries a trailing 1
            # ([H, W] vs [H, W, 1]); pin both down before combining.
            shape = (render.height, render.width, 1)
            time = torch.where(
                measured,
                sweep.time.reshape(shape),
                camera.shutter.image_time(sweep.time).reshape(shape)
            )
            pose_time = camera.world_se3_camera.pose_time
            time_delta = (time - pose_time).reshape(render.height, render.width)

        return ray_o.contiguous(), ray_d.contiguous(), time_delta

    def should_render_features(self, sensor_type: SensorType) -> bool:
        return self.render_features.get(sensor_type, False)

    @staticmethod
    def sensor_density(data: dict, sensor_type: SensorType) -> torch.Tensor:
        if sensor_type == SensorType.LIDAR and LidarDensity.field_name in data:
            return data[LidarDensity.field_name]

        return data[GF.density]

    @staticmethod
    def sky_channel(data: dict, sensor_type: SensorType) -> Optional[torch.Tensor]:
        return data.get(SkyLogit.field_name) if sensor_type == SensorType.CAMERA else None

    def non_euclidean_depth(self, render) -> bool:
        return self.convert_depth and isinstance(render.camera, PERSPECTIVE_CAMERAS)

    def cache_rays(self, render, ray_o=None, ray_d=None):
        camera = render.camera
        info = render.result.info

        if ray_o is None or ray_d is None:
            ray_o, ray_d = camera.camera_rays(
                width=render.width, height=render.height,
                return_origins=True, world_space=True, normalized_rays=True,
            )

        info['ray_origins'] = ray_o
        info['ray_directions'] = ray_d

        if self.non_euclidean_depth(render):
            _, ray_dir_depth = camera.camera_rays(
                width=render.width, height=render.height,
                return_origins=False, world_space=True, normalized_rays=False,
            )
        else:
            ray_dir_depth = ray_d

        info['ray_dir_depth'] = ray_dir_depth

    def alpha_weighted_depth(self, depth, alpha):
        if depth is None or alpha is None or not self.expected_depth:
            return depth

        alpha = alpha.reshape(depth.shape)

        # Clamped inside the division as well: an inf in the branch torch.where
        # discards still propagates NaN through the backward pass.
        return torch.where(
            alpha > self.min_expected_alpha,
            depth / alpha.clamp(min=self.min_expected_alpha),
            torch.zeros_like(depth)
        )

    def planar_depth(self, depth, render):
        if depth is None or not self.non_euclidean_depth(render):
            return depth
        coeff = render.result.info['ray_dir_depth'].norm(dim=-1).reshape(depth.shape)
        return depth / coeff

    def finalize_render(self, render, rgb, depth, alpha, meta,
                        ray_o=None, ray_d=None, sky_map=None, depth_alpha=None):
        # Geometry-only renders (features off, e.g. lidar) come back with a
        # 0-channel color tensor; expose it as None so downstream color consumers
        # (env light, CNN, image loss) skip it via their `color is not None` guard.
        render.result.color = rgb if rgb is not None and rgb.shape[-1] > 0 else None
        render.result.alpha = alpha
        render.result.info = meta

        self.cache_rays(render, ray_o=ray_o, ray_d=ray_d)

        render.result.depth = self.planar_depth(
            self.alpha_weighted_depth(depth, alpha if depth_alpha is None else depth_alpha),
            render
        )

        if sky_map is not None:
            sky_map = sky_map.reshape(render.height, render.width)
            render.result.info['sky_map'] = sky_map

            if not self.training and render.result.depth is not None:
                cur_depth = render.result.depth
                render.result.depth = cur_depth * (
                    sky_map.reshape(cur_depth.shape) < self.sky_threshold
                )
