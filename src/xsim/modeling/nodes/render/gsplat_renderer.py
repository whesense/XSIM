import torch

from xsimgs.structures import ROIBox3D
from xsimgs.cameras import PerspectiveCamera
from xsimgs.cameras.base.shutter import RollingShutterDirection

from xsim.data import SceneReconstructionDataset
from xsim.data.loaders import SensorType
from xsim.modeling.scene import Scene
from xsim.modeling.gaussian import GaussianField as GF

from ..common import LossesType
from .base_render_node import BaseRenderNode


class GSplatRenderNode(BaseRenderNode):
    @staticmethod
    def create(
            sim_ds: SceneReconstructionDataset,
            init: dict,
            render_features: bool = True,
            convert_depth: bool = False,
            losses: LossesType = None,
    ):
        return GSplatRenderNode(
            roi=sim_ds.roi,
            render_features=render_features,
            convert_depth=convert_depth,
            losses=losses or [],
        )

    @staticmethod
    def shutter_type_mapping():
        if not hasattr(GSplatRenderNode, 'SHUTTER_TYPES'):
            from gsplat.cuda._wrapper import RollingShutterType

            # xsimgs rolling-shutter readout direction -> gsplat rolling-shutter type.
            GSplatRenderNode.SHUTTER_TYPES = {
                RollingShutterDirection.GLOBAL_SHUTTER: RollingShutterType.GLOBAL,
                RollingShutterDirection.LEFT_TO_RIGHT: RollingShutterType.ROLLING_LEFT_TO_RIGHT,
                RollingShutterDirection.RIGHT_TO_LEFT: RollingShutterType.ROLLING_RIGHT_TO_LEFT,
                RollingShutterDirection.TOP_TO_BOTTOM: RollingShutterType.ROLLING_TOP_TO_BOTTOM,
                RollingShutterDirection.BOTTOM_TO_TOP: RollingShutterType.ROLLING_BOTTOM_TO_TOP,
            }

        return GSplatRenderNode.SHUTTER_TYPES

    def __init__(
            self,
            roi: ROIBox3D,
            render_features: bool = True,
            convert_depth: bool = False,
            losses: LossesType = None,
    ):
        try:
            import gsplat  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "GSplatRenderNode requires the optional 'gsplat' package, which is "
                "not installed. Install it to use the gsplat renderer."
            ) from e

        super().__init__(
            roi=roi, render_features=render_features,
            convert_depth=convert_depth, losses=losses,
        )

    @staticmethod
    def viewmats(camera: PerspectiveCamera):
        """world-to-camera matrices at the shutter start and end times."""
        start, end = camera.pose_range  # camera-to-world at (t_min, t_max)
        return torch.linalg.inv(start), torch.linalg.inv(end)

    def render(self, camera: PerspectiveCamera, data: dict, width: int, height: int):
        from gsplat import rasterization

        points = data[GF.positions]
        colors = (data.get(GF.features)
                  if self.should_render_features(SensorType.CAMERA) else None)
        mask = data.get(GF.mask)

        opacity = data[GF.density].reshape(-1)
        scale = data[GF.scale]
        rotation = data[GF.rotation]

        num_particles = points.shape[0]
        indices = None
        if mask is not None:
            indices = mask.nonzero().reshape(-1)
            points = points[indices]
            opacity = opacity[indices]
            scale = scale[indices]
            rotation = rotation[indices]
            if colors is not None:
                colors = colors[indices]

        viewmat, viewmat_rs = self.viewmats(camera)
        K = camera.k_matrix(width, height)
        rolling_shutter = self.shutter_type_mapping()[camera.shutter.direction]

        if colors is None:
            # geometry only: still need a colors tensor, render expected depth.
            colors = torch.zeros_like(points[..., :1])
            render_mode = "ED"
        else:
            render_mode = "RGB+ED"

        out_colors, alpha, gsplat_meta = rasterization(
            means=points,
            quats=rotation,
            scales=scale,
            opacities=opacity,
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            packed=False,
            camera_model="pinhole",
            render_mode=render_mode,
            with_ut=True,
            with_eval3d=True,
            rolling_shutter=rolling_shutter,
            viewmats_rs=viewmat_rs[None],
        )

        # gsplat radii are [1, N, 2] int32 over the (masked) particles; scatter
        # back into a full-size float32 [num_particles, 2] tensor.
        radii = gsplat_meta["radii"].reshape(-1, 2).float()
        if indices is not None:
            full = points.new_zeros((num_particles, 2))
            full[indices] = radii
            radii = full
        meta = dict(radii=radii, width=width, height=height)

        out_colors = out_colors[0]
        alpha = alpha[0]
        if render_mode == "ED":
            return None, out_colors, alpha, meta
        return out_colors[..., :3], out_colors[..., 3:4], alpha, meta

    def forward(self, scene: Scene):
        container = scene.container
        data = container.data

        positions = data[GF.positions]
        velocity = data.get(GF.velocity)

        for render in scene.renders:
            # print('gsplat render:', render.sensor_type)
            if render.sensor_type.value != SensorType.CAMERA.value:
                continue
            # print('inside')

            camera = render.camera

            # advance positions from scene.world_time to this camera's pose_time
            points = positions
            if velocity is not None and scene.world_time is not None:
                pose_time = camera.world_se3_camera.pose_time.reshape(-1)[0]
                points = positions + velocity * (pose_time - scene.world_time)

            frame = dict(data)
            frame[GF.positions] = points

            color, depth, alpha, meta = self.render(
                camera=camera,
                data=frame,
                width=render.width,
                height=render.height,
            )
            # print('color:', color)

            self.finalize_render(render, color, depth, alpha, meta)
