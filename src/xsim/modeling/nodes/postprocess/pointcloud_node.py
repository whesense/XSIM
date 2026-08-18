import torch
from xsimgs.structures import ROIBox3D

from xsim.data import SceneReconstructionDataset, SensorType
from xsim.modeling.scene import Scene

from ..common import SceneNode, LossesType


class PointCloudNode(SceneNode):
    @staticmethod
    def create(
            sim_ds: SceneReconstructionDataset,
            init,
            create_pointcloud: dict[SensorType, bool] | bool = True,
            eval_only: bool = False,
            points_in_sensor_frame: bool = False,
            losses: LossesType = None,
    ):
        return PointCloudNode(
            roi=sim_ds.roi,
            create_pointcloud=create_pointcloud,
            eval_only=eval_only,
            points_in_sensor_frame=points_in_sensor_frame,
            losses=losses or [],
        )

    def __init__(
            self,
            roi: ROIBox3D = None,
            create_pointcloud: dict[SensorType, bool] | bool = True,
            eval_only: bool = False,
            points_in_sensor_frame: bool = False,
            losses: LossesType = None,
    ):
        super().__init__(roi=roi, losses=losses)
        if isinstance(create_pointcloud, bool):
            create_pointcloud = {st: create_pointcloud for st in SensorType}
        self.create_pointcloud = {
            (SensorType[k.upper()] if isinstance(k, str) else k): v
            for k, v in create_pointcloud.items()
        }
        self.eval_only = eval_only
        self.points_in_sensor_frame = points_in_sensor_frame

    def enabled(self, sensor_type: SensorType) -> bool:
        return self.create_pointcloud.get(sensor_type, False)

    def forward(self, scene: Scene):
        if self.eval_only and self.training:
            return
        for i, render in enumerate(scene.renders):
            if not self.enabled(render.sensor_type):
                continue
            depth = render.result.depth
            if depth is None:
                continue
            depth = depth.reshape(render.height, render.width)
            info = render.result.info
            points = info['ray_origins'] + info['ray_dir_depth'] * depth.unsqueeze(-1)

            valid = depth > 1e-6
            if i < len(scene.batch) and scene.batch[i].gt_mask is not None:
                gt_mask = scene.batch[i].gt_mask.reshape(render.height, render.width)
                valid = valid & gt_mask.bool()
            points = points[valid]

            if self.points_in_sensor_frame:
                points = render.camera.world_se3_camera.pose.inv().transform(points)

            render.result.info['points'] = points
