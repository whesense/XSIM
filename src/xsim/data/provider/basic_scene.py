import torch

from xsimgs.structures import ROIBox3D, OrientedBoxes

from xsim.data.utils.smpl import SMPLData
from xsim.structures import Sweep
from xsim.data.loaders.sensor_data import SensorData, SensorType
from xsim.data.loaders.sensor_loader import SensorLoader


def roi_from_points(points: torch.Tensor, quantile: float):
    k = int(len(points) * quantile)
    vmin = torch.kthvalue(points, k, dim=0).values.cpu()
    vmax = torch.kthvalue(points, len(points) - k, dim=0).values.cpu()
    return ROIBox3D(vmin, vmax)



class DatasetScene:
    sensors: dict[SensorType, SensorData]

    def __init__(self):
        self.sensors = {}

    @property
    def cameras(self) -> SensorData:
        return self.sensors[SensorType.CAMERA]

    @property
    def lidar(self) -> SensorData:
        return self.sensors[SensorType.LIDAR]

    @property
    def scene_alias(self) -> str:
        """Short, stable name for this scene, used to group runs by subject.

        Experiment directories are named after it, so it has to be filesystem
        safe and mean the same thing across runs. The default is the zero-padded
        scene index; a provider whose index alone does not identify a scene --
        one split across training and validation sets, say -- overrides it.
        """
        scene_idx = getattr(self, 'scene_idx', None)
        if isinstance(scene_idx, int):
            return '{:03d}'.format(scene_idx)

        return str(scene_idx) if scene_idx is not None else type(self).__name__

    @property
    def num_objects(self) -> int:
        raise NotImplementedError()

    @property
    def vehicle_ids(self) -> list[int]:
        raise NotImplementedError()

    @property
    def human_ids(self) -> list[int]:
        raise NotImplementedError()

    def get_boxes(self, object_id: int) -> OrientedBoxes:
        raise NotImplementedError()

    def get_smpl_data(self, object_id: int) -> SMPLData | None:
        raise NotImplementedError()

    # Other
    def get_roi(self, sweeps: list[Sweep], quantile: float = 0.01):
        points = torch.cat([sweep.masked.xyz for sweep in sweeps], dim=0).float()
        return roi_from_points(points, quantile=quantile)

    def camera_lidar_synchronization(self):
        # fully synchronized setup, all sensors have same number of samples
        lidar_closest_cameras = {
            lidar_id: [
                [(camera_id, sample_idx) for camera_id in self.cameras.indices]
                for sample_idx in range(self.lidar.num_images(lidar_id))
            ] for lidar_id in self.lidar.indices
        }
        cameras_closest_lidar = {
            camera_id: [
                [(lidar_id, sample_idx) for lidar_id in self.lidar.indices]
                for sample_idx in range(self.cameras.num_images(camera_id))
            ] for camera_id in self.cameras.indices
        }
        lidar_synchronized_groups = [tuple(self.lidar.indices)]

        return lidar_closest_cameras, cameras_closest_lidar, lidar_synchronized_groups

    def get_segmentation_loader(self) -> SensorLoader:
        raise NotImplementedError()
