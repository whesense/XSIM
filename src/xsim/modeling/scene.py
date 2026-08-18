from typing import Optional

import torch
from xsimgs.cameras import RSCamera
from xsimgs.structures import OrientedBoxes

from xsim.data.loaders import SensorType
from xsim.data.reconstruction.dataset import (
    SensorImage, SensorBatch, SceneReconstructionDataset
)
from xsim.structures.time_range import TimeRange


class RenderResult:
    color: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    info: dict

    def __init__(
            self,
            color: Optional[torch.Tensor] = None,
            alpha: Optional[torch.Tensor] = None,
            depth: Optional[torch.Tensor] = None,
            info: Optional[dict] = None
    ):
        self.color = color
        self.alpha = alpha
        self.depth = depth
        self.info = info or {}

    @property
    def sky_map(self) -> Optional[torch.Tensor]:
        """Rendered per-pixel sky probability (the splatted sky_logit channel),
        set by the render node for camera renders when the models compose the
        SkyLogit mixin. Exposed as an attribute so RenderLoss subclasses can
        read it via their ``render_key``."""
        return self.info.get('sky_map')


class RenderInfo:
    sensor_type: SensorType
    sensor_id: int
    sample_idx: Optional[int]
    camera: RSCamera
    render_tag: str

    width: int
    height: int
    params: dict

    result: RenderResult

    @staticmethod
    def from_sensor_image(
            sensor_image: SensorImage
    ):
        return RenderInfo(
            sensor_type=sensor_image.sensor_type,
            sensor_id=sensor_image.sensor_id,
            sample_idx=sensor_image.sample_idx,
            camera=sensor_image.camera,
            width=sensor_image.image.shape[1],
            height=sensor_image.image.shape[0]
        )

    def __init__(
            self,
            sensor_type: SensorType,
            sensor_id: int,
            camera: RSCamera,
            width: int,
            height: int,

            sample_idx: Optional[int] = None,
            params: Optional[dict] = None,
            render_tag: Optional[str] = None,
            result: Optional[RenderResult] = None
    ):
        self.sensor_type = sensor_type
        self.sensor_id = sensor_id
        # Which sample of that sensor this render reproduces. Per-sensor
        # appearance models key on it; None when the render is not replaying a
        # recorded sample.
        self.sample_idx = sample_idx
        self.camera = camera
        self.render_tag = render_tag or ''

        self.width = width
        self.height = height
        self.params = params or {}

        self.result = result or RenderResult()


class Scene:
    renders: list[RenderInfo]
    batch: SensorBatch

    time_bounds: TimeRange
    renderables: list
    objects: OrientedBoxes

    world_time: Optional[float] = None
    container = None

    render_data: dict

    @staticmethod
    def from_sensor_batch(
            dataset: SceneReconstructionDataset,
            batch: SensorBatch
    ):
        return Scene(
            time_bounds=dataset.time_range,
            batch=batch,
            renders=[RenderInfo.from_sensor_image(v) for v in batch],
        )

    def __init__(
            self,
            time_bounds: TimeRange,
            renders: list[RenderInfo] = None,
            batch: list = None,

            objects: OrientedBoxes = None,
            renderables: list = None,

            world_time: float = None,
    ):
        self.renders = renders or []
        self.batch = batch or []

        self.time_bounds = time_bounds
        self.renderables = renderables or []
        self.objects = objects or None

        self.world_time = world_time
        self.container = None
        self.render_data = {}
