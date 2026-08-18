from typing import Literal
import torch
import torchvision as tv
import torchvision.transforms.functional as tvf

from xsim.utils.image_utils import interpolate, grid_sample

from .sensor_loader import SensorLoader, BasicSensorLoadingDataset
from .sensor_data import SensorData
from .sensor_iterator import load_images, all_sensor_indices, sensor_load_iterator, \
    images_from_iterator


DownsampleMode = Literal['nearest'] | Literal['bilinear'] | Literal['bicubic']


def downsample_size(
        h: int,
        w: int,
        downsample_factor: float,
        ensure_divisible: int = 1
) -> tuple[int, int]:
    new_h = int(round(h / (downsample_factor * ensure_divisible))) * ensure_divisible
    new_w = int(round(w / (downsample_factor * ensure_divisible))) * ensure_divisible
    return new_h, new_w


def resize_mask(
        mask: torch.Tensor,
        downsample_factor: float,
        downsample_mode: DownsampleMode = 'nearest',
        ensure_divisible: int = 1
):
    h, w = mask.shape[-2:]
    target_size = downsample_size(h, w, downsample_factor, ensure_divisible)
    return interpolate(mask, target_size, mode=downsample_mode)



class PathBasedImageLoader(SensorLoader):
    def __init__(self, path_fmt: str):
        self.path_fmt = path_fmt

    def format_path(self, sensor_id: int, sample_idx: int):
        return self.path_fmt.format(sensor_id=sensor_id, sample_idx=sample_idx)

    def __call__(self, sensor_id: int, sample_idx: int):
        return tv.io.read_image(self.format_path(sensor_id, sample_idx))



class ImageLoadingDataset(BasicSensorLoadingDataset):
    def __init__(
            self,
            loader: SensorLoader,
            indices,
            downsample_factor: float,
            downsample_mode: DownsampleMode = 'bicubic',
            ensure_divisible: int = 1,
            img_remap: dict[int, torch.Tensor] | None = None
    ):
        super().__init__(loader=loader, indices=indices)
        self.downsample_factor = downsample_factor
        self.downsample_mode = downsample_mode
        self.ensure_divisible = ensure_divisible
        self.img_remap = img_remap

        match downsample_mode:
            case 'nearest':
                self.tvf_downsample_mode = tvf.InterpolationMode.NEAREST
            case 'bilinear':
                self.tvf_downsample_mode = tvf.InterpolationMode.BILINEAR
            case 'bicubic':
                self.tvf_downsample_mode = tvf.InterpolationMode.BICUBIC
            case _:
                raise NotImplementedError(
                    'Unknown downsample mode: {}'.format(downsample_mode))

    def load_image(self, sensor_id: int, sample_idx: int):
        key, img = super().load_image(sensor_id, sample_idx)
        _, h, w = img.shape
        if self.downsample_factor != 1:
            img = tvf.resize(
                img, size=downsample_size(h, w, self.downsample_factor, ),
                interpolation=self.tvf_downsample_mode,
            )
        if self.img_remap is not None and sensor_id in self.img_remap:
            img = grid_sample(
                img, self.img_remap[sensor_id],
                channels_first=True,
                mode=self.downsample_mode
            )
        img = img.permute(1, 2, 0).contiguous()
        return key, img


class SegmentationLoadingDataset(ImageLoadingDataset):
    def load_image(self, sensor_id: int, sample_idx: int):
        key, img = super().load_image(sensor_id, sample_idx)
        return key, img.argmax(dim=-1).byte()


def load_camera_images_iterator(
        sensor_type: SensorData,
        num_workers: int,
        downsample_factor: float = 1,
        downsample_mode: DownsampleMode = 'bicubic',
        ensure_divisible: int = 1,
        img_remap: dict[int, torch.Tensor] | None = None,
        target_device: str = 'cpu'
):
    return sensor_load_iterator(
        ImageLoadingDataset(
            sensor_type.get_loader(),
            indices=all_sensor_indices(sensor_type),
            downsample_factor=downsample_factor,
            downsample_mode=downsample_mode,
            ensure_divisible=ensure_divisible,
            img_remap=img_remap
        ),
        num_workers=num_workers,
        target_device=target_device,
    )


def load_camera_images(
        sensor_type: SensorData,
        num_workers: int,
        downsample_factor: float = 1,
        downsample_mode: DownsampleMode = 'bicubic',
        ensure_divisible: int = 1,
        img_remap: dict[int, torch.Tensor] | None = None,
        target_device: str = 'cpu',
        tqdm_desc: str = 'Loading images'
):
    return images_from_iterator(load_camera_images_iterator(
        sensor_type=sensor_type,
        num_workers=num_workers,
        downsample_factor=downsample_factor,
        downsample_mode=downsample_mode,
        ensure_divisible=ensure_divisible,
        img_remap=img_remap,
        target_device=target_device
    ), total_images=sensor_type.total_num_images, tqdm_desc=tqdm_desc)


def load_segmentation_masks(
        cameras: SensorData,
        seg_loader: SensorLoader,
        num_workers: int,
        downsample_factor: float = 1,
        downsample_mode: DownsampleMode = 'bicubic',
        ensure_divisible: int = 1,
        img_remap: dict[int, torch.Tensor] | None = None,
        target_device: str = 'cpu',
        tqdm_desc: str = 'Loading images'
):
    return load_images(
        SegmentationLoadingDataset(
            seg_loader,
            indices=all_sensor_indices(cameras),
            downsample_factor=downsample_factor,
            downsample_mode=downsample_mode,
            ensure_divisible=ensure_divisible,
            img_remap=img_remap
        ),
        num_workers=num_workers,
        target_device=target_device,
        tqdm_desc=tqdm_desc
    )
