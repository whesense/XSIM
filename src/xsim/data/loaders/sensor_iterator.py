from collections import defaultdict
from typing import Optional

import torch

from xsim.data.loaders.sensor_data import SensorData
from xsim.data.loaders.sensor_loader import *
from xsim.utils import torch_move, progress_bar


def no_collate(lst):
    return lst[0]


def sensor_indices(sensor_type: SensorData, sensor_ids: list[int]):
    return sum([
        [(sensor_id, i) for i in range(sensor_type.num_images(sensor_id))]
        for sensor_id in sensor_ids
    ], start=[])


def all_sensor_indices(sensor_type: SensorData):
    return sensor_indices(sensor_type, sensor_ids=sensor_type.indices)



def use_non_blocking(target_device: str | torch.device):
    return torch.device(target_device).type == 'cuda'

def sensor_load_iterator(
        dataset: BasicSensorLoadingDataset,
        num_workers: int,
        target_device: str = 'cpu',
        infinite_iterator: bool = False,
        shuffle: bool = False,
):
    num_workers = num_workers if not dataset.loader.preloaded else 0
    persistent_workers = infinite_iterator and num_workers > 0
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=no_collate,
        in_order=True
    )
    loader_iter = iter(loader)
    non_blocking = use_non_blocking(target_device)

    while True:
        try:
            result = torch_move(
                next(loader_iter),
                device=target_device,
                non_blocking=non_blocking
            )
            yield result
        except StopIteration:
            if infinite_iterator:
                loader_iter = iter(loader)
            else:
                break


def multi_sensor_load_iterator(
        datasets: dict[str, BasicSensorLoadingDataset],
        indices: MultiLoaderIndices,
        num_workers: int,
        target_device: str = 'cpu',
        infinite_iterator: bool = False,
        shuffle: bool = False,
):
    preloaded_keys = [key for key, ds in datasets.items() if ds.loader.preloaded]
    all_preloaded = len(preloaded_keys) == len(datasets)
    num_workers = num_workers if not all_preloaded else 0
    persistent_workers = infinite_iterator and num_workers > 0

    loader = torch.utils.data.DataLoader(
        MultiLoaderDataset(
            {key: ds for key, ds in datasets.items() if key not in preloaded_keys},
            indices,
        ),
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=no_collate,
        pin_memory=True,
        in_order=True
    )
    loader_iter = iter(loader)
    non_blocking = use_non_blocking(target_device)


    while True:
        try:
            result = next(loader_iter)
            cur_indices = result['_indices']
            for dataset_key in preloaded_keys:
                result[dataset_key] = [
                    datasets[dataset_key].load_image(sensor_id, sample_idx)
                    for (sensor_id, sample_idx) in cur_indices[dataset_key]
                ]
            result.pop('_indices')
            result.pop('_index')
            result = torch_move(
                result,
                device=target_device,
                non_blocking=non_blocking
            )
            yield result
        except StopIteration:
            if infinite_iterator:
                loader_iter = iter(loader)
            else:
                break


def images_from_iterator(
        load_iter,
        total_images: int,
        tqdm_desc: Optional[str] = 'Loading images'
):
    images_dict = defaultdict(dict)

    with progress_bar(tqdm_desc, total=total_images) as pbar:
        for (sensor_id, sample_idx), img in iter(load_iter):
            images_dict[sensor_id][sample_idx] = img
            pbar.update()

    images = {}
    for sensor_id, imgs in images_dict.items():
        num_images = max(imgs.keys()) + 1
        images[sensor_id] = [None] * num_images
        for sample_idx, img in imgs.items():
            images[sensor_id][sample_idx] = img

    return images


def load_images(
        dataset: BasicSensorLoadingDataset,
        num_workers: int,
        target_device: str = 'cpu',
        tqdm_desc: str = 'Loading images'
):
    return images_from_iterator(sensor_load_iterator(
        dataset,
        num_workers=num_workers,
        target_device=target_device
    ), total_images=len(dataset), tqdm_desc=tqdm_desc)


def load_images_from_sensor_data(
        sensor_data: SensorData,
        num_workers: int,
        target_device: str = 'cpu',
        tqdm_desc: str = 'Loading images'
):
    return load_images(
        BasicSensorLoadingDataset(
            sensor_data.get_loader(),
            indices=all_sensor_indices(sensor_data),
        ),
        num_workers=num_workers,
        target_device=target_device,
        tqdm_desc=tqdm_desc
    )
