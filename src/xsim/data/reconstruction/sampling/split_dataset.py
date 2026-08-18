from dataclasses import dataclass, field

import numpy as np

from ..dataset import SceneReconstructionDataset
from .subset import Subset

@dataclass
class SplitConfig:
    test_ratio: float = 0.1
    test_selection_method: str = 'sequential'
    test_camera_ids: list[int] = field(default_factory=list)
    test_lidar_ids: list[int] = field(default_factory=list)
    random_seed: int = 42


def select_ids(ids, cfg):
    num_test_ids = int(np.round(len(ids) * cfg.test_ratio))
    test_ids = []

    if cfg.test_ratio > 0:
        match cfg.test_selection_method:
            case 'random':
                test_ids = np.random.default_rng(seed=cfg.random_seed).choice(
                    ids, size=(num_test_ids,), replace=False, shuffle=False
                ).tolist()
            case 'sequential':
                stride = int(np.round(1.0 / cfg.test_ratio))
                test_ids = ids[stride::stride]
            case 'middle':
                offset = int(np.round((len(ids) - num_test_ids) / 2))
                test_ids = ids[offset:offset + num_test_ids]
            case 'start':
                test_ids = ids[:num_test_ids]
            case 'end':
                test_ids = ids[-num_test_ids:]

    train_ids = list(sorted(set(ids).difference(test_ids)))
    return train_ids, test_ids


def split_dataset(
        dataset: SceneReconstructionDataset,
        cfg: SplitConfig = SplitConfig()
):
    train_camera_ids, test_camera_ids = {}, {}
    train_lidar_ids, test_lidar_ids = {}, {}

    for camera_id in dataset.camera_indices:
        num_images = dataset.scene.cameras.num_images(camera_id)
        if camera_id in cfg.test_camera_ids:
            train_ids, test_ids = [], list(range(num_images))
        else:
            train_ids, test_ids = select_ids(list(range(num_images)), cfg)
        train_camera_ids[camera_id] = train_ids
        test_camera_ids[camera_id] = test_ids

    for lidar_id in dataset.lidar_indices:
        num_images = dataset.scene.lidar.num_images(lidar_id)
        if lidar_id in cfg.test_lidar_ids:
            train_ids, test_ids = [], list(range(num_images))
        else:
            train_ids, test_ids = select_ids(list(range(num_images)), cfg)
        train_lidar_ids[lidar_id] = train_ids
        test_lidar_ids[lidar_id] = test_ids

    train = Subset(dataset, train_camera_ids, train_lidar_ids, split='train')
    test = Subset(dataset, test_camera_ids, test_lidar_ids, split='test')

    return train, test
