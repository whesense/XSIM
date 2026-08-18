import torch


ImageIndices = list[tuple[int, int]]
MultiLoaderIndices = list[dict[str, ImageIndices]]


class SensorLoader:
    preloaded: bool = False

    def __call__(self, sensor_id: int, sample_idx: int):
        pass


class BasicSensorLoadingDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            loader: SensorLoader,
            indices
    ):
        self.loader = loader
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def load_image(self, sensor_id: int, sample_idx: int):
        return (sensor_id, sample_idx), self.loader(sensor_id, sample_idx)

    def __getitem__(self, idx: int):
        return self.load_image(*self.indices[idx])


class MultiLoaderDataset(BasicSensorLoadingDataset):
    def __init__(
            self,
            datasets: dict[str, BasicSensorLoadingDataset],
            indices: MultiLoaderIndices
    ):
        super().__init__(loader=None, indices=indices)
        self.datasets = datasets

    def __getitem__(self, idx: int):
        cur_indices = self.indices[idx]
        result = {'_index': idx, '_indices': cur_indices}

        for dataset_key in cur_indices:
            if dataset_key not in self.datasets:
                continue
            cur_dataset = self.datasets[dataset_key]
            result[dataset_key] = [
                cur_dataset.load_image(sensor_id, sample_idx)
                for sensor_id, sample_idx in cur_indices[dataset_key]
            ]
        return result
