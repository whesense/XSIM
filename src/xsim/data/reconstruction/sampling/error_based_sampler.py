import random
from typing import Optional

import torch
import torch.multiprocessing as mp



class ErrorBasedImageSampler(torch.nn.Module):
    def __init__(
            self,
            img_indices: list[tuple[int, int]],
            metric_cls = None,
            metric_kwargs: Optional[dict] = None,
            error_based_ratio: float = 0.5,
            start_weight: float = 3.0,
            start_ratio: float = 0.1
    ):
        super().__init__()
        if metric_cls is None:
            from xsim.metrics import L1ImageMetric
            metric_cls = L1ImageMetric

        self.metric = metric_cls(**(metric_kwargs if metric_kwargs is not None else {}))
        self.img_indices = img_indices
        self.img_to_id = {index: i for i, index in enumerate(img_indices)}
        self.num_images = len(img_indices)
        self.error_based_ratio = error_based_ratio
        self.start_weight = start_weight
        self.start_ratio = start_ratio
        self.start_num_images = int(start_ratio * self.num_images)
        self.error_buffer = None
        self.reset_buffer()
        self.error_weights = torch.ones(self.num_images)
        self.error_weights[:self.start_num_images] = torch.linspace(
            self.start_weight, 1, self.start_num_images
        )

    def reset_buffer(self):
        self.error_buffer = torch.ones(self.num_images)

    def update_error(
            self,
            sensor_id: int,
            sample_idx: int,
            render: torch.Tensor,
            target: torch.Tensor,
            target_mask: torch.Tensor
    ):
        img_id = self.img_to_id[(sensor_id, sample_idx)]
        self.error_buffer[img_id] = self.metric(render, target, target_mask)

    def propose_sample(self) -> int:
        use_buffer = random.random() < self.error_based_ratio
        if not use_buffer:
            img_idx = int(torch.randint(low=0, high=self.num_images, size=(1,)).item())
        else:
            img_idx = int(torch.multinomial(
                self.error_buffer * self.error_weights,
                num_samples=1,
                replacement=False
            ).item())
        return img_idx

    def __getitem__(self, *args, **kwargs):
        return self.propose_sample()



class ErrorBasedIndicesProvider:
    def __init__(
            self,
            indices,
    ):
        self.indices = indices
        self.mp_queue = mp.Queue()

    def put(self, idx: int):
        self.mp_queue.put(idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, *args, **kwargs):
        return self.indices[self.mp_queue.get()]


