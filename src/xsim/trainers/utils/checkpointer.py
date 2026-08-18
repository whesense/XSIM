import os
import glob
from typing import Optional

import torch


class Checkpointer:
    def __init__(
            self,
            nodes: torch.nn.Module,
            experiment_dir: str,
            device: torch.device | str,
            cfg: dict
    ):
        self.nodes = nodes
        self.checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.device = device
        self.interval = cfg.get('checkpoint_interval', 5000)

    def step(self, step: int):
        if self.interval and step % self.interval == 0:
            self.save(step)

    def save(self, step: int) -> str:
        path = os.path.join(self.checkpoints_dir, 'nodes_{:06d}.pth'.format(step))
        torch.save(self.nodes.state_dict(), path)
        return path

    def latest(self) -> str | None:
        paths = sorted(glob.glob(os.path.join(self.checkpoints_dir, 'nodes_*.pth')))
        return paths[-1] if paths else None

    def load(self, path: Optional[str] = None) -> str:
        if path is None:
            path = self.latest()

        if path is None:
            raise FileNotFoundError('No checkpoint found in {}'.format(self.checkpoints_dir))

        state = torch.load(path, map_location=self.device)
        self.nodes.load_state_dict(state, strict=False)
        print('Loaded checkpoint:', path)
        return path
