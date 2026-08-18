import os
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np

from xsim.data import SceneReconstructionDataset
from xsim.visualization import image_grid, encode_jpeg
from xsim.visualization.colormaps import Turbo


def flatten(x):
    return (
        [] if x is None else sum((flatten(e) for e in x), [])
        if isinstance(x, list) else [x]
    )


@dataclass
class VisualizerConfig:
    interval: int = 2000
    img_downscale: float = 1.0
    jpeg_quality: int = 95
    camera_order: Optional[list[int | None] | list[list[int | None]]] = None
    num_samples: int = 3
    samples: list[int | list[tuple[int, int]]] = None


@dataclass
class Visualization:
    idx: int
    # sensor_id -> sample_idx
    samples: dict[int, int]


class ImageVisualizer:
    def __init__(
            self,
            trainer: "xsim.trainers.Trainer",
            cfg: dict,
    ):
        self.trainer = trainer
        self.cfg = VisualizerConfig(**(cfg or {}))

        self.visualizations_dir = os.path.join(
            self.trainer.experiment_dir, 'visualizations'
        )
        os.makedirs(self.visualizations_dir, exist_ok=True)

        self.camera_vis_order = (
            self.cfg.camera_order or self.trainer.sim_ds.camera_vis_order
        )
        self.camera_ids = sorted(set(flatten(self.camera_vis_order)))
        self.samples = self.pick_samples()
        self.output_fmt = os.path.join(self.visualizations_dir, '{:04d}/{}_step_{:05d}.jpg')
        print('visualization samples:', self.samples)

    def pick_samples(self) -> list[Visualization]:
        if self.cfg.samples:
            return [
                (
                    Visualization(
                        idx=v,
                        samples={cam_id: v for cam_id in self.camera_ids}
                    ) if isinstance(v, int) else
                    Visualization(
                        idx=v[0][1],
                        samples={cam_id: sample_idx for cam_id, sample_idx in v}
                    )
                ) for v in self.cfg.samples
            ]

        sim_ds: SceneReconstructionDataset = self.trainer.sim_ds
        sample_idx = {
            cam_id: np.linspace(
                0, len(sim_ds.cameras[cam_id]) - 1, self.cfg.num_samples
            ).round().astype(int).tolist()
            for cam_id in self.camera_ids
        }

        return [
            Visualization(
                idx=sample_idx[self.camera_ids[0]][i],
                samples={cam_id: sample_idx[cam_id][i] for cam_id in self.camera_ids},
            ) for i in range(self.cfg.num_samples)
        ]

    def render_frame(self, sample: Visualization):
        colors, depths = {}, {}
        for (camera_id, sample_idx) in sample.samples.items():
            scene = self.trainer.replay_camera_scene(camera_id, sample_idx)
            self.trainer.render_scene(scene)
            render = scene.renders[0]
            result = render.result

            if result.color is not None and result.color.numel() > 0:
                colors[camera_id] = result.color.clamp(0, 1)

            if result.depth is not None and result.depth.numel() > 0:
                depth = result.depth.reshape(render.height, render.width)
                depths[camera_id] = Turbo(
                    1.0 / depth, normalize=True, q_min=0.02, q_max=0.98
                )
        return colors, depths

    @torch.no_grad()
    def step(self, step: int):
        if (not (self.cfg.interval and step % self.cfg.interval == 0)) and (step != 1):
            return

        # Render the way the evaluator and the render script do: in eval mode.
        # Nodes behave differently there -- the renderer takes camera depth from
        # the max of the optical and lidar densities, for one -- so a training
        # mode visualization would not show what the experiment is judged on.
        nodes = self.trainer.nodes
        was_training = nodes.training
        nodes.eval()
        try:
            self.write_visualizations(step)
        finally:
            nodes.train(was_training)

    def write_visualizations(self, step: int):
        for sample in self.samples:
            colors, depths = self.render_frame(sample)
            if len(colors) == 0:
                continue

            sample_dir = os.path.join(self.visualizations_dir, '{:04d}'.format(sample.idx))
            os.makedirs(sample_dir, exist_ok=True)

            self.write_img_grid(
                self.output_fmt.format(sample.idx, 'color', step), colors
            )
            self.write_img_grid(
                self.output_fmt.format(sample.idx, 'depth', step), depths
            )

    def write_img_grid(self, path: str, images: dict):
        if len(images) == 0:
            return

        grid = image_grid(images, self.camera_vis_order, downscale=self.cfg.img_downscale)
        jpeg = encode_jpeg(grid, quality=self.cfg.jpeg_quality)
        with open(path, 'wb') as f:
            f.write(jpeg.cpu().numpy().tobytes())
