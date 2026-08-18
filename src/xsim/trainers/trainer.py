import os
import sys
import shlex
from datetime import datetime
from typing import Optional

import torch
from jetcon import read

from xsim.data import SceneReconstructionDataset, SensorType, Subset
from xsim.data.reconstruction.dataset import SensorImage
from xsim.data.reconstruction.init import prepare_initialization
from xsim.data.reconstruction.sampling.split_dataset import split_dataset
from xsim.modeling import RenderInfo
from xsim.modeling.nodes import NodeCollection
from xsim.modeling.scene import Scene
from xsim.trainers.utils.evaluator import EvaluatorConfig
from xsim.utils import (
    snapshot_files_list, stage, progress_bar, profiling, console, apply_overrides
)
from xsim.trainers.utils import (
    TrainLogger, ImageVisualizer, Evaluator, Checkpointer, TrainProfiler, logged
)


class Trainer:
    config_path: str
    experiment_dir: str
    device: torch.device | str

    sim_ds: SceneReconstructionDataset = None
    train_split: Subset = None
    test_split: Subset = None
    nodes: NodeCollection

    def __init__(self, config_path: str, overrides: list[str] = None):
        self.config_path = config_path
        self.overrides = overrides or []
        self.cfg = read(config_path)
        if self.overrides:
            apply_overrides(self.cfg, self.overrides)
        self.trainer_cfg = self.cfg.get('trainer') or {}

        self.device = self.trainer_cfg.get('device', 'cuda')
        self.max_steps = self.trainer_cfg.get('max_steps', 40000)
        self.output_dir = self.trainer_cfg.get('output_dir', 'outputs')
        # Provisional: setup() re-resolves this once the scene is known, so the
        # run lands under its scene. Kept here so a Trainer that never calls
        # setup() still has somewhere to point.
        self.experiment_dir = self.resolve_experiment_dir()

        self.logger = TrainLogger(self.experiment_dir, self.config_path)
        self.visualizer = None
        self.evaluator = None
        self.checkpointer = None

    @classmethod
    def from_experiment(cls, experiment_dir: str):
        trainer = cls(os.path.join(experiment_dir, 'config.yaml'))
        trainer.experiment_dir = experiment_dir
        # enabled=False leaves the original run's train.log untouched.
        trainer.logger = TrainLogger(experiment_dir, trainer.config_path, enabled=False)
        trainer.checkpointer
        return trainer

    def save_config(self, path: str) -> None:
        """Record this run's config, keeping ``${env:...}`` markers unresolved.

        ``self.cfg`` holds paths already expanded against this machine, so
        saving it would pin the experiment here: moving the directory to a
        machine with different dataset roots leaves ``from_experiment``
        pointing at paths that do not exist. Re-reading without environment
        resolution keeps the markers, letting each machine resolve its own.
        """
        cfg = read(self.config_path, resolve_environ=False)
        if self.overrides:
            apply_overrides(cfg, self.overrides)
        cfg.save(path)

    def resolve_experiment_dir(self, scene_alias: Optional[str] = None) -> str:
        """Where this run writes: config name, scene, then start time."""
        parts = [
            self.output_dir,
            os.path.splitext(os.path.basename(self.config_path))[0],
        ]
        if scene_alias:
            parts.append(scene_alias)
        parts.append(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        return os.path.join(*parts)

    def setup(self):
        # The scene names the run directory, and it is only known once the
        # provider exists, so the provider is built before anything is written.
        with stage('Building scene provider'):
            provider = self.cfg.provider.build()

        self.experiment_dir = self.resolve_experiment_dir(provider.scene_alias)
        self.logger = TrainLogger(self.experiment_dir, self.config_path)

        os.makedirs(self.experiment_dir, exist_ok=True)
        self.logger.start()

        self.save_config(os.path.join(self.experiment_dir, 'config.yaml'))
        with open(os.path.join(self.experiment_dir, 'cli.txt'), 'w') as f:
            f.write(shlex.join([sys.executable, *sys.argv]) + '\n')
        if self.trainer_cfg.get('snapshot', True):
            snapshot_files_list(
                os.path.join(self.experiment_dir, 'code_snapshot.zip'),
                exclude=(self.output_dir,),
            )

        self.build_pipeline(provider)

    def build_pipeline(self, provider=None):
        if provider is None:
            with stage('Building scene provider'):
                provider = self.cfg.provider.build()

        with stage('Loading dataset'):
            self.sim_ds = SceneReconstructionDataset(provider, self.cfg.dataset)
            self.train_split, self.test_split = split_dataset(
                self.sim_ds, self.cfg.split.build()
            )

        with stage('Initializing scene'):
            init = prepare_initialization(self.sim_ds, self.cfg.init.build())
            init['train_split'] = self.train_split

        with stage('Building model'):
            self.nodes = self.cfg.nodes.build()(self.sim_ds, init).to(self.device)
            self.nodes.init_optimization()

        self.checkpointer = Checkpointer(
            self.nodes, self.experiment_dir, self.device, self.trainer_cfg)

        eval_cfg = EvaluatorConfig(**(
            self.trainer_cfg.eval.build()
            if 'eval' in self.trainer_cfg else {}
        ))
        self.evaluator = Evaluator(self, eval_cfg)

        self.visualizer = ImageVisualizer(
            self, self.trainer_cfg.get('visualizer', {}))

    @logged
    def train(self):
        if self.nodes is None:
            self.setup()

        profiler = TrainProfiler(self.trainer_cfg, self.experiment_dir)
        step = 0

        with profiler, progress_bar('train', total=self.max_steps) as progress:
            for sensor_batch in self.train_split.train_iterator(target_device=self.device):
                self.nodes.zero_grad()

                with profiling.annotate('forward'):
                    scene = Scene.from_sensor_batch(self.sim_ds, sensor_batch)
                    result = self.nodes(scene)

                with profiling.annotate('compute_losses'):
                    losses = self.nodes.compute_losses(scene, result)
                    loss = sum(losses.values())

                for render, gt in zip(scene.renders, scene.batch):
                    if gt.sensor_type != SensorType.CAMERA:
                        continue
                    self.train_split.img_sampler.update_error(
                        sensor_id=gt.sensor_id,
                        sample_idx=gt.sample_idx,
                        render=render.result.color.clamp(0, 1),
                        target=gt.image,
                        target_mask=gt.gt_mask,
                    )

                with profiling.annotate('pre_backward'):
                    self.nodes.pre_backward(scene, result)

                with profiling.annotate('backward'):
                    loss.backward()

                with profiling.annotate('post_backward'):
                    self.nodes.post_backward(scene, result)

                with profiling.annotate('optimizer_step'):
                    self.nodes.optimizer_step(scene)

                step += 1
                progress.update()
                progress.set_postfix(loss=float(loss.detach()))

                if profiler.step(step):
                    break

                self.checkpointer.step(step)
                self.visualizer.step(step)

                if step >= self.max_steps:
                    break

        if profiler.active:
            # a profiling run stops early; skip the final checkpoint + eval.
            return

        self.checkpointer.save(step)
        self.evaluator.evaluate()

        console().print(
            "Output directory: [bold cyan]{}[/]".format(self.experiment_dir))

    @torch.no_grad()
    def render_scene(self, scene: Scene) -> Scene:
        self.nodes(scene)
        return scene

    def replay_camera_scene(
            self,
            sensor_id: int,
            sample_idx: int
    ) -> Scene:
        mask = self.sim_ds.camera_masks[sensor_id]
        height, width = mask.shape[:2]

        return Scene(
            time_bounds=self.sim_ds.time_range,
            renders=[
                RenderInfo(
                    sensor_type=SensorType.CAMERA,
                    sensor_id=sensor_id,
                    sample_idx=sample_idx,
                    camera=self.sim_ds.cameras[sensor_id][sample_idx].to(self.device),
                    width=width, height=height,
                )
            ]
        )

    def replay_lidar_scene(
            self,
            sensor_id: int,
            sample_idx: int
    ):
        sweep = self.sim_ds.sweeps[sensor_id][sample_idx]
        height, width = sweep.shape[:2]
        return Scene(
            time_bounds=self.sim_ds.time_range,
            renders=[
                RenderInfo(
                    sensor_type=SensorType.LIDAR,
                    sensor_id=sensor_id,
                    sample_idx=sample_idx,
                    camera=self.sim_ds.lidar_cameras[sensor_id][sample_idx],
                    width=width, height=height,
                )
            ],
            batch=[
                SensorImage(
                    sensor_type=SensorType.LIDAR,
                    sensor_id=sensor_id,
                    sample_idx=sample_idx,
                    camera=self.sim_ds.lidar_cameras[sensor_id][sample_idx],
                    image=sweep,
                    gt_mask=sweep.mask,
                    info={}
                )
            ]
        )
