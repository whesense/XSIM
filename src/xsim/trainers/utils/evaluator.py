import os
import json
from datetime import datetime
from dataclasses import field, dataclass

from rich.table import Table
import torch

from xsim.data import SensorType, SensorData, Subset
from xsim.metrics import Metric
from xsim.modeling.scene import Scene
from xsim.utils import track, console


@dataclass
class EvaluatorConfig:
    enabled: bool = True
    image_metrics: list[Metric] = field(default_factory=list)
    lidar_metrics: list[Metric] = field(default_factory=list)


class Evaluator:
    def __init__(
            self,
            trainer: "xsim.trainers.Trainer",
            config: EvaluatorConfig
    ):
        self.trainer = trainer
        self.config = config
        self.config.image_metrics = [m.to(trainer.device) for m in config.image_metrics]
        self.config.lidar_metrics = [m.to(trainer.device) for m in config.lidar_metrics]
        scene_sensors: dict[SensorType, SensorData] = trainer.sim_ds.scene.sensors
        self.sensor_names = {
            sensor_type: {
                sensor_id: name
                for sensor_id, name in zip(
                    scene_sensors[sensor_type].indices,
                    scene_sensors[sensor_type].captions
                )
            }
            for sensor_type in trainer.sim_ds.scene.sensors
        }

    @torch.no_grad()
    def compute_metrics(
            self,
            data_iter,
            metrics: list[Metric],
            tqdm_desc: str
    ):
        if not metrics:
            return None

        for metric in metrics:
            metric.reset()

        for sensor_batch in track(data_iter, description=tqdm_desc):
            scene = Scene.from_sensor_batch(self.trainer.sim_ds, sensor_batch)
            self.trainer.render_scene(scene)

            for metric in metrics:
                metric.update(scene)

        active_metrics = [m for m in metrics if m.total_cnt > 0]
        metric_names = [type(m).__name__ for m in active_metrics]
        sensor_ids = sorted({
            sensor_id
            for m in active_metrics
            for sensor_id in m.sensor_ids()
        })

        return {
            'metric_names': metric_names,
            'total': {
                metric_name: float(metric.value())
                for metric_name, metric in zip(metric_names, active_metrics)
            },
            'per_sensor': {
                sensor_id: {
                    metric_name: float(metric.value(sensor_id))
                    for metric_name, metric in zip(metric_names, active_metrics)
                    if sensor_id in metric.per_sensor_accum
                }
                for sensor_id in sensor_ids
            },
        }

    @torch.no_grad()
    def evaluate_on_split(self, split: Subset, split_name: str):
        result = {}
        if self.config.image_metrics:
            result['camera'] = self.compute_metrics(
                split.eval_iterator(target_device=self.trainer.device),
                self.config.image_metrics,
                tqdm_desc='Eval images on "{}"'.format(split_name)
            )

        if self.config.lidar_metrics:
            result['lidar'] = self.compute_metrics(
                split.eval_lidar_iterator(target_device=self.trainer.device),
                self.config.lidar_metrics,
                tqdm_desc='Eval lidars on "{}"'.format(split_name)
            )
        return result

    @torch.no_grad()
    def evaluate(self):
        if not self.config.enabled:
            return {}

        was_training = self.trainer.nodes.training
        self.trainer.nodes.eval()
        try:
            results = {
                split_name: self.evaluate_on_split(split, split_name)
                for split_name, split in [
                    ('train', self.trainer.train_split),
                    ('test', self.trainer.test_split)
                ]
            }
        finally:
            self.trainer.nodes.train(was_training)

        metrics_path = os.path.join(self.trainer.experiment_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            metrics_path = os.path.join(
                self.trainer.experiment_dir, 'metrics_{}.json'.format(stamp))
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)

        cons = console()
        cons.print("Saved metrics to [bold cyan]{}[/]".format(metrics_path))
        for modality, title in [('camera', 'Image metrics'), ('lidar', 'LiDAR metrics')]:
            table = self.metrics_table(results, modality, title)
            if table is not None:
                cons.print(table)

        return results

    def metrics_table(self, results, modality: str, title: str):
        sensor_type = SensorType[modality.upper()]
        train = results['train'].get(modality)
        test = results['test'].get(modality)
        if train is None and test is None:
            return None
        metric_names = (train or test)['metric_names']

        metrics = (
            self.config.image_metrics if sensor_type == SensorType.CAMERA
            else self.config.lidar_metrics
        )
        formatters = {type(m).__name__: m for m in metrics}

        def sensors(split):
            return split['per_sensor'] if split else {}

        sensor_ids = sorted(set(sensors(train)) | set(sensors(test)))

        table = Table(title=title, title_justify='left')
        table.add_column('Sensor', style='bold')
        for split in ('train', 'test'):
            for name in metric_names:
                table.add_column('{}\n{}'.format(split, name), justify='right')

        def fmt(name, v):
            if v is None:
                return '-'
            metric = formatters.get(name)
            return metric.format(v) if metric else '{}'.format(v)

        def row(caption, train_vals, test_vals):
            cells = [caption]
            cells += [fmt(n, train_vals.get(n)) for n in metric_names]
            cells += [fmt(n, test_vals.get(n)) for n in metric_names]
            return cells

        for sensor_id in sensor_ids:
            table.add_row(*row(
                "#{} {}".format(
                    sensor_id,
                    self.sensor_names[sensor_type][sensor_id]
                ),
                sensors(train).get(sensor_id, {}),
                sensors(test).get(sensor_id, {}),
            ))

        table.add_section()
        table.add_row(*row(
            'total',
            train['total'] if train else {},
            test['total'] if test else {},
        ), style='bold')
        return table
