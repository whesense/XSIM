"""Render every frame of a trained experiment next to its ground truth.

Loads the experiment's saved config and a checkpoint, rebuilds the pipeline and
writes, per camera:

    <experiment_dir>/renders/<sensor_caption>/frame_XXXXX_render.png
    <experiment_dir>/renders/<sensor_caption>/frame_XXXXX_gt.png
    <experiment_dir>/renders/<sensor_caption>/frame_XXXXX_depth.png

and per LiDAR, the rendered depth back-projected into world space beside the
sweep it is compared against:

    <experiment_dir>/renders/<sensor_caption>/frame_XXXXX_render.ply
    <experiment_dir>/renders/<sensor_caption>/frame_XXXXX_gt.ply

Usage:
    python tools/render.py outputs/waymo_788/2026-07-07_12-37-34
    python tools/render.py <experiment_dir> --stride 5 --checkpoint <path/to/nodes_XXXXXX.pth>
    python tools/render.py <experiment_dir> --skip-lidar
    python tools/render.py <experiment_dir> --skip-color
"""
import os
import argparse

import torch

from xsim.data import SensorType
from xsim.modeling.nodes.postprocess import PointCloudNode
from xsim.modeling.nodes.render import XSIMGSRenderNode
from xsim.modeling.scene import Scene
from xsim.trainers import Trainer
from xsim.utils import track, update_env
from xsim.visualization import write_image
from xsim.visualization.colormaps import Turbo


def sensor_dirs(output_dir: str, sensor_ids: list[int], captions: list[str]):
    dirs = {}
    for sensor_id, caption in zip(sensor_ids, captions):
        dirs[sensor_id] = os.path.join(output_dir, caption)
        os.makedirs(dirs[sensor_id], exist_ok=True)
    return dirs


def rendered_scenes(trainer: Trainer, loader: str, indices: list, description: str):
    """Render one sensor sample per step, yielding ``(scene, sensor_batch)``.

    Only the named loader is built, so rendering cameras does not pay for
    loading sweeps (and vice versa).
    """
    sim_ds = trainer.sim_ds
    iterator = sim_ds.multi_loader_iterator(
        {loader: sim_ds.loader_datasets(None)[loader]}, indices,
        infinite_iterator=False,
        shuffle=False,
        target_device=trainer.device,
    )

    for iter_batch in track(iterator, description=description, total=len(indices)):
        sensor_batch = sim_ds.make_sensor_batch(iter_batch)
        scene = Scene.from_sensor_batch(sim_ds, sensor_batch)
        trainer.render_scene(scene)
        yield scene, sensor_batch


@torch.no_grad()
def render_cameras(trainer: Trainer, output_dir: str, stride: int = 1):
    sim_ds = trainer.sim_ds
    dirs = sensor_dirs(output_dir, sim_ds.camera_indices, sim_ds.camera_captions)

    indices = [
        {'image': [(camera_id, sample_idx)]}
        for camera_id in sim_ds.camera_indices
        for sample_idx in range(0, len(sim_ds.cameras[camera_id]), stride)
    ]

    for scene, sensor_batch in rendered_scenes(
            trainer, 'image', indices, 'Rendering cameras'):
        for render, gt in zip(scene.renders, sensor_batch):
            if gt.sensor_type != SensorType.CAMERA:
                continue

            path_fmt = os.path.join(
                dirs[gt.sensor_id],
                'frame_{:05d}_{{}}.png'.format(gt.sample_idx)
            )
            write_image(path_fmt.format('render'), render.result.color)
            write_image(path_fmt.format('gt'), gt.image)

            depth = render.result.depth
            if depth is not None and depth.numel() > 0:
                # Inverse depth on Turbo, percentile-normalized -- the same
                # mapping the training visualizer uses, so a rendered frame here
                # is comparable to the ones logged during training. Sky pixels
                # carry depth 0, and the colormap paints those 1/0 infinities
                # with its "bad" color instead of letting them skew the range.
                write_image(path_fmt.format('depth'), Turbo(
                    1.0 / depth.reshape(render.height, render.width),
                    normalize=True, q_min=0.02, q_max=0.98
                ))


def pointcloud_node(trainer: Trainer):
    """A node to build the rendered clouds with, or ``None`` if the pipeline has one.

    The cloud is the rendered depth back-projected along the cached rays, which
    is what ``PointCloudNode`` does. Configs wire it in for the ChamferDist
    metric, but an experiment trained without it would otherwise render nothing
    here, so supply one instead.
    """
    for node in trainer.nodes.nodes:
        if isinstance(node, PointCloudNode):
            return None

    return PointCloudNode(create_pointcloud={SensorType.LIDAR: True}).eval()


@torch.no_grad()
def render_lidars(trainer: Trainer, output_dir: str, stride: int = 1):
    from plytorch import PointCloud

    sim_ds = trainer.sim_ds
    dirs = sensor_dirs(output_dir, sim_ds.lidar_indices, sim_ds.lidar_captions)
    build_points = pointcloud_node(trainer)

    # One sweep per step: make_sensor_batch keeps a random max_lidars_per_batch
    # of the lidars it is handed, so a batch of several would silently drop all
    # but one of them.
    indices = [
        {'lidar': [(lidar_id, sample_idx)]}
        for lidar_id in sim_ds.lidar_indices
        for sample_idx in range(0, len(sim_ds.sweeps[lidar_id]), stride)
    ]

    for scene, sensor_batch in rendered_scenes(
            trainer, 'lidar', indices, 'Rendering lidars'):
        if build_points is not None:
            build_points(scene)

        for render, gt in zip(scene.renders, sensor_batch):
            if gt.sensor_type != SensorType.LIDAR:
                continue

            path_fmt = os.path.join(
                dirs[gt.sensor_id],
                'frame_{:05d}_{{}}.ply'.format(gt.sample_idx)
            )
            points = render.result.info.get('points')
            if points is not None:
                PointCloud(points=points).save(path_fmt.format('render'))
            # The sweep's own returns, dropping the cells that measured nothing.
            PointCloud(points=gt.image.masked.xyz.reshape(-1, 3)).save(
                path_fmt.format('gt'))


def set_camera_depth_max_density(trainer: Trainer, enabled: bool):
    nodes = [
        node for node in trainer.nodes.modules()
        if isinstance(node, XSIMGSRenderNode)
    ]
    if not nodes:
        raise SystemExit(
            '--camera-depth-max-density: no XSIMGSRenderNode in this pipeline')

    for node in nodes:
        node.camera_depth_max_density = enabled


def main():
    update_env()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('experiment_dir', help='experiment directory to render')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint to load (default: latest in the dir)')
    parser.add_argument('--stride', type=int, default=1,
                        help='render only every k-th frame of each sensor')
    parser.add_argument('--output-dir', default=None,
                        help='where to write renders (default: <experiment_dir>/renders)')
    parser.add_argument('--skip-color', action='store_true',
                        help='do not render the camera images')
    parser.add_argument('--skip-lidar', action='store_true',
                        help='do not render the lidar point clouds')
    parser.add_argument('--camera-depth-max-density',
                        action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.skip_color and args.skip_lidar:
        parser.error('--skip-color and --skip-lidar together leave nothing to render')

    trainer = Trainer.from_experiment(args.experiment_dir)
    trainer.build_pipeline()
    trainer.checkpointer.load(args.checkpoint)
    trainer.nodes.eval()

    set_camera_depth_max_density(trainer, args.camera_depth_max_density)

    output_dir = args.output_dir or os.path.join(args.experiment_dir, 'renders')
    if not args.skip_color:
        render_cameras(trainer, output_dir, stride=args.stride)
    if not args.skip_lidar:
        render_lidars(trainer, output_dir, stride=args.stride)

    print('Renders written to:', output_dir)


if __name__ == '__main__':
    main()
