import itertools
from typing import Optional

import numpy as np
import torch
from xsimgs.structures import ROIBox3D

from xsim.modeling.scene import Scene
from xsim.modeling.gaussian import ActivatedGaussians, GaussianField as GF
from xsim.utils import profiling

from ..common import NodeCollection, LossesType
from .gaussian_node import GaussianSceneNode


def set_cur_slice(scene: Scene, cur_slice):
    mask = None if cur_slice is None else np.s_[cur_slice]
    for render in scene.renders:
        render.result.info["global_mask"] = mask


class GaussianContainer(NodeCollection):
    @classmethod
    def create(cls, sim_ds, init, nodes, losses=None):
        return GaussianContainer(
            nodes=[node(sim_ds, init) for node in nodes],
            roi=sim_ds.roi,
            losses=losses or [],
        )

    def __init__(
            self,
            roi: ROIBox3D,
            nodes: list[GaussianSceneNode],
            losses: LossesType = None
    ):
        super().__init__(nodes=nodes, roi=roi, losses=losses)
        self.container: ActivatedGaussians | None = None
        self.offsets: list[int] = []

    def forward(self, scene: Scene):
        if self.inference_mode:
            return self.store_to_container(scene)

        return self.forward_train(scene)

    def forward_train(self, scene: Scene):
        scene.renderables = []
        outputs = []
        slices = []
        offset = 0
        for _, node in self.active_nodes():
            start = len(scene.renderables)
            name = type(node).__name__
            with profiling.annotate('{}.forward'.format(name)):
                output = node.forward(scene)
            profiling.mark_backward(output, name)
            outputs.append(output)
            added = 0
            for renderable in scene.renderables[start:]:
                renderable.check()
                added += len(renderable)
            slices.append(slice(offset, offset + added) if added else None)
            offset += added
        # Individual nodes may render nothing (an empty node is a no-op, and its
        # slice is already None above); all of them at once leaves cat() without
        # even a reference renderable to take the fields from.
        if not scene.renderables:
            raise RuntimeError(
                'Every gaussian node is empty -- there is nothing to render. '
                'The models were pruned down to zero particles; check the '
                'densification / prune thresholds of their strategies.'
            )

        scene.container = ActivatedGaussians.cat(scene.renderables)
        return outputs, slices

    def pre_backward(self, scene: Scene, fwd_result):
        _, slices = fwd_result
        for (_, node), cur_slice in zip(self.active_nodes(), slices):
            set_cur_slice(scene, cur_slice)
            node.pre_backward(scene, cur_slice)

    def post_backward(self, scene: Scene, fwd_result):
        _, slices = fwd_result
        for (_, node), cur_slice in zip(self.active_nodes(), slices):
            set_cur_slice(scene, cur_slice)
            node.post_backward(scene, cur_slice)

    def compute_losses(self, scene: Scene, fwd_result) -> dict[str, torch.Tensor]:
        outputs, _ = fwd_result
        result = {}
        for (i, node), output in zip(self.active_nodes(), outputs):
            for key, value in node.compute_losses(scene, output).items():
                result[f"{i}_{key}"] = value
        return result

    @torch.no_grad()
    def optimize_inference(self):
        # call to each gaussian node optimize_inference
        super().optimize_inference()
        self.allocate()

    @torch.no_grad()
    def allocate(self):
        nodes = [node for _, node in self.active_nodes()]
        counts = [node.num_particles for node in nodes]
        self.offsets = [0] + list(itertools.accumulate(counts))[:-1]
        total = sum(counts)

        ref = nodes[0].model
        data: dict[str, torch.Tensor] = {}
        for key, spec in ref.specs.items():
            if spec.ret:
                data[key] = torch.zeros(total, ref.param_size(key), device=ref.device)
        data[GF.features] = torch.zeros(
            total, max(node.model.color_dim for node in nodes), device=ref.device
        )
        data[GF.mask] = torch.zeros(total, dtype=torch.bool, device=ref.device)
        if any(node.return_velocity for node in nodes):
            data[GF.velocity] = torch.zeros(total, 3, device=ref.device)

        self.container = ActivatedGaussians(data, ref.specs)
        for node, offset in zip(nodes, self.offsets):
            node.populate_container(self.container, offset)

    def store_to_container(self, scene: Scene):
        scene.container = self.container
        for (_, node), offset in zip(self.active_nodes(), self.offsets):
            node.store_to_container(scene, offset)
        return scene
