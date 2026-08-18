from contextlib import contextmanager

import torch
from xsimgs.structures import ROIBox3D

from xsim.modeling.scene import Scene
from xsim.utils import profiling
from .scene_node import SceneNode, LossesType


class NodeCollection(SceneNode):
    def __init__(
            self,
            roi: ROIBox3D,
            nodes: list[SceneNode],
            losses: LossesType = None,
    ):
        super().__init__(
            roi=roi if roi is not None else nodes[0].roi,
            losses=losses
        )
        self.nodes = torch.nn.ModuleList(nodes)
        self._enabled_node_ids = None   # None == all nodes active

    @classmethod
    def create(cls, sim_ds, init, nodes, losses=None):
        return cls(
            nodes=[node(sim_ds, init) for node in nodes],
            roi=sim_ds.roi,
            losses=losses or [],
        )

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, idx) -> SceneNode:
        return self.nodes[idx]

    def enabled_node_ids(self) -> set[int]:
        if self._enabled_node_ids is None:
            return set(range(len(self.nodes)))
        return set(self._enabled_node_ids)

    def active_nodes(self):
        active = self.enabled_node_ids()
        return [(i, node) for i, node in enumerate(self.nodes) if i in active]

    @contextmanager
    def _enable_ctx(self, ids: set[int]):
        prev = self._enabled_node_ids
        self._enabled_node_ids = ids
        try:
            yield self
        finally:
            self._enabled_node_ids = prev

    def enable_nodes(self, ids: list[int]):
        return self._enable_ctx(self.enabled_node_ids() & set(ids))

    def disable_nodes(self, ids: list[int]):
        return self._enable_ctx(self.enabled_node_ids() - set(ids))

    def init_optimization(self):
        for _, node in self.active_nodes():
            node.init_optimization()

    def forward(self, scene: Scene):
        outputs = []
        for _, node in self.active_nodes():
            name = type(node).__name__
            with profiling.annotate('{}.forward'.format(name)):
                output = node.forward(scene)
            profiling.mark_backward(output, name)
            outputs.append(output)
        return outputs

    def pre_backward(self, scene: Scene, outputs):
        for (_, node), output in zip(self.active_nodes(), outputs):
            node.pre_backward(scene, output)

    def post_backward(self, scene: Scene, outputs):
        for (_, node), output in zip(self.active_nodes(), outputs):
            node.post_backward(scene, output)

    def optimizer_step(self, scene: Scene):
        for _, node in self.active_nodes():
            node.optimizer_step(scene)
        self.step += 1

    def compute_losses(self, scene: Scene, outputs) -> dict[str, torch.Tensor]:
        result = {}
        for (i, node), output in zip(self.active_nodes(), outputs):
            for key, value in node.compute_losses(scene, output).items():
                result[f"{i}_{key}"] = value
        return result

    def optimize_inference(self):
        super().optimize_inference()
        for _, node in self.active_nodes():
            node.optimize_inference()

    def remap_instances(self, instances):
        for _, node in self.active_nodes():
            node.remap_instances(instances)
