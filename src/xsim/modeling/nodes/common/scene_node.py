from typing import Optional

import torch
from xsim.data import SceneReconstructionDataset

from xsimgs.structures import ROIBox3D, ROIBox3DModule
from xsim.utils import remove_prefix

from xsim.modeling.scene import Scene


from functools import partial


def try_build(node, sim_ds, init):
    if isinstance(node, partial):
        return node(sim_ds, init)
    return node


LossesType = Optional[list[tuple[torch.nn.Module, float]]]


class SceneNode(torch.nn.Module):
    @classmethod
    def create(
            cls,
            sim_ds: SceneReconstructionDataset,
            init: dict,
            losses: LossesType = None
    ):
        return cls(roi=sim_ds.roi, losses=losses)

    @classmethod
    def params_from_dict_and_cfg(cls, weights, node_cfg):
        roi = ROIBox3DModule.from_state_dict(remove_prefix(weights, 'roi.'))
        return dict(roi=roi, **node_cfg)

    @classmethod
    def load_from_state_dict(cls, state_dict, node_cfg):
        params = cls.params_from_dict_and_cfg(state_dict, node_cfg)
        node = cls(**params)
        result = node.load_state_dict(state_dict, strict=False)
        print(result)
        return node

    opt: torch.optim.Optimizer = None
    schedulers: list = []

    def __init__(
            self,
            roi: ROIBox3D,
            losses: LossesType = None
    ):
        super().__init__()
        self.roi = ROIBox3DModule.from_roi(roi)
        losses = losses or []
        self.loss_modules = torch.nn.ModuleList([x[0] for x in losses])
        self.loss_weights = [x[1] for x in losses]
        self.step = torch.nn.Buffer(torch.zeros(1, dtype=torch.int32))
        self.inference_mode = False

    def init_optimization(self):
        pass

    def resolve_lr_scale_factor(self, name: str | float | int | None):
        if isinstance(name, float) or isinstance(name, int):
            return name

        match name:
            case 'scene_radius':
                return self.roi.size.amax().mul(1.1 * 0.5).item()
            case None:
                return 1.0
            case _:
                raise ValueError('Unsupported scale factor: {}'.format(name))

    def forward(self, scene: Scene):
        return scene

    def pre_backward(self, scene: Scene, output=None):
        pass

    def post_backward(self, scene: Scene, output=None):
        pass

    def optimizer_step(self, scene: Scene):
        if self.opt is not None:
            self.opt.step()
        for scheduler in self.schedulers:
            scheduler.step()
        self.step += 1

    def _compute_single_loss(self, scene, result, weight, loss, prefix: str):
        loss_result = loss(self, scene, result)
        if isinstance(loss_result, dict):
            ret = {}
            for key in loss_result:
                ret['_'.join([prefix, key])] = loss_result[key] * weight
            return ret

        return {prefix: loss_result * weight}

    def compute_losses(self, scene: Scene, fwd_result) -> dict[str, torch.Tensor]:
        prefix = self.__class__.__name__ + '_{}'
        dicts = [
            self._compute_single_loss(
                scene, fwd_result, weight, loss,
                prefix.format(loss.__class__.__name__)
            ) for loss, weight in zip(self.loss_modules, self.loss_weights)
        ]
        result = {}
        for d in dicts:
            result.update(d)
        return result

    def optimize_inference(self):
        self.inference_mode = True

    def remap_instances(self, instances):
        pass
