from xsim.data import SceneReconstructionDataset
from .gaussian_node import GaussianSceneNode
from ...gaussian import init_from_instance


class StaticSceneNode(GaussianSceneNode):
    @classmethod
    def create(
            cls,
            sim_ds: SceneReconstructionDataset,
            init: dict,

            model_type: type,
            init_configs: list = None,
            model_params: dict = None,
            return_velocity: bool = False,

            strategy_type: type = None,
            strategy_cfg: dict = None,
            opt_cfg = None,
            fixed_params: list[str] = None,
            losses: list[str] = None
    ):
        model_params_init = init_from_instance(
            model_type, init['bg'], init_configs or []
        )
        model_params_init.update(model_params or {})

        return cls(
            roi=sim_ds.roi,
            model_type=model_type,
            model_params=model_params_init,
            return_velocity=return_velocity,
            strategy_type=strategy_type,
            strategy_cfg=strategy_cfg,
            opt_cfg=opt_cfg,
            fixed_params=fixed_params,
            losses=losses
        )
