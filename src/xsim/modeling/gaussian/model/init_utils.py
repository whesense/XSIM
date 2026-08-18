from dataclasses import dataclass, field

import torch

from xsim.structures.init_instance import SceneInitInstance


@dataclass
class InitContext:
    instances: dict[int, SceneInitInstance]

    device: torch.device | str = "cpu"
    built_params: dict = field(default_factory=dict)
    model_cls: type = None
    configs: dict = None

    @property
    def num_particles(self) -> int:
        return sum([len(inst.points) for inst in self.instances.values()])


@dataclass
class EmptyInitCfg:
    pass


def resolve_configs(cls: type, configs) -> dict:
    out = {}
    cfg_owner = {}   # InitCfg type -> contributing class
    for class_type in cls.__mro__:
        if "param_init" in vars(class_type):
            cfg_cls = vars(class_type).get("InitCfg")
            # a contributor with no InitCfg gets an empty cfg
            out[class_type] = cfg_cls() if cfg_cls is not None else EmptyInitCfg()
            if cfg_cls is not None:
                cfg_owner[cfg_cls] = class_type
    if configs is None:
        return out

    if isinstance(configs, dict):
        unknown = [k for k in configs if k not in out]
        if unknown:
            raise TypeError(
                f"{cls.__name__}.build_params got config(s) for class(es) it "
                f"does not init: {[k.__name__ for k in unknown]}. "
                f"Expected one of {[k.__name__ for k in out]}."
            )
        out.update(configs)
        return out

    for cfg in configs:
        owner = cfg_owner.get(type(cfg))
        if owner is None:
            raise TypeError(
                f"{cls.__name__}.build_params got an unexpected config "
                f"{type(cfg).__qualname__}; this model has no class that uses it. "
                f"Expected instances of {[t.__qualname__ for t in cfg_owner]}."
            )
        out[owner] = cfg
    return out


def build_params(cls, ctx: InitContext, configs=None) -> dict:
    ctx.model_cls = cls
    ctx.configs = resolve_configs(cls, configs)

    # for cls_t, cfg in ctx.configs.items():
    #     print(cls_t, cfg)

    for class_type in cls.__mro__:
        own = vars(class_type).get("param_init")
        if not own:
            continue
        cfg = ctx.configs.get(class_type)
        for name, fn in own.items():
            if name not in cls._param_specs or name in ctx.built_params:
                continue
            value = fn(ctx, cfg)
            if value is not None:
                ctx.built_params[name] = value
    return ctx.built_params
