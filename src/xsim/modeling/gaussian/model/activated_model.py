from typing import Optional

import torch

from .specs import ParamSpec
from .fields import GaussianField as GF


class ActivatedGaussians:
    # keys allowed in data on top of the model specs: colors and the
    # per-particle fields produced at render time rather than declared as specs.
    RUNTIME = {GF.features, GF.velocity, GF.mask, GF.local_ids}

    data: dict[str, torch.Tensor]
    specs: dict[str, ParamSpec]

    def __init__(
            self,
            data: dict[str, torch.Tensor],
            specs: Optional[dict[str, ParamSpec]] = None
    ):
        self.__dict__["data"] = dict(data)
        self.__dict__["specs"] = specs or {}

    def __getattr__(self, name):
        data = self.__dict__["data"]
        if name in data:
            return data[name]
        if name in self.__dict__["specs"]:
            return None
        raise AttributeError(name)

    def __len__(self):
        return len(self.data[GF.positions])

    def keys(self):
        return self.data.keys()

    def check(self) -> "ActivatedGaussians":
        allowed = set(self.specs) | self.RUNTIME
        unknown = set(self.data) - allowed
        assert not unknown, (
            f"ActivatedGaussians has unknown keys {sorted(unknown)}; "
            f"allowed are specs {sorted(self.specs)} + runtime {sorted(self.RUNTIME)}"
        )
        n = len(self)
        mismatched = {k: tuple(v.shape) for k, v in self.data.items() if len(v) != n}
        assert not mismatched, (
            f"ActivatedGaussians num particles mismatch (expected {n}): {mismatched}"
        )
        return self

    @staticmethod
    def cat(
            lst: list["ActivatedGaussians"],
            exclude_keys: Optional[list[str]] = None
    ) -> "ActivatedGaussians":
        exclude = set(exclude_keys) if exclude_keys else set()
        ref = lst[0]
        specs = ref.specs
        out: dict[str, torch.Tensor] = {}

        keys = {k for g in lst for k in g.data} - exclude - {GF.mask, GF.local_ids}

        for key in sorted(keys):
            missing = [i for i, g in enumerate(lst) if key not in g.data]
            if missing:
                raise ValueError(
                    'Cannot assemble the gaussian container: field "{}" is '
                    'missing from renderable(s) {} of {}. Every renderable '
                    'must carry the same fields -- if "{}" comes from a model '
                    'mixin, compose that mixin on every gaussian node of the '
                    'container config (renderables appear in node order).'.format(
                        key, missing, len(lst), key
                    )
                )
            out[key] = torch.cat([g.data[key] for g in lst], dim=0)

        if any(GF.mask in g.data for g in lst):
            out[GF.mask] = torch.cat([
                g.data[GF.mask] if GF.mask in g.data
                else torch.ones(len(g), dtype=torch.bool, device=g.positions.device)
                for g in lst
            ], dim=0)

        if any(GF.local_ids in g.data for g in lst):
            out[GF.local_ids] = torch.cat([
                g.data[GF.local_ids] if GF.local_ids in g.data
                else torch.arange(len(g), device=g.positions.device)
                for g in lst
            ], dim=0)

        return ActivatedGaussians(out, specs)

    def to_ply(self, path: str):
        from plytorch import PointCloud

        PointCloud(
            points=self.data[GF.positions],
            colors=self.data[GF.features][:, :3].clamp(0, 1).mul(255).byte(),
        ).save(path)
