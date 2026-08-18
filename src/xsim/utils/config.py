import os
from pathlib import Path

import yaml

from jetcon.node import JetNode


# Where each machine records its own dataset locations. Gitignored, so a pull
# never rewrites it, and the tracked configs can reference "${env:NAME}" and
# stay identical everywhere.
DATASET_PATHS_FILE = Path(__file__).resolve().parents[3] / 'configs/data/dataset_paths.yaml'


def update_env(path: str | Path = DATASET_PATHS_FILE) -> dict[str, str]:
    """Export the dataset path variables from `path` that the environment lacks.

    The environment wins: a variable already exported is left alone, so a single
    path can be redirected for one run without editing the file. Returns just
    the variables this call set.

    A missing file is not an error — the variables may be exported already, and
    one that is genuinely absent fails later when the config is read, with
    jetcon naming both the variable and the config key that wanted it.
    """
    path = Path(path)
    if not path.exists():
        return {}

    with path.open() as file:
        entries = yaml.safe_load(file) or {}

    if not isinstance(entries, dict):
        raise ValueError(
            f'{path} must hold a flat "NAME: value" mapping, '
            f'got {type(entries).__name__}.'
        )

    applied = {}
    for name, value in entries.items():
        # An empty variable counts as unset, matching jetcon's "${env:NAME}".
        if os.environ.get(name):
            continue

        os.environ[name] = str(value)
        applied[name] = str(value)

    return applied


def parse_value(text: str):
    """Parse an override value with YAML rules (so 555 -> int, true -> bool, ...)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(cfg: JetNode, overrides: list[str]) -> JetNode:
    """Apply `dotted.key=value` overrides in place, returning the same node.

    Example: `provider.scene_idx=555` sets `cfg['provider']['scene_idx'] = 555`.
    Intermediate nodes are created on demand; the value is YAML-parsed for typing.
    """
    for override in overrides:
        if '=' not in override:
            raise ValueError(
                f'Invalid override "{override}", expected "dotted.key=value".'
            )
        key, _, value = override.partition('=')
        path = key.strip().split('.')

        node = cfg
        for part in path[:-1]:
            child = node.get(part)
            if not isinstance(child, JetNode):
                child = JetNode()
                node[part] = child
            node = child
        node[path[-1]] = parse_value(value)

    return cfg
