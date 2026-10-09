"""Load a run config, resolving the `inherit` chain.

Each run is described by one YAML file. A file may name a parent with
`inherit: <path>`; the parent is loaded first and the child's values are
laid on top of it. Chains can be several files long, e.g.

    configs/part2/lstm_glove_frozen.yaml -> configs/part1/lstm.yaml -> configs/base.yaml

Print a fully resolved config with:

    python -m papertrail.config configs/part2/lstm_glove_frozen.yaml
"""
import sys
from pathlib import Path

import yaml

# `inherit` paths are written relative to the repo root (e.g. "configs/base.yaml"),
# so resolve them from there rather than from wherever the command was run.
REPO_ROOT = Path(__file__).resolve().parent.parent


def merge(parent: dict, child: dict) -> dict:
    """Return `parent` with `child` laid on top, without modifying either.

    Nested dicts are merged key by key, so a child that sets only
    `model.type` keeps the parent's `model.emb_dim`. Anything else,
    lists included, is replaced whole.
    """
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict:
    """Load the YAML file at `path` and everything it inherits from."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    parent_path = config.pop("inherit", None)
    if parent_path is None:
        return config
    return merge(load_config(parent_path), config)


if __name__ == "__main__":
    yaml.safe_dump(load_config(sys.argv[1]), sys.stdout, sort_keys=False)
