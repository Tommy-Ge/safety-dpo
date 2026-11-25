"""YAML config loading with dotted-key command-line overrides (``-o training.beta=0.05``)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _parse_scalar(text: str) -> Any:
    # yaml gives us ints / floats / bools / null / lists for free
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like a.b.c=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
            if not isinstance(node, (dict, list)):
                raise ValueError(f"cannot descend into scalar at {p!r} in {key!r}")
        last = parts[-1]
        if isinstance(node, list):
            node[int(last)] = _parse_scalar(value)
        else:
            node[last] = _parse_scalar(value)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return apply_overrides(cfg, overrides)
