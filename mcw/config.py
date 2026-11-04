"""YAML configs with single inheritance: a run config names its parent in ``base:`` and
overrides only what differs (here: the ``windowing:`` block and ``run_name``)."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _expand(x):
    if isinstance(x, dict):
        return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v) for v in x]
    if isinstance(x, str):
        return os.path.expandvars(x)
    return x


def load_config(path: str) -> dict:
    p = Path(path)
    cfg = yaml.safe_load(p.read_text()) or {}
    parent = cfg.pop("base", None)
    if parent:
        cfg = deep_merge(load_config(str(p.parent / parent)), cfg)
    return _expand(cfg)


def set_by_dotted(cfg: dict, dotted: str, value: str) -> None:
    """``--set training.epochs=2`` style override (value parsed as YAML)."""
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)
