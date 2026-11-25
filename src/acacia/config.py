"""Nạp config YAML theo tầng: base.yaml <- models/<model>.yaml <- experiments/<exp>.yaml <- --set a.b=c."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .paths import ROOT

CONFIG_DIR = ROOT / "configs"


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf8")) or {}


def parse_set(items: list[str] | None) -> dict:
    """['train.max_epochs=3', 'model.base=12'] -> dict lồng nhau (giá trị parse bằng YAML)."""
    out: dict = {}
    for it in items or []:
        key, val = it.split("=", 1)
        d = out
        *parents, last = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(val)
    return out


def load_config(model: str | None = None, variant: dict | None = None, overrides: list[str] | None = None) -> dict:
    cfg = _read(CONFIG_DIR / "base.yaml")
    if model:
        cfg = deep_update(cfg, _read(CONFIG_DIR / "models" / f"{model}.yaml"))
    cfg = deep_update(cfg, variant or {})
    cfg = deep_update(cfg, parse_set(overrides))
    return cfg


def load_experiment(name: str) -> dict:
    return _read(CONFIG_DIR / "experiments" / f"{name}.yaml")
