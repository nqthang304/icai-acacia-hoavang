"""Chạy một nhóm thực nghiệm: huấn luyện + đánh giá, bỏ qua lần chạy đã xong (chạy tiếp được trên Kaggle).

Thư mục: work/runs/<thực nghiệm>/<cấu hình>/s<seed>/
  config.yaml, log.jsonl, best.pt, summary.json
  pred_<split>.npz   dự đoán uint8 [n_mảnh, T, H, W] + chỉ số mảnh
  eval_<split>.json  chỉ số + các phân tách bổ sung
  cm_<split>.npz     ma trận nhầm lẫn theo mảnh / theo quý
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import paths
from .config import deep_update, load_config, load_experiment
from .data.context import DataContext
from .data.datasets import make_dataset
from .data.normalize import regime_of_quarters
from .eval.evaluator import evaluate, to_jsonable
from .eval.predict import predict
from .models import input_channels
from .train.trainer import load_trained, patch_sets, train_run


def run_dir(exp: str, run: str, seed: int, cfg=None) -> Path:
    return paths.runs_dir(cfg) / exp / run / f"s{seed}"


def eval_run(rd: Path, splits, device=None, log=print):
    cfg, model, device = load_trained(rd, device)
    ctx = DataContext(cfg)
    ps = patch_sets(ctx, cfg)
    K = cfg["data"]["num_classes"]
    reg = regime_of_quarters(ctx.quarters, cfg["data"]["toa_years"])
    for sp in splits:
        idx = ps[sp]
        ds = make_dataset(ctx, cfg, idx, ctx.eval_labels, input_channels(cfg))
        pred = predict(model, ds, device, K, cfg["eval"]["batch_size"], amp=cfg["train"]["amp"])
        res = evaluate(pred, ctx.eval_labels[idx], K, ctx.sources[idx], ctx.s2_filled[idx], reg)
        np.savez_compressed(rd / f"pred_{sp}.npz", pred=pred, patches=idx)
        np.savez_compressed(rd / f"cm_{sp}.npz", per_patch=res["per_patch_cm"], per_quarter=res["per_quarter_cm"],
                            patches=idx)
        (rd / f"eval_{sp}.json").write_text(json.dumps(to_jsonable(res), indent=1))
        log(f"[{rd.parent.name}/{rd.name}] {sp}: mIoU {res['scores']['mIoU']:.4f} "
            f"macro-F1 {res['scores']['macro_F1']:.4f} ({len(idx)} mảnh)")


def run_experiment(name: str, seeds=None, only=None, device=None, overrides=None, eval_only=False,
                   force=False, log=print):
    exp = load_experiment(name)
    base_seeds = load_config()["seeds"]
    seeds = seeds or exp.get("seeds", base_seeds)
    for run, spec in exp["runs"].items():
        if only and run not in only:
            continue
        variant = deep_update(exp.get("overrides", {}), spec.get("variant", {}))
        cfg = load_config(spec["model"], variant, overrides)
        for s in seeds:
            rd = run_dir(name, run, s, cfg)
            done = (rd / "summary.json").exists()
            if not eval_only and (force or not done):
                train_run(cfg, s, rd, device, log)
            elif done:
                log(f"[{name}/{run}/s{s}] đã huấn luyện - bỏ qua")
            if (rd / "best.pt").exists():
                todo = [sp for sp in exp.get("eval_splits", ["val"]) if force or eval_only
                        or not (rd / f"eval_{sp}.json").exists()]
                if todo:
                    eval_run(rd, todo, device, log)
