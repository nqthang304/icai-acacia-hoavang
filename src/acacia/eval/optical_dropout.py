"""Xóa quang học có kiểm soát [paper IV.C, Bảng IV].

Với D ∈ {0,1,2,4,8}, mỗi mảnh val bị xóa D quý (mask seed 20261018, dùng chung cho mọi mô hình).
Tại quý bị xóa (trên dữ liệu đã chuẩn hóa): 21 kênh quang học = 0, S2 count = -5, overlap = 1; SAR giữ nguyên.
Chỉ số chính (như bài báo): mIoU trên TOÀN chuỗi; retention R_D = 100 m_D / m_0.
Bổ sung (phản biện #27): mIoU riêng trên quý bị xóa và quý còn nguyên; lặp với nhiều mask seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.context import DataContext
from ..data.datasets import make_dataset
from ..models import input_channels
from ..train.trainer import load_trained, patch_sets
from .evaluator import evaluate
from .predict import predict


def make_masks(patches, T: int, D: int, seed: int) -> dict[int, set]:
    """Mỗi mảnh: hoán vị quý theo seed, lấy D quý đầu (lồng nhau: tập của D nhỏ nằm trong D lớn)."""
    rng = np.random.default_rng(seed)
    return {int(p): set(rng.permutation(T)[:D].tolist()) for p in patches}


def run_dropout(run_dirs: dict[str, str], levels=(0, 1, 2, 4, 8), mask_seeds=(20261018,), device=None, log=print):
    rows = []
    for name, rd in run_dirs.items():
        cfg, model, dev = load_trained(rd, device)
        ctx = DataContext(cfg)
        idx = patch_sets(ctx, cfg)["val"]
        T = len(ctx.quarters)
        K = cfg["data"]["num_classes"]
        for ms in mask_seeds:
            for D in levels:
                masks = make_masks(idx, T, D, ms)
                ds = make_dataset(ctx, cfg, idx, ctx.eval_labels, input_channels(cfg), optical_mask=masks)
                pred = predict(model, ds, dev, K, cfg["eval"]["batch_size"], amp=cfg["train"]["amp"])
                dropped = np.zeros((len(idx), T), bool)
                for i, p in enumerate(idx):
                    dropped[i, list(masks[int(p)])] = True
                r = evaluate(pred, ctx.eval_labels[idx], K, dropped=dropped)
                row = {"model": name, "mask_seed": ms, "D": D, "n_patches": len(idx),
                       "mIoU_sequence": r["scores"]["mIoU"],
                       "mIoU_dropped_quarters": r["by_dropout"]["dropped_quarters"]["scores"]["mIoU"] if D else np.nan,
                       "mIoU_intact_quarters": r["by_dropout"]["intact_quarters"]["scores"]["mIoU"]}
                rows.append(row)
                log(f"[dropout] {name} seed {ms} D={D}: seq {row['mIoU_sequence']:.4f} "
                    f"dropped {row['mIoU_dropped_quarters']:.4f}")
    df = pd.DataFrame(rows)
    base = df[df.D == 0].set_index(["model", "mask_seed"])["mIoU_sequence"]
    df["retention_pct"] = [100 * r.mIoU_sequence / base[(r.model, r.mask_seed)] for r in df.itertuples()]
    return df
