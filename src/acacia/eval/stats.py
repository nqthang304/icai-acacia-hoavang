"""Kiểm định cặp giữa các mô hình trên tập mảnh chung [paper III.C].

- Tập chung = mảnh test có pixel hợp lệ (sau khi che polygon vắt ranh giới) ở mọi mô hình.
- Mỗi mô hình: cộng ma trận nhầm lẫn theo mảnh qua 5 seed.
- Bootstrap theo mảnh (10.000 lần, seed 20261018): CI 95% của chênh lệch mIoU (tính từ CM cộng dồn).
- Wilcoxon signed-rank trên mIoU theo mảnh; hiệu chỉnh Holm cho mọi cặp.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .metrics import scores


def miou_from_cm(cm: np.ndarray) -> float:
    return scores(cm)["mIoU"]


def per_patch_miou(cms: np.ndarray) -> np.ndarray:
    return np.array([miou_from_cm(c) for c in cms])


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_tests(cm_by_model: dict[str, dict[int, np.ndarray]], n_boot=10000, seed=20261018):
    """cm_by_model: tên -> {chỉ số mảnh: CM [K,K] cộng qua seed}. Trả về (bảng cặp, danh sách mảnh chung)."""
    common = None
    for d in cm_by_model.values():
        ok = {p for p, c in d.items() if c.sum() > 0}
        common = ok if common is None else common & ok
    common = sorted(common or [])
    if not common:
        raise ValueError("không có mảnh test chung nào có pixel hợp lệ ở mọi mô hình")
    arr ={m: np.stack([d[p] for p in common]) for m, d in cm_by_model.items()}
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, len(common), size=(n_boot, len(common)))
    rows = []
    for a, b in itertools.combinations(arr, 2):
        A, B = arr[a], arr[b]
        diff = miou_from_cm(A.sum(0)) - miou_from_cm(B.sum(0))
        bd = np.array([miou_from_cm(A[ix].sum(0)) - miou_from_cm(B[ix].sum(0)) for ix in boots])
        pa, pb = per_patch_miou(A), per_patch_miou(B)
        ok = ~np.isnan(pa) & ~np.isnan(pb)
        try:
            p = float(wilcoxon(pa[ok], pb[ok]).pvalue)
        except ValueError:
            p = 1.0
        rows.append({"model_a": a, "model_b": b, "n_patches": len(common), "mIoU_a": miou_from_cm(A.sum(0)),
                     "mIoU_b": miou_from_cm(B.sum(0)), "diff": diff, "ci_low": float(np.percentile(bd, 2.5)),
                     "ci_high": float(np.percentile(bd, 97.5)), "p_wilcoxon": p})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p_wilcoxon.tolist())
    return df, common
