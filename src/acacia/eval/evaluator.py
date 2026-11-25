"""Đánh giá từ dự đoán đã lưu: chỉ số chính + các phân tách bổ sung.

Phân tách bổ sung (không đổi giao thức chính):
  - per_patch: ma trận nhầm lẫn theo mảnh (cho kiểm định cặp, đánh giá trên tập mảnh chung)  [#31]
  - by_source: nguồn nhãn 1 gốc / 2 điều chỉnh / 3 vẽ thêm                                  [#35]
  - by_fill:   pixel S2 phải lấp (S2_nobs = 0) và không lấp                                    [#11]
  - by_regime: TOA (2018) và SR (2019+)
  - by_dropout: quý bị xóa quang học và quý còn nguyên (thí nghiệm optical dropout)          [#27]
"""
from __future__ import annotations

import numpy as np

from .metrics import confusion, scores

SOURCE_NAMES = {1: "original", 2: "adjusted", 3: "added"}


def evaluate(pred, labels, K, sources=None, filled=None, regimes=None, dropped=None) -> dict:
    """pred, labels, sources, filled: [n, T, H, W]; regimes: [T]; dropped: [n, T] bool."""
    valid_pred = pred != 255
    lab = np.where(valid_pred, labels, 255)
    cm = confusion(lab, pred, K)
    res = {"cm": cm, "scores": scores(cm)}
    res["per_patch_cm"] = np.stack([confusion(lab[i], pred[i], K) for i in range(len(pred))])
    res["per_quarter_cm"] = np.stack([confusion(lab[:, t], pred[:, t], K) for t in range(pred.shape[1])])

    def sub(mask):
        c = confusion(np.where(mask, lab, 255), pred, K)
        return {"cm": c, "scores": scores(c)}

    if sources is not None:
        res["by_source"] = {n: sub(sources == k) for k, n in SOURCE_NAMES.items()}
    if filled is not None:
        res["by_fill"] = {"filled": sub(filled), "observed": sub(~filled)}
    if regimes is not None:
        r = np.broadcast_to(np.asarray(regimes)[None, :, None, None], pred.shape)
        res["by_regime"] = {"TOA_2018": sub(r == 0), "SR_2019plus": sub(r == 1)}
    if dropped is not None:
        d = np.broadcast_to(dropped[:, :, None, None], pred.shape)
        res["by_dropout"] = {"dropped_quarters": sub(d), "intact_quarters": sub(~d)}
    return res


def to_jsonable(res: dict) -> dict:
    """Bỏ các mảng lớn, đổi cm sang list để ghi JSON."""
    out = {}
    for k, v in res.items():
        if k in ("per_patch_cm", "per_quarter_cm"):
            continue
        if isinstance(v, dict):
            out[k] = to_jsonable(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out
