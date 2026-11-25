"""Chỉ số từ ma trận nhầm lẫn cộng dồn [paper III.C].

IoU_c = TP / (TP + FP + FN) trên pixel có nhãn hợp lệ; mIoU = trung bình 5 lớp; Macro-F1 = trung bình F1.
Một ma trận cho mỗi lần chạy (mô hình x seed) - KHÔNG phải trung bình IoU theo mảnh.
"""
from __future__ import annotations

import numpy as np

IGNORE = 255


def confusion(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> np.ndarray:
    """Hàng = nhãn thật, cột = dự đoán."""
    m = y_true != IGNORE
    return np.bincount(K * y_true[m].astype(np.int64) + y_pred[m].astype(np.int64), minlength=K * K).reshape(K, K)


def scores(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        recall = tp / (tp + fn)
    present = ~np.isnan(iou)
    return {
        "mIoU": float(iou[present].mean()) if present.any() else float("nan"),
        "macro_F1": float(f1[present].mean()) if present.any() else float("nan"),
        "pixel_acc": float(tp.sum() / max(cm.sum(), 1)),
        "IoU": iou.tolist(),
        "F1": f1.tolist(),
        "recall": recall.tolist(),
        "support": cm.sum(1).astype(int).tolist(),
    }
