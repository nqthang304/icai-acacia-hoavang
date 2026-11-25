"""Loss = 0,6 * weighted CE + 0,4 * soft Dice (+ monotonicity penalty cho QuaST) [paper IV.B].

Chi tiết bản gốc (bài báo nêu rõ): ở Dice, xác suất dự đoán tại pixel bị bỏ qua (255) VẪN nằm trong
mẫu số theo lớp, còn target và intersection được che. `dice_mask_ignore=False` tái hiện đúng hành vi đó;
`True` là bản đã sửa (phản biện #25).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = 255


def soft_dice(logits, y, K, mask_ignore: bool, smooth: float = 1.0):
    """logits [N,K,H,W], y [N,H,W]."""
    prob = logits.float().softmax(1)
    valid = (y != IGNORE)
    onehot = F.one_hot(y.clamp(max=K - 1), K).permute(0, 3, 1, 2).float() * valid[:, None]
    dims = (0, 2, 3)
    inter = (prob * onehot).sum(dims)
    p_sum = (prob * valid[:, None]).sum(dims) if mask_ignore else prob.sum(dims)
    dice = (2 * inter + smooth) / (p_sum + onehot.sum(dims) + smooth)
    return 1 - dice.mean()


class SegLoss(nn.Module):
    def __init__(self, cfg: dict, class_weight):
        super().__init__()
        lc = cfg["loss"]
        self.K = cfg["data"]["num_classes"]
        self.wce, self.wdice = lc["ce_weight"], lc["dice_weight"]
        self.mask_ignore, self.smooth = lc["dice_mask_ignore"], lc["dice_smooth"]
        self.mono = lc.get("mono_weight", 0.0)
        self.register_buffer("cw", torch.as_tensor(class_weight, dtype=torch.float32))

    def forward(self, logits, y, aux=None, pad=None):
        if logits.dim() == 5:                       # [B,T,K,H,W] -> [B*T,K,H,W]
            logits, y = logits.flatten(0, 1), y.flatten(0, 1)
        parts = {}
        if (y != IGNORE).any():
            parts["ce"] = F.cross_entropy(logits.float(), y, weight=self.cw, ignore_index=IGNORE)
        else:
            parts["ce"] = logits.sum() * 0.0
        parts["dice"] = soft_dice(logits, y, self.K, self.mask_ignore, self.smooth)
        total = self.wce * parts["ce"] + self.wdice * parts["dice"]
        if self.mono > 0 and aux is not None and aux.get("gates") is not None:
            from ..models.quast import monotonicity_penalty
            valid = None if pad is None else ~pad
            parts["mono"] = monotonicity_penalty(aux["gates"].float(), aux["q"].float(), valid)
            total = total + self.mono * parts["mono"]
        return total, {k: float(v.detach()) for k, v in parts.items()}
