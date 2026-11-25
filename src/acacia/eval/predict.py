"""Chạy mô hình trên một tập mảnh, ghép dự đoán thành mảng [n_mảnh, T, H, W] (255 = không dự đoán)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.datasets import SequenceDataset, SingleQuarterDataset


@torch.no_grad()
def predict(model, ds, device, K=5, batch_size=4, num_workers=0, amp=True, return_probs=False):
    model.eval()
    P = ds.patches
    pos = {int(p): i for i, p in enumerate(P)}
    T, H, W = ds.labels.shape[1:]
    out = np.full((len(P), T, H, W), 255, np.uint8)
    probs = np.zeros((len(P), T, K, H, W), np.float16) if return_probs else None
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    use_amp = amp and device.type == "cuda"
    for b in dl:
        x = b["x"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp):
            if isinstance(ds, SingleQuarterDataset):
                logits = model(x)[:, None]                            # [B,1,K,H,W]
            else:
                logits = model(x, b["t"].to(device), b["pad"].to(device))
        pr = logits.float().softmax(2)
        lab = pr.argmax(2).cpu().numpy().astype(np.uint8)
        for i in range(len(b["p"])):
            pi = pos[int(b["p"][i])]
            ts = b["t"][i].numpy()
            if isinstance(ds, SequenceDataset) and ds.window is not None:
                sel = [len(ts) - 1]                                   # chỉ quý mục tiêu
            elif isinstance(ds, SingleQuarterDataset):
                sel = [0]
            else:
                sel = list(range(len(ts)))
            for j in sel:
                out[pi, ts[j]] = lab[i, j]
                if probs is not None:
                    probs[pi, ts[j]] = pr[i, j].cpu().numpy()
    return (out, probs) if return_probs else out
