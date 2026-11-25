"""Bước 2: chia tập, thống kê chuẩn hóa, KS 2018-2019, bảng lớp theo tập, đối chiếu mốc bài báo.

Dùng: python scripts/02_prepare.py [--search-split]
Đầu ra: work/prepared/{norm_stats.json, anchors.json, class_stats_by_split.csv}
"""
import argparse
import json

import numpy as np

import _bootstrap  # noqa: F401
from acacia.config import load_config
from acacia.data.context import DataContext
from acacia.data.normalize import fit_stats, ks_2018_2019
from acacia.data.split import class_stats, search_split
from acacia.paths import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--search-split", action="store_true", help="tìm lại configs/split_blocks.json")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()

cfg = load_config(overrides=a.set)
ctx = DataContext(cfg)
ctx.prepared.mkdir(parents=True, exist_ok=True)

if a.search_split:
    mp, cost = search_split(ctx.grid, ctx.labels, ctx.pids, cfg["split"]["block_size"])
    (ROOT / cfg["split"]["file"]).write_text(json.dumps(mp, indent=0))
    print("split mới, cost =", cost)

sp = ctx.split
train = sp.idx("train")
print("fit chuẩn hóa trên", len(train), "mảnh train ...")
stats = fit_stats(ctx.cache.X, train, ctx.quarters, cfg)
(ctx.prepared / "norm_stats.json").write_text(json.dumps(stats))
ks = ks_2018_2019(ctx.cache.X, train, ctx.quarters, tuple(cfg["data"]["optical"]))

L = ctx.labels
anchors = {
    "valid_pixel_quarters": {"ours": int((L != 255).sum()), "paper": 11891184},
    "total_pixel_quarters": {"ours": int(L.size), "paper": 38928384},
    "polygon_identities": {"ours": int(len(np.unique(ctx.pids[ctx.pids >= 0]))), "paper": 521},
    "labeled_patches_manual": {"ours": int(ctx.grid.has_manual_label.sum()), "paper": 201},
    "split": sp.summary(),
    "split_paper": {"train": [186, 170], "val": [54, 51], "test": [57, 43],
                    "cross_split_polygons": 60, "adjacent_cross_pairs": 205},
    "buffer_train_eligible": {
        str(b): int(len(np.intersect1d(sp.idx("train", buffer_m=b), sp.idx("train", eligible_only=True))))
        for b in (0, 640, 1280)},
    "buffer_paper": {"0": 170, "640": 143},
    "test_patches_with_valid_eval_pixels": {
        "ours": int((ctx.eval_labels[sp.idx("test", eligible_only=True)] != 255).any(axis=(1, 2, 3)).sum()),
        "paper_common_set": 34},
    "ks_2018_2019_mean": {"ours": ks["mean_ks"], "paper": 0.410},
}
(ctx.prepared / "anchors.json").write_text(json.dumps(anchors, indent=1))
cs = class_stats(sp, L, ctx.pids, cfg["data"]["class_names"])
cs.to_csv(ctx.prepared / "class_stats_by_split.csv", index=False)
print(json.dumps(anchors, indent=1))
print(cs.pivot(index="class", columns="split", values=["pixel_quarters", "polygons"]))
