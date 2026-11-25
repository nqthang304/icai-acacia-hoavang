"""Bước 5: kiểm định cặp (bootstrap, Wilcoxon, Holm) giữa 4 mô hình chính trên tập mảnh test chung.
Bổ sung: mIoU của từng mô hình trên tập chung và trên 43 mảnh đủ điều kiện (phản biện #31).
Đầu ra: work/results/paired_tests.csv, common_set_scores.csv
"""
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from acacia import paths
from acacia.config import load_config
from acacia.eval.metrics import scores
from acacia.eval.stats import paired_tests

MODELS = {"U-Net": "unet", "QuaST": "quast", "TSViT": "tsvit", "U-TAE": "utae"}
cfg = load_config()
root = paths.runs_dir() / "main_test"
cms = {}
for name, run in MODELS.items():
    acc = {}
    for s in cfg["seeds"]:
        f = root / run / f"s{s}" / "cm_test.npz"
        if not f.exists():
            print("thiếu", f); continue
        z = np.load(f)
        for p, c in zip(z["patches"], z["per_patch"]):
            acc[int(p)] = acc.get(int(p), 0) + c
    if acc:
        cms[name] = acc
df, common = paired_tests(cms)
out = paths.work_dir() / "results"; out.mkdir(exist_ok=True)
df.to_csv(out / "paired_tests.csv", index=False)
print(f"tập mảnh chung: {len(common)} mảnh (bài báo: 34)")
print(df.round(4).to_string(index=False))
# mIoU từng mô hình trên tập chung, tính lại theo từng seed rồi lấy trung bình ± SD
rows = []
for name, run in MODELS.items():
    vals = {"common": [], "all": []}
    for s in cfg["seeds"]:
        f = root / run / f"s{s}" / "cm_test.npz"
        if not f.exists(): continue
        z = np.load(f)
        m = np.isin(z["patches"], common)
        vals["common"].append(scores(z["per_patch"][m].sum(0))["mIoU"])
        vals["all"].append(scores(z["per_patch"].sum(0))["mIoU"])
    rows.append({"model": name, "n_seeds": len(vals["all"]),
                 "mIoU_own_patches": np.mean(vals["all"]), "sd_own": np.std(vals["all"], ddof=1) if len(vals["all"]) > 1 else np.nan,
                 "mIoU_common_patches": np.mean(vals["common"]), "sd_common": np.std(vals["common"], ddof=1) if len(vals["common"]) > 1 else np.nan})
pd.DataFrame(rows).to_csv(out / "common_set_scores.csv", index=False)
(out / "common_patches.json").write_text(json.dumps([int(p) for p in common]))
print(pd.DataFrame(rows).round(4).to_string(index=False))
