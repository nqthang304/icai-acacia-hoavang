"""Bước 4: thí nghiệm xóa quang học (Bảng IV) từ checkpoint đã có.
Dùng: python scripts/04_optical_dropout.py [--extra]   (--extra: lặp thêm nhiều mask seed)
Đầu ra: work/results/optical_dropout.csv
"""
import argparse

import _bootstrap  # noqa: F401
from acacia import paths
from acacia.config import load_experiment
from acacia.eval.optical_dropout import run_dropout

ap = argparse.ArgumentParser()
ap.add_argument("--extra", action="store_true")
ap.add_argument("--device")
a = ap.parse_args()
e = load_experiment("optical_dropout")
seeds = e["mask_seeds"] + (e["extra_mask_seeds"] if a.extra else [])
runs = {k: paths.runs_dir() / v for k, v in e["models"].items()}
missing = [k for k, v in runs.items() if not (v / "best.pt").exists()]
if missing:
    raise SystemExit(f"Thiếu checkpoint: {missing} - chạy main_test và modality (seed 101) trước")
df = run_dropout(runs, e["levels"], seeds, a.device)
out = paths.work_dir() / "results"
out.mkdir(exist_ok=True)
df.to_csv(out / "optical_dropout.csv", index=False)
print(df.pivot_table(index="model", columns="D", values="mIoU_sequence").round(3))
