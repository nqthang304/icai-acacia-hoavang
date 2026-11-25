"""Bước 7: sàng lọc ứng viên khảo sát (Bảng VI, Hình 8-9) + kiểm tra DEM (phản biện #39).
Dùng: python scripts/07_screening.py [--min-pixels 25]
Đầu ra: work/results/screening_*.csv
"""
import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from acacia import paths
from acacia.config import load_config
from acacia.data.context import DataContext
from acacia.screening.screen import candidate_polygons, group_profiles, indicators, score, sensitivity, GROUPS

ap = argparse.ArgumentParser()
ap.add_argument("--min-pixels", type=int, default=25)
a = ap.parse_args()
cfg = load_config()
ctx = DataContext(cfg)
out = paths.work_dir() / "results"; out.mkdir(exist_ok=True)
cands = candidate_polygons(ctx.labels, ctx.pids, a.min_pixels)
print(f"{len(cands)} polygon từng là Keo (≥ {a.min_pixels} pixel); bài báo: 105")
df = pd.DataFrame([indicators(ctx, ctx.quarters, p) for p in cands])
sc = score(df)
names = cfg["data"]["class_names"]
sc["latest_label_name"] = sc.latest_label.map(lambda v: names[int(v)] if pd.notna(v) else "unlabeled")
sc.sort_values("score", ascending=False).to_csv(out / "screening_polygons.csv", index=False)
prof = group_profiles(sc)
prof.to_csv(out / "screening_groups.csv")
sens = sensitivity(df)
sens.to_csv(out / "screening_sensitivity.csv", index=False)
pr = sc[sc.group == GROUPS[0]]
print(prof.round(3))
print("Nhãn gần nhất của nhóm Priority:", pr.latest_label_name.value_counts().to_dict())
bad = sc[(sc.dem_zero_frac > 0) | sc.elevation.isna()]
print(f"Kiểm tra DEM: {len(bad)} polygon có pixel độ cao = độ dốc = 0")
print(sens.round(1).to_string(index=False))
