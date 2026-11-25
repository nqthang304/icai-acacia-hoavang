"""Bước 6: phân loại cấp polygon - quý (Bảng V) + bảng số mẫu theo lớp + ma trận nhầm lẫn.
Dùng: python scripts/06_polygon.py [--quast-run main_test/quast/s101] [--include-dem]
Đầu ra: work/results/polygon_*.csv, polygon_cm.json
"""
import argparse
import json

import _bootstrap  # noqa: F401
from acacia import paths
from acacia.config import load_config
from acacia.data.context import DataContext
from acacia.polygon.classify import (assign_split, load_pixels, polygon_features, quast_polygon_features,
                                     run_classic, run_quast_mlp, support_table)

ap = argparse.ArgumentParser()
ap.add_argument("--quast-run", default="main_test/quast/s101")
ap.add_argument("--include-dem", action="store_true")
ap.add_argument("--device")
a = ap.parse_args()
cfg = load_config()
ctx = DataContext(cfg)
out = paths.work_dir() / "results"; out.mkdir(exist_ok=True)

meta = json.loads((paths.data_dir() / "metadata.json").read_text(encoding="utf8"))
channels = meta["kenh_anh"]
px = load_pixels(paths.data_dir())
feat = assign_split(polygon_features(px, channels, a.include_dem), ctx)
names = cfg["data"]["class_names"]
sup = support_table(feat, names)
sup.to_csv(out / "polygon_support.csv")
print("Số mẫu polygon-quý theo lớp:\n", sup)

res, search, cms = run_classic(feat)
qr = paths.runs_dir() / a.quast_run
if (qr / "best.pt").exists():
    zf = quast_polygon_features(feat, ctx, qr, a.device)
    row, cm = run_quast_mlp(zf)
    res.loc[len(res)] = row
    cms["QuaST encoder + MLP"] = cm
else:
    print("chưa có checkpoint QuaST -> bỏ qua QuaST encoder + MLP")
res.to_csv(out / "polygon_results.csv", index=False)
search.to_csv(out / "polygon_hparam_search.csv", index=False)
(out / "polygon_cm.json").write_text(json.dumps({m: {s: c.tolist() for s, c in d.items()} for m, d in cms.items()}))
print(res.round(3).to_string(index=False))
