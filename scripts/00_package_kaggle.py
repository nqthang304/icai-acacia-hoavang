"""Đóng gói 2 file zip để upload làm Kaggle Dataset (chạy trên máy, sau bước 01).

  kaggle_upload/acacia-hoavang-code.zip  : src/, scripts/, configs/, tests/, notebooks/, *.md, requirements.txt
  kaggle_upload/acacia-hoavang-data.zip  : data/{grid.csv, metadata.json, README.md, dem/, features_quy/} + cache/
"""
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
from acacia import paths

ROOT = paths.ROOT
out = ROOT / "kaggle_upload"
out.mkdir(exist_ok=True)


def add_tree(z, src: Path, arc: str, skip=("__pycache__",)):
    for f in sorted(src.rglob("*")):
        if f.is_file() and not any(s in f.parts for s in skip):
            z.write(f, f"{arc}/{f.relative_to(src).as_posix()}")


with zipfile.ZipFile(out / "acacia-hoavang-code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for d in ("src", "scripts", "configs", "tests", "notebooks"):
        add_tree(z, ROOT / d, d)
    for f in ("README.md", "ASSUMPTIONS.md", "requirements.txt"):
        if (ROOT / f).exists():
            z.write(ROOT / f, f)
print("->", out / "acacia-hoavang-code.zip")

data, cache = paths.data_dir(), paths.cache_dir()
with zipfile.ZipFile(out / "acacia-hoavang-data.zip", "w", zipfile.ZIP_STORED, allowZip64=True) as z:
    for f in ("grid.csv", "metadata.json", "README.md"):
        z.write(data / f, f"data/{f}")
    add_tree(z, data / "dem", "data/dem")
    add_tree(z, data / "features_quy", "data/features_quy")
    for f in ("X.npy", "Y.npy", "dem.npy", "index.json"):
        z.write(cache / f, f"cache/{f}")
print("->", out / "acacia-hoavang-data.zip")
