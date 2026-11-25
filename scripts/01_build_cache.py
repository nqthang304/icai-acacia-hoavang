"""Bước 1: GeoTIFF -> cache numpy. Dùng: python scripts/01_build_cache.py [--overwrite]"""
import argparse

import _bootstrap  # noqa: F401
from acacia import paths
from acacia.data.cache import build_cache

ap = argparse.ArgumentParser()
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()
idx = build_cache(paths.data_dir(), paths.cache_dir(), overwrite=a.overwrite)
print(f"{len(idx['patch_ids'])} mảnh x {len(idx['quarters'])} quý -> {paths.cache_dir()}")
print(f"(mảnh, quý) có giá trị không hữu hạn: {len(idx['nonfinite'])}")
