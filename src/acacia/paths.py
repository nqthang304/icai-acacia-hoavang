"""Đường dẫn dữ liệu / thư mục làm việc, dùng chung cho máy cục bộ và Kaggle.

Thứ tự ưu tiên: biến môi trường ACACIA_DATA / ACACIA_WORK > giá trị trong config > mặc định.
Trên Kaggle: ACACIA_DATA=/kaggle/input/<dataset>/data, ACACIA_WORK=/kaggle/working.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_dir(cfg: dict | None = None) -> Path:
    p = os.environ.get("ACACIA_DATA") or (cfg or {}).get("paths", {}).get("data_dir") or ROOT / "data"
    return Path(p)


def work_dir(cfg: dict | None = None) -> Path:
    p = os.environ.get("ACACIA_WORK") or (cfg or {}).get("paths", {}).get("work_dir") or ROOT / "work"
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir(cfg: dict | None = None) -> Path:
    p = os.environ.get("ACACIA_CACHE") or work_dir(cfg) / "cache"
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs_dir(cfg: dict | None = None) -> Path:
    p = work_dir(cfg) / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p
