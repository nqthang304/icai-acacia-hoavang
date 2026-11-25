"""Mốc kiểm chứng: số liệu công bố trong bài báo mà code dựng lại phải khớp.

Chạy: pytest -q   (cần đã chạy scripts/01_build_cache.py; thiếu cache thì các test dữ liệu bị bỏ qua)
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acacia import paths  # noqa: E402
from acacia.config import load_config  # noqa: E402
from acacia.models import build_model, count_params  # noqa: E402

HAS_CACHE = (paths.cache_dir() / "index.json").exists()
needs_cache = pytest.mark.skipif(not HAS_CACHE, reason="chưa có cache dữ liệu")


@pytest.fixture(scope="module")
def ctx():
    from acacia.data.context import DataContext
    return DataContext(load_config())


# ---------------- số tham số (Bảng I, III) ----------------
@pytest.mark.parametrize("model,variant,target", [
    ("unet", {}, 1.95), ("unet", {"model": {"base": 12}}, 1.10), ("unet", {"model": {"base": 9}}, 0.62),
    ("quast", {}, 1.16), ("tsvit", {}, 1.03), ("utae", {}, 0.63), ("plain_tatt", {}, 0.63),
])
def test_param_counts(model, variant, target):
    m = build_model(load_config(model, variant))
    assert abs(count_params(m) / 1e6 - target) / target < 0.03


@pytest.mark.parametrize("model", ["utae", "tsvit", "quast", "plain_tatt"])
def test_temporal_output_per_quarter(model):
    m = build_model(load_config(model)).eval()
    x = torch.randn(1, 3, 27, 64, 64)
    y = m(x, torch.arange(3)[None], torch.tensor([[True, False, False]]))
    assert y.shape == (1, 3, 5, 64, 64) and torch.isfinite(y).all()


# ---------------- dữ liệu (mục III) ----------------
@needs_cache
def test_valid_pixels(ctx):
    assert int((ctx.labels != 255).sum()) == 11_891_184
    assert ctx.labels.size == 38_928_384


@needs_cache
def test_polygon_identities(ctx):
    assert len(np.unique(ctx.pids[ctx.pids >= 0])) == 521


@needs_cache
def test_split_anchors(ctx):
    s = ctx.split.summary()
    assert [s[k]["patches"] for k in ("train", "val", "test")] == [186, 54, 57]
    assert [s[k]["eligible"] for k in ("train", "val", "test")] == [170, 51, 43]
    assert abs(s["cross_split_polygons"] - 60) <= 4      # bài báo 60; cách chia tìm được: 64
    assert s["adjacent_cross_pairs"] == 205


@needs_cache
def test_buffer_and_common_set(ctx):
    sp = ctx.split
    tr = np.intersect1d(sp.idx("train", buffer_m=640), sp.idx("train", eligible_only=True))
    assert len(tr) == 143
    te = sp.idx("test", eligible_only=True)
    assert int((ctx.eval_labels[te] != 255).any(axis=(1, 2, 3)).sum()) == 34


@needs_cache
def test_every_class_in_every_split(ctx):
    for k in range(3):
        L = ctx.labels[ctx.split.assign == k]
        assert all((L == c).any() for c in range(5)), f"thiếu lớp ở tập {k}"


@needs_cache
def test_screening_candidates(ctx):
    from acacia.screening.screen import candidate_polygons
    assert len(candidate_polygons(ctx.labels, ctx.pids)) == 105


def test_dice_ignore_behaviour():
    """Bản gốc: xác suất ở pixel 255 vẫn nằm trong mẫu số Dice; bản sửa thì không."""
    from acacia.train.losses import soft_dice
    logits = torch.zeros(1, 5, 4, 4)
    y = torch.full((1, 4, 4), 255)
    y[0, 0, 0] = 2
    assert soft_dice(logits, y, 5, mask_ignore=False) > soft_dice(logits, y, 5, mask_ignore=True)
