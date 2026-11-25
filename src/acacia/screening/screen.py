"""Sàng lọc ứng viên khảo sát thực địa cho rừng gỗ lớn [paper IV.D, V.E; thesis 3.6].

Đối tượng: polygon từng mang nhãn Keo ít nhất một lần. Lọc ≥ 25 pixel cho ra đúng 105 polygon như
bài báo [ASSUMPTION: tài liệu không ghi tiêu chí lọc; xem ASSUMPTIONS.md].

Chỉ báo (tính trên chuỗi trung bình theo polygon, giá trị ảnh thô):
  G  growth      : độ dốc hồi quy NDVI (theo quý) từ sự kiện trồng lại suy ra tới cuối chuỗi
  T  terrain     : độ dốc trung bình (SRTM); độ cao chỉ mô tả [thesis Bảng 3.3]
  D  disturbance : mức giảm NDVI quanh 2020Q4
  V  vigor       : NDRE cực đại
Điểm: G, V theo percentile-rank; T, D theo percentile-rank ngược (dốc/giảm ít -> điểm cao).
  S = 100 (0,35 G + 0,25 T + 0,15 D + 0,25 V); G thiếu -> 0,5; D thiếu -> 0 trước khi xếp hạng [thesis].
  Nhóm: S ≥ 60 Priority; 40 ≤ S < 60 Conversion consideration; S < 40 Maintain short rotation [thesis].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

NDVI, NDRE = 12, 15
GROUPS = ("Priority", "Conversion consideration", "Maintain short rotation")

BASE_WEIGHTS = (0.35, 0.25, 0.15, 0.25)  # [paper] G, T, D, V
# [ASSUMPTION] 7 bộ trọng số thay thế (bài báo không công bố)
ALT_WEIGHTS = {
    "equal": (0.25, 0.25, 0.25, 0.25),
    "growth_heavy": (0.50, 0.20, 0.10, 0.20),
    "terrain_heavy": (0.25, 0.45, 0.10, 0.20),
    "disturbance_heavy": (0.30, 0.20, 0.30, 0.20),
    "vigor_heavy": (0.30, 0.20, 0.10, 0.40),
    "no_disturbance": (0.40, 0.30, 0.00, 0.30),
    "no_terrain": (0.45, 0.00, 0.20, 0.35),
}


def candidate_polygons(labels, pids, min_pixels=25, acacia=2) -> list[int]:
    ac = np.unique(pids[(labels == acacia) & (pids >= 0)])
    T = labels.shape[1]
    size = {int(a): max(int((pids[:, t] == a).sum()) for t in range(T)) for a in ac}
    return sorted(a for a, n in size.items() if n >= min_pixels)


def polygon_series(X, pids, labels, pid: int) -> dict:
    """Chuỗi NDVI/NDRE trung bình theo quý trên pixel của polygon (mặt nạ của từng quý; thiếu -> hợp các quý)."""
    P, T = pids.shape[:2]
    union = (pids == pid).any(axis=1)                      # [P,H,W]
    patches = np.where(union.any(axis=(1, 2)))[0]
    ndvi, ndre, lab = np.full(T, np.nan), np.full(T, np.nan), np.full(T, -1)
    for t in range(T):
        vals_v, vals_r, labs = [], [], []
        for p in patches:
            m = pids[p, t] == pid
            if not m.any():
                m = union[p]
            else:
                labs.append(labels[p, t][m])
            vals_v.append(np.asarray(X[p, t, NDVI], np.float32)[m])
            vals_r.append(np.asarray(X[p, t, NDRE], np.float32)[m])
        ndvi[t] = np.nanmean(np.concatenate(vals_v))
        ndre[t] = np.nanmean(np.concatenate(vals_r))
        if labs:
            ll = np.concatenate(labs)
            ll = ll[ll != 255]
            if len(ll):
                lab[t] = np.bincount(ll).argmax()
    return {"ndvi": ndvi, "ndre": ndre, "label": lab, "patches": patches, "union": union}


def replant_index(ndvi: np.ndarray, lab: np.ndarray, drop=0.15) -> int | None:
    """Sự kiện trồng lại [ASSUMPTION]: lần chuyển Đất trống (1) -> Keo (2) cuối cùng theo nhãn;
    nếu không có, đáy NDVI cuối cùng sau khi giảm ≥ `drop` so với đỉnh trước đó."""
    tr = [t for t in range(1, len(lab)) if lab[t] == 2 and lab[t - 1] == 1]
    if tr:
        return tr[-1] - 1
    best = None
    for t in range(1, len(ndvi)):
        if np.nanmax(ndvi[:t]) - ndvi[t] >= drop and (t + 1 == len(ndvi) or ndvi[t] <= ndvi[t + 1]):
            best = t
    return best


def indicators(ctx, quarters, pid: int) -> dict:
    s = polygon_series(ctx.cache.X, ctx.pids, ctx.labels, pid)
    ndvi, ndre, lab = s["ndvi"], s["ndre"], s["label"]
    r = replant_index(ndvi, lab)
    G = np.nan
    if r is not None and len(ndvi) - r >= 4:
        t = np.arange(r, len(ndvi))
        ok = ~np.isnan(ndvi[r:])
        if ok.sum() >= 4:
            G = float(np.polyfit(t[ok], ndvi[r:][ok], 1)[0])
    qi = {q: i for i, q in enumerate(quarters)}
    pre = np.nanmean(ndvi[[qi["2020Q2"], qi["2020Q3"]]])
    post = np.nanmin(ndvi[[qi["2020Q4"], qi["2021Q1"]]])
    D = float(pre - post) if np.isfinite(pre) and np.isfinite(post) else np.nan
    dem = ctx.cache.dem
    el, sl = [], []
    for p in s["patches"]:
        m = s["union"][p]
        el.append(dem[p, 0][m])
        sl.append(dem[p, 1][m])
    el, sl = np.concatenate(el), np.concatenate(sl)
    latest = int(lab[-1]) if lab[-1] >= 0 else None
    return {"pid": pid, "growth": G, "replant_q": quarters[r] if r is not None else None,
            "slope": float(np.mean(sl)), "elevation": float(np.mean(el)),
            "dem_zero_frac": float(np.mean((el == 0) & (sl == 0))),
            "disturbance": D, "vigor": float(np.nanmax(ndre)), "n_pixels": int(s["union"].sum()),
            "latest_label": latest, "patches": ",".join(ctx.grid.patch_id[s["patches"]])}


def _pct(v):
    return (rankdata(v, method="average") - 1) / max(len(v) - 1, 1)


def score(df: pd.DataFrame, w=BASE_WEIGHTS, thresholds=(40, 60)) -> pd.DataFrame:
    df = df.copy()
    g = df.growth.to_numpy(float)
    G = np.full(len(df), 0.5)
    ok = ~np.isnan(g)
    G[ok] = _pct(g[ok])
    Tt = 1 - _pct(df.slope.to_numpy(float))
    d = np.nan_to_num(df.disturbance.to_numpy(float), nan=0.0)      # thiếu -> 0 trước khi xếp hạng
    Dd = 1 - _pct(d)
    V = _pct(df.vigor.to_numpy(float))
    df["G"], df["T"], df["D"], df["V"] = G, Tt, Dd, V
    df["score"] = 100 * (w[0] * G + w[1] * Tt + w[2] * Dd + w[3] * V)
    lo, hi = thresholds
    df["group"] = np.where(df.score >= hi, GROUPS[0], np.where(df.score >= lo, GROUPS[1], GROUPS[2]))
    return df


def sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    base = score(df)
    rows = [{"scheme": "baseline", "weights": BASE_WEIGHTS, "n_priority": int((base.group == GROUPS[0]).sum()),
             "agreement_pct": 100.0}]
    for name, w in ALT_WEIGHTS.items():
        s = score(df, w)
        rows.append({"scheme": name, "weights": w, "n_priority": int((s.group == GROUPS[0]).sum()),
                     "agreement_pct": 100 * float((s.group.values == base.group.values).mean())})
    return pd.DataFrame(rows)


def group_profiles(scored: pd.DataFrame) -> pd.DataFrame:
    g = scored.groupby("group").agg(N=("pid", "size"), growth=("growth", "mean"), elevation=("elevation", "mean"),
                                    slope=("slope", "mean"), score=("score", "mean"))
    g["pct"] = 100 * g.N / g.N.sum()
    return g.reindex(GROUPS)
