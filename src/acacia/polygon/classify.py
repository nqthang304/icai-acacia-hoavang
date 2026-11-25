"""Phân loại ở cấp polygon - quý [paper IV.C, Bảng V].

Mẫu = (polygon, quý) lấy từ data/features_quy (≤ 250 pixel/polygon/quý). Nhãn = lớp đa số.
Đặc trưng mô tả [ASSUMPTION - bài báo nói danh sách đặc trưng chưa đầy đủ, phản biện #24]:
  với mỗi kênh trong 27 kênh ảnh: mean, std, p10, p50, p90; cộng số pixel. DEM không dùng (tùy chọn).
Polygon vắt ranh giới các tập bị loại [paper]. Siêu tham số chọn theo Macro-F1 trên val; báo cáo cả val
(như bài báo) và test (bổ sung, phản biện #38).
QuaST encoder + MLP: đặc trưng 16x16 của QuaST đã huấn luyện, gộp trung bình theo mặt nạ polygon,
rồi MLP [ASSUMPTION: encoder đông cứng, không fine-tune].
"""
from __future__ import annotations

import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

STATS = ("mean", "std", "p10", "p50", "p90")


def load_pixels(data_dir: Path) -> pd.DataFrame:
    # Kaggle tự giải nén .csv.gz khi tạo dataset -> chấp nhận cả .csv lẫn .csv.gz (không lấy trùng quý)
    root = data_dir / "features_quy"
    fs = {}
    for f in sorted(root.rglob("*.csv*")):
        if f.is_file() and f.name.endswith((".csv", ".csv.gz")):
            fs.setdefault(f.name.removesuffix(".gz"), f)
    if not fs:
        raise FileNotFoundError(f"Không thấy file .csv/.csv.gz nào trong {root}")
    return pd.concat([pd.read_csv(f) for _, f in sorted(fs.items())], ignore_index=True)


def polygon_features(px: pd.DataFrame, channels: list[str], include_dem=False) -> pd.DataFrame:
    cols = channels + (["elevation", "slope"] if include_dem else [])
    g = px.groupby(["pid", "quy"])
    parts = {
        "mean": g[cols].mean(), "std": g[cols].std().fillna(0),
        "p10": g[cols].quantile(0.1), "p50": g[cols].median(), "p90": g[cols].quantile(0.9),
    }
    feat = pd.concat({k: v for k, v in parts.items()}, axis=1)
    feat.columns = [f"{c}_{s}" for s, c in feat.columns]
    feat["n_pixels"] = g.size()
    feat["label"] = g["class"].agg(lambda s: s.value_counts().idxmax())
    feat["patch_id"] = g["patch_id"].agg(lambda s: s.value_counts().idxmax())
    return feat.reset_index()


def assign_split(feat: pd.DataFrame, ctx) -> pd.DataFrame:
    sp = ctx.split
    pmap = dict(zip(ctx.grid.patch_id, sp.assign))
    feat = feat[~feat.pid.isin(sp.cross_pids)].copy()
    feat["split"] = feat.patch_id.map(pmap).map({0: "train", 1: "val", 2: "test"})
    return feat


def _metrics(y, p, acacia=2, K=5):
    return {"macro_F1": f1_score(y, p, average="macro", labels=list(range(K)), zero_division=0),
            "balanced_acc": balanced_accuracy_score(y, p),
            "acacia_F1": f1_score(y, p, labels=[acacia], average="macro", zero_division=0)}


def _grid(d: dict):
    keys = list(d)
    for vals in itertools.product(*d.values()):
        yield dict(zip(keys, vals))


def model_zoo(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    return {
        "Random Forest": (lambda **k: RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                                             n_jobs=-1, random_state=seed, **k),
                          {"max_depth": [None, 20], "min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.3]}),
        "SVM-RBF": (lambda **k: make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced",
                                                                     random_state=seed, **k)),
                    {"C": [1, 10, 100], "gamma": ["scale", 0.003, 0.01]}),
        "XGBoost": (lambda **k: XGBClassifier(n_estimators=400, subsample=0.8, colsample_bytree=0.8,
                                              random_state=seed, n_jobs=-1, eval_metric="mlogloss", **k),
                    {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]}),
    }


def balanced_sample_weight(y):
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)


def run_classic(feat: pd.DataFrame, seed=101, K=5):
    Xcols = [c for c in feat.columns if c not in ("pid", "quy", "label", "patch_id", "split")]
    tr, va, te = (feat[feat.split == s] for s in ("train", "val", "test"))
    rows, cms, search = [], {}, []
    for name, (make, grid) in model_zoo(seed).items():
        best = None
        for hp in _grid(grid):
            m = make(**hp)
            fit_kw = {"sample_weight": balanced_sample_weight(tr.label)} if name == "XGBoost" else {}
            m.fit(tr[Xcols].values, tr.label.values, **fit_kw)
            s = _metrics(va.label.values, m.predict(va[Xcols].values), K=K)
            search.append({"model": name, **{k: str(v) for k, v in hp.items()}, **{f"val_{k}": v for k, v in s.items()}})
            if best is None or s["macro_F1"] > best[0]["macro_F1"]:
                best = (s, hp, m)
        s_val, hp, m = best
        pt = m.predict(te[Xcols].values)
        s_test = _metrics(te.label.values, pt, K=K)
        rows.append({"model": name, "hyperparams": str(hp), **{f"val_{k}": v for k, v in s_val.items()},
                     **{f"test_{k}": v for k, v in s_test.items()}})
        cms[name] = {"val": confusion_matrix(va.label, m.predict(va[Xcols].values), labels=list(range(K))),
                     "test": confusion_matrix(te.label, pt, labels=list(range(K)))}
    return pd.DataFrame(rows), pd.DataFrame(search), cms


def support_table(feat: pd.DataFrame, class_names) -> pd.DataFrame:
    t = feat.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int)
    t.index = [class_names[i] for i in t.index]
    return t


def quast_polygon_features(feat: pd.DataFrame, ctx, run_dir, device=None) -> pd.DataFrame:
    """Gộp đặc trưng 16x16 của QuaST theo mặt nạ polygon cho từng (pid, quý)."""
    import torch
    import torch.nn.functional as F

    from ..models import input_channels
    from ..train.trainer import load_trained

    cfg, model, dev = load_trained(Path(run_dir), device)
    ch = input_channels(cfg)
    T = len(ctx.quarters)
    qidx = {q: i for i, q in enumerate(ctx.quarters)}
    want = set(zip(feat.pid, feat.quy))
    acc: dict = {}
    pids_arr = ctx.pids
    for p in range(len(ctx.grid)):
        present = np.unique(pids_arr[p][pids_arr[p] >= 0])
        if not len(present):
            continue
        x = ctx.normalizer(ctx.cache.X[p], np.arange(T))[:, ch]
        with torch.no_grad():
            z = model.features16(torch.from_numpy(x)[None].to(dev), torch.arange(T, device=dev)[None])[0]
        z = z.float().cpu()                                           # [T, C, 16, 16]
        for pid in present:
            for q, t in qidx.items():
                if (pid, q) not in want:
                    continue
                m = torch.from_numpy((pids_arr[p, t] == pid).astype(np.float32))[None, None]
                if m.sum() == 0:
                    continue
                w = F.adaptive_avg_pool2d(m, 16)[0, 0]                # tỉ lệ diện tích trong mỗi ô 16x16
                s, n = (z[t] * w).sum((1, 2)).numpy(), float(w.sum())
                a = acc.setdefault((int(pid), q), [0.0, 0.0])
                a[0] = a[0] + s
                a[1] += n
    rows = [{"pid": k[0], "quy": k[1], **{f"z{i}": v for i, v in enumerate(s / n)}} for k, (s, n) in acc.items()]
    return feat[["pid", "quy", "label", "split"]].merge(pd.DataFrame(rows), on=["pid", "quy"])


def run_quast_mlp(zf: pd.DataFrame, seed=101, K=5):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xc = [c for c in zf.columns if c.startswith("z")]
    tr, va, te = (zf[zf.split == s] for s in ("train", "val", "test"))
    best = None
    for alpha in (1e-4, 1e-3, 1e-2):
        m = make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=alpha, max_iter=500, early_stopping=True,
                                                          random_state=seed))
        m.fit(tr[Xc].values, tr.label.values)
        s = _metrics(va.label.values, m.predict(va[Xc].values), K=K)
        if best is None or s["macro_F1"] > best[0]["macro_F1"]:
            best = (s, alpha, m)
    s_val, alpha, m = best
    pt = m.predict(te[Xc].values)
    return ({"model": "QuaST encoder + MLP", "hyperparams": f"alpha={alpha}",
             **{f"val_{k}": v for k, v in s_val.items()},
             **{f"test_{k}": v for k, v in _metrics(te.label.values, pt, K=K).items()}},
            {"val": confusion_matrix(va.label, m.predict(va[Xc].values), labels=list(range(K))),
             "test": confusion_matrix(te.label, pt, labels=list(range(K)))})
