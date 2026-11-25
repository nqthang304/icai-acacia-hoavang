"""Tổng hợp kết quả thành bảng (CSV + Markdown) và hình (PNG) - tái tạo Bảng I-VI, Hình 5, 7-9 của bài báo
cùng các đầu ra bổ sung cho phản biện (ma trận nhầm lẫn, bản đồ dự đoán, phân tách theo nguồn nhãn / pixel lấp,
bảng thiết lập huấn luyện thực tế)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from .. import paths  # noqa: E402
from ..eval.metrics import scores  # noqa: E402

MAIN = {"U-Net": "unet", "QuaST": "quast", "TSViT": "tsvit", "U-TAE": "utae"}
CLASS_COLORS = ["#1b7837", "#d9b365", "#7fbf3f", "#d6604d", "#4393c3"]


def _seed_dirs(runs: Path, exp: str, run: str):
    d = runs / exp / run
    return sorted(d.glob("s*")) if d.exists() else []


def _load_eval(sd: Path, split: str):
    f = sd / f"eval_{split}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _msd(v, nd=3):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "--"
    if len(v) == 1:
        return f"{v[0]:.{nd}f}"
    return f"{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}"


class Report:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.runs = paths.runs_dir(cfg)
        self.res = paths.work_dir(cfg) / "results"
        self.out = self.res / "report"
        self.out.mkdir(parents=True, exist_ok=True)
        self.names = cfg["data"]["class_names"]
        self.md: list[str] = ["# Kết quả tái hiện\n"]

    def _table(self, title: str, df: pd.DataFrame, fname: str, note: str = ""):
        df.to_csv(self.out / f"{fname}.csv", index=False)
        self.md += [f"\n## {title}\n", note + "\n" if note else "", df.to_markdown(index=False), "\n"]

    # ---------------- Bảng I, II + bổ sung ----------------
    def main_tables(self):
        rows, cls_rows, extra = [], [], []
        for name, run in MAIN.items():
            sds = _seed_dirs(self.runs, "main_test", run)
            ev = [e for e in (_load_eval(s, "test") for s in sds) if e]
            if not ev:
                continue
            summ = json.loads((sds[0] / "summary.json").read_text())
            sc = [e["scores"] for e in ev]
            rows.append({"Model": name, "Params (M)": round(summ["params"] / 1e6, 2), "Seeds": len(ev),
                         "mIoU": _msd([s["mIoU"] for s in sc]), "Macro-F1": _msd([s["macro_F1"] for s in sc]),
                         "Pixel acc.": _msd([s["pixel_acc"] for s in sc])})
            iou = np.array([s["IoU"] for s in sc], float)
            cls_rows.append({"Model": name, **{c: f"{np.nanmean(iou[:, i]):.3f}" for i, c in enumerate(self.names)}})
            for key in ("by_source", "by_fill", "by_regime"):
                for sub in ev[0].get(key, {}):
                    vals = [e[key][sub]["scores"]["mIoU"] for e in ev]
                    sup = int(np.sum(ev[0][key][sub]["cm"]))
                    extra.append({"Model": name, "Breakdown": key, "Subset": sub, "mIoU": _msd(vals),
                                  "valid pixels (1 seed)": sup})
            cm = np.sum([np.array(e["cm"]) for e in ev], 0)
            self._cm_figure(cm, f"{name} - test, cộng {len(ev)} seed", f"cm_test_{run}")
            pd.DataFrame(cm, index=self.names, columns=self.names).to_csv(self.out / f"cm_test_{run}.csv")
        if rows:
            self._table("Bảng I - Kết quả test (trung bình ± SD qua seed)", pd.DataFrame(rows), "table1_test",
                        "U-Net đánh giá trên mọi mảnh test, mô hình chuỗi trên mảnh đủ điều kiện.")
            self._table("Bảng II - IoU theo lớp trên test", pd.DataFrame(cls_rows), "table2_class_iou")
            self._table("Bổ sung - mIoU theo nguồn nhãn / pixel lấp / chế độ TOA-SR", pd.DataFrame(extra),
                        "extra_breakdowns", "Phản biện #11, #35. Ma trận nhầm lẫn: cm_test_*.png (#32).")
        cs = self.res / "common_set_scores.csv"
        if cs.exists():
            self._table("Bổ sung - mIoU trên tập mảnh test chung (#31)", pd.read_csv(cs).round(4), "extra_common_set")
        pt = self.res / "paired_tests.csv"
        if pt.exists():
            self._table("Kiểm định cặp (bootstrap 10.000, Wilcoxon, Holm)", pd.read_csv(pt).round(4), "paired_tests")

    def _cm_figure(self, cm, title, fname):
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]:,})", ha="center", va="center", fontsize=6.5,
                        color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(cm)), self.names, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(cm)), self.names, fontsize=8)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Nhãn tham chiếu")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(self.out / f"{fname}.png", dpi=180)
        plt.close(fig)

    # ---------------- Bảng III + các đối chứng val ----------------
    def _val_score(self, exp, run):
        ev = [e for e in (_load_eval(s, "val") for s in _seed_dirs(self.runs, exp, run)) if e]
        return [e["scores"]["mIoU"] for e in ev], [e["scores"]["macro_F1"] for e in ev]

    def _params(self, exp, run):
        sds = _seed_dirs(self.runs, exp, run)
        if not sds or not (sds[0] / "summary.json").exists():
            return None
        return round(json.loads((sds[0] / "summary.json").read_text())["params"] / 1e6, 2)

    def validation_tables(self):
        rows = []
        for T in (1, 2, 4, 8, 16, 32):
            m, f = self._val_score("context_sweep", f"T{T}")
            if m:
                rows.append({"T (quarters)": T, "mIoU": _msd(m), "Macro-F1": _msd(f), "_m": np.mean(m)})
        if rows:
            df = pd.DataFrame(rows)
            self._context_figure(df)
            self._table("Bảng III Panel A - Quét độ dài ngữ cảnh (val, seed 101)", df.drop(columns="_m"), "table3a_context")
        rows = []
        for label, exp, run, T in [("U-Net base-16", "main_test", "unet", 1), ("U-Net base-12", "capacity", "unet_base12", 1),
                                   ("U-Net base-9", "capacity", "unet_base9", 1), ("WindowNet-TAtt", "context_sweep", "T2", 2),
                                   ("QuaST-W2", "short_window", "quast_w2", 2), ("U-TAE-W2", "short_window", "utae_w2", 2)]:
            m, f = self._val_score(exp, run)
            if m:
                rows.append({"Model": label, "T": T, "Params (M)": self._params(exp, run), "Seeds": len(m),
                             "mIoU": _msd(m), "Macro-F1": _msd(f)})
        if rows:
            self._table("Bảng III Panel B - Dung lượng và cửa sổ ngắn (val)", pd.DataFrame(rows), "table3b_capacity")
        for title, exp, runs, fname in [
            ("Đối chứng nguồn cảm biến - PlainTAtt (val, seed 101)", "modality",
             {"S1-only": "s1_only", "S2-only": "s2_only", "S1+S2": "s1s2", "S1+S2+quality": "s1s2_q"}, "modality"),
            ("Ablation thành phần QuaST (val, seed 101)", "quast_ablation",
             {"Mean of two encoders": "mean_encoders", "Gate w/o quality": "gate_no_quality",
              "Gate w/o monotonicity": "gate_no_mono"}, "quast_ablation"),
            ("Độ nhạy buffer không gian - QuaST (val, seed 101)", "buffer",
             {"buffer 640 m": "buffer640", "buffer 1280 m": "buffer1280"}, "buffer"),
        ]:
            rows = []
            if exp in ("quast_ablation", "buffer"):
                ev = _load_eval(self.runs / "main_test" / "quast" / "s101", "val")
                if ev:
                    rows.append({"Config": "QuaST full / no buffer", "mIoU": f"{ev['scores']['mIoU']:.4f}",
                                 "Macro-F1": f"{ev['scores']['macro_F1']:.4f}"})
            for lab, run in runs.items():
                m, f = self._val_score(exp, run)
                if m:
                    row = {"Config": lab, "mIoU": _msd(m, 4), "Macro-F1": _msd(f, 4)}
                    if exp == "buffer":
                        sd = _seed_dirs(self.runs, exp, run)[0]
                        row["train patches"] = json.loads((sd / "summary.json").read_text())["n_train_patches"]
                    rows.append(row)
            if rows:
                self._table(title, pd.DataFrame(rows), fname)

    def _context_figure(self, df):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.plot(df["T (quarters)"], df["_m"], "o-", color="#2166ac")
        ax.set_xscale("log", base=2)
        ax.set_xticks(df["T (quarters)"], df["T (quarters)"])
        ax.set_xlabel("Độ dài ngữ cảnh T (quý)")
        ax.set_ylabel("val mIoU")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(self.out / "fig5_context_sweep.png", dpi=180)
        plt.close(fig)

    # ---------------- Bảng IV ----------------
    def dropout_table(self):
        f = self.res / "optical_dropout.csv"
        if not f.exists():
            return
        df = pd.read_csv(f)
        g = df.groupby(["model", "D"]).agg(seq=("mIoU_sequence", "mean"), dropped=("mIoU_dropped_quarters", "mean"),
                                           ret=("retention_pct", "mean")).reset_index()
        t = g.pivot(index="model", columns="D", values="seq").round(3)
        t.columns = [f"D={c}" for c in t.columns]
        t["Retention (D=8)"] = g[g.D == g.D.max()].set_index("model").ret.round(1).astype(str) + "%"
        self._table("Bảng IV - mIoU toàn chuỗi khi xóa quang học (val)", t.reset_index(), "table4_dropout",
                    f"Số mask seed: {df.mask_seed.nunique()}.")
        t2 = g.pivot(index="model", columns="D", values="dropped").round(3)
        t2.columns = [f"D={c}" for c in t2.columns]
        self._table("Bổ sung - mIoU chỉ trên các quý bị xóa quang học (#27)", t2.reset_index(), "extra_dropout_dropped")
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
        for m, d in g.groupby("model"):
            axs[0].plot(d.D, d.seq, "o-", label=m)
            axs[1].plot(d.D[d.D > 0], d.dropped[d.D > 0], "o-", label=m)
        axs[0].set_title("Toàn chuỗi (như bài báo)", fontsize=9)
        axs[1].set_title("Chỉ quý bị xóa (bổ sung)", fontsize=9)
        for a in axs:
            a.set_xlabel("D (số quý bị xóa quang học)")
            a.grid(alpha=.3)
        axs[0].set_ylabel("val mIoU")
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.out / "fig7b_optical_dropout.png", dpi=180)
        plt.close(fig)

    # ---------------- Bảng V, VI ----------------
    def polygon_tables(self):
        f = self.res / "polygon_results.csv"
        if f.exists():
            self._table("Bảng V - Phân loại polygon-quý (val như bài báo, test bổ sung)", pd.read_csv(f).round(3),
                        "table5_polygon")
            sup = pd.read_csv(self.res / "polygon_support.csv")
            self._table("Bổ sung - Số mẫu polygon-quý theo lớp (#38)", sup, "extra_polygon_support")
        f = self.res / "screening_groups.csv"
        if f.exists():
            self._table("Bảng VI - Hồ sơ nhóm sàng lọc", pd.read_csv(f).round(3), "table6_screening")
            sens = pd.read_csv(self.res / "screening_sensitivity.csv")
            self._table("Độ nhạy trọng số (Hình 9)", sens.round(1), "screening_sensitivity")
            self._screening_figures(pd.read_csv(self.res / "screening_polygons.csv"), sens)

    def _screening_figures(self, sc, sens):
        g = paths.data_dir(self.cfg) / "grid.csv"
        grid = pd.read_csv(g).set_index("patch_id")
        xy = np.array([grid.loc[p.split(",")[0], ["col", "row"]].to_numpy(float) for p in sc.patches])
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        cols = {"Priority": "#d73027", "Conversion consideration": "#fdae61", "Maintain short rotation": "#4575b4"}
        for gname, c in cols.items():
            m = sc.group == gname
            axs[0].scatter(xy[m, 0], xy[m, 1], c=c, s=25, label=f"{gname} ({m.sum()})", edgecolor="k", lw=.3)
        axs[0].scatter(grid.col, grid.row, c="#ddd", s=4, zorder=0)
        axs[0].legend(fontsize=7)
        axs[0].set_title("Nhóm sàng lọc (vị trí theo mảnh)", fontsize=9)
        sca = axs[1].scatter(xy[:, 0], xy[:, 1], c=sc.score, cmap="RdYlBu_r", s=25, edgecolor="k", lw=.3)
        axs[1].scatter(grid.col, grid.row, c="#ddd", s=4, zorder=0)
        fig.colorbar(sca, ax=axs[1], label="Điểm tổng hợp")
        for a in axs:
            a.set_aspect("equal")
            a.set_xlabel("cột")
            a.set_ylabel("hàng")
        fig.tight_layout()
        fig.savefig(self.out / "fig8_screening_map.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.scatter(sens.n_priority, sens.agreement_pct, c="#2166ac")
        for r in sens.itertuples():
            ax.annotate(r.scheme, (r.n_priority, r.agreement_pct), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("Số polygon Priority")
        ax.set_ylabel("Trùng nhóm với baseline (%)")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(self.out / "fig9_sensitivity.png", dpi=180)
        plt.close(fig)

    # ---------------- Bổ sung: thiết lập huấn luyện thực tế ----------------
    def training_table(self):
        rows = []
        for f in sorted(self.runs.glob("*/*/s*/summary.json")):
            s = json.loads(f.read_text())
            exp, run, seed = f.parts[-4], f.parts[-3], f.parts[-2]
            if exp == "smoke":
                continue
            rows.append({"Experiment": exp, "Run": run, "Seed": seed[1:], "Params (M)": round(s["params"] / 1e6, 3),
                         "Epochs run": s["epochs_run"], "Best epoch": s["best_epoch"],
                         "Best val mIoU": round(s["best_val_mIoU"], 4), "LR": s["lr"],
                         "Batch x accum": f"{s['batch_size']} x {s['accum_steps']}",
                         "Train time (min)": round(s["train_time_s"] / 60, 1), "Device": s["device"]})
        if rows:
            self._table("Bổ sung - Thiết lập và ngân sách huấn luyện thực tế từng lần chạy (#26)", pd.DataFrame(rows),
                        "extra_training_log")
            for f in sorted(self.runs.glob("main_test/*/s101/summary.json")):
                s = json.loads(f.read_text())
                df = pd.DataFrame([{"Module": k, "Params": v} for k, v in s["params_by_module"].items()])
                self._table(f"Bổ sung - Số tham số theo module: {f.parts[-3]} (#22)", df, f"extra_params_{f.parts[-3]}")

    # ---------------- Bổ sung: bản đồ dự đoán ----------------
    def prediction_maps(self, ctx, quarters=("2024Q3", "2019Q2", "2021Q4"), n_patches=3):
        """Ảnh RGB | nhãn tham chiếu | dự đoán từng mô hình (seed 101) cho vài mảnh test (#34)."""
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(CLASS_COLORS)
        preds = {}
        for name, run in MAIN.items():
            f = self.runs / "main_test" / run / "s101" / "pred_test.npz"
            if f.exists():
                z = np.load(f)
                preds[name] = dict(zip(z["patches"].tolist(), z["pred"]))
        if not preds:
            return
        common = set.intersection(*[set(d) for d in preds.values()])
        L = ctx.eval_labels
        qi = {q: i for i, q in enumerate(ctx.quarters)}
        for q in quarters:
            t = qi[q]
            ranked = sorted(common, key=lambda p: -(L[p, t] != 255).sum())[:n_patches]
            ncol = 2 + len(preds)
            fig, axs = plt.subplots(len(ranked), ncol, figsize=(2.1 * ncol, 2.2 * len(ranked)), squeeze=False)
            for r, p in enumerate(ranked):
                rgb = np.asarray(ctx.cache.X[p, t, [3, 2, 1]], np.float32).transpose(1, 2, 0)
                lo, hi = np.nanpercentile(rgb.reshape(-1, 3), [2, 98], axis=0)
                axs[r, 0].imshow(np.clip(np.nan_to_num((rgb - lo) / np.maximum(hi - lo, 1e-6)), 0, 1))
                gt = np.ma.masked_equal(L[p, t], 255)
                axs[r, 1].imshow(gt, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
                for c, (name, d) in enumerate(preds.items()):
                    pr = d[p][t].astype(float)
                    axs[r, 2 + c].imshow(np.ma.masked_equal(pr, 255), cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
                    if r == 0:
                        axs[r, 2 + c].set_title(name, fontsize=9)
                axs[r, 0].set_ylabel(ctx.grid.patch_id[p], fontsize=8)
            axs[0, 0].set_title(f"Sentinel-2 RGB {q}", fontsize=9)
            axs[0, 1].set_title("Nhãn tham chiếu", fontsize=9)
            for a in axs.ravel():
                a.set_xticks([])
                a.set_yticks([])
            handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLORS]
            fig.legend(handles, self.names, loc="lower center", ncol=5, fontsize=8)
            fig.tight_layout(rect=(0, 0.05, 1, 1))
            fig.savefig(self.out / f"pred_maps_{q}.png", dpi=170)
            plt.close(fig)
        self.md.append("\n## Bổ sung - Bản đồ dự đoán (#34)\n\n" + "\n".join(
            f"![{q}](pred_maps_{q}.png)" for q in quarters) + "\n")

    def write(self):
        (self.out / "REPORT.md").write_text("\n".join(self.md), encoding="utf8")
        return self.out / "REPORT.md"
