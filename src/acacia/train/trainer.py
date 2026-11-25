"""Huấn luyện một cấu hình (mô hình x seed) [paper IV.B, thesis Bảng 3.1].

AdamW (wd 1e-4), cosine + 1 epoch warm-up, clip 1,0, dừng sớm sau `patience` epoch không tăng val mIoU,
checkpoint tốt nhất theo val mIoU. Ghi log từng epoch và bảng tóm tắt (phản biện #26: số epoch thực tế,
epoch tốt nhất, thời gian, phần cứng, số tham số theo module).
"""
from __future__ import annotations

import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.context import DataContext
from ..data.datasets import class_weights, make_dataset
from ..eval.evaluator import evaluate
from ..eval.predict import predict
from ..models import build_model, count_params, input_channels, params_by_child
from .losses import SegLoss


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def patch_sets(ctx: DataContext, cfg: dict) -> dict[str, np.ndarray]:
    """U-Net dùng mọi mảnh của tập; mô hình chuỗi chỉ dùng mảnh đủ điều kiện [paper III.C]."""
    sp, sc = ctx.split, cfg["split"]
    elig = cfg["model"]["name"] != "unet"
    out = {
        "train": sp.idx("train", eligible_only=elig, buffer_m=sc["buffer_m"], patch_m=sc["patch_m"]),
        "val": sp.idx("val", eligible_only=elig),
        "test": sp.idx("test", eligible_only=elig),
    }
    n = cfg["train"].get("debug_patches")      # chỉ để chạy thử nhanh
    if n:
        has = (ctx.eval_labels != 255).any(axis=(1, 2, 3))
        out = {k: v[has[v]][:n] for k, v in out.items()}
    return out


def _worker_init(i):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def train_run(cfg: dict, seed: int, run_dir: Path, device=None, log=print) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf8")
    seed_everything(seed)
    device = get_device(device)
    ctx = DataContext(cfg)
    tc = cfg["train"]
    ch = input_channels(cfg)
    ps = patch_sets(ctx, cfg)

    train_ds = make_dataset(ctx, cfg, ps["train"], ctx.labels, ch, augment=tc["augment"],
                            skip_unlabeled=tc["skip_unlabeled"])
    train_ds.rng = np.random.default_rng(seed)
    val_ds = make_dataset(ctx, cfg, ps["val"], ctx.eval_labels, ch)
    cw = class_weights(ctx.labels[ps["train"]], cfg["data"]["num_classes"], cfg["loss"]["class_weight_cap"])
    sw = train_ds.sample_weights(cfg["sampling"]["oversample"])
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double), len(train_ds), replacement=True, generator=g)
    nw = tc["num_workers"]
    dl = DataLoader(train_ds, batch_size=tc["batch_size"], sampler=sampler, num_workers=nw, drop_last=False,
                    pin_memory=device.type == "cuda", worker_init_fn=_worker_init if nw else None,
                    persistent_workers=nw > 0)

    model = build_model(cfg).to(device)
    is_quast = cfg["model"]["name"] == "quast"
    loss_fn = SegLoss(cfg, cw).to(device)
    oc = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"])
    accum = tc["accum_steps"]
    steps_per_epoch = math.ceil(len(dl) / accum)
    warm = oc["warmup_epochs"] * steps_per_epoch
    total = tc["max_epochs"] * steps_per_epoch

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    info = {
        "seed": seed, "params": count_params(model), "params_by_module": params_by_child(model),
        "class_weights": cw.tolist(), "n_train_items": len(train_ds), "n_val_items": len(val_ds),
        "n_train_patches": int(len(ps["train"])), "n_val_patches": int(len(ps["val"])),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
        "torch": torch.__version__, "batch_size": tc["batch_size"], "accum_steps": accum, "lr": oc["lr"],
        "max_epochs": tc["max_epochs"], "patience": tc["patience"],
    }
    log(f"[{run_dir.parent.name}/{run_dir.name}] params={info['params']/1e6:.3f}M train_items={len(train_ds)} device={info['device']}")

    best, best_ep, bad, t0 = -1.0, -1, 0, time.time()
    hist = []
    for ep in range(1, tc["max_epochs"] + 1):
        model.train()
        te = time.time()
        agg, nb = {}, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(dl):
            x, y = b["x"].to(device, non_blocking=True), b["y"].to(device, non_blocking=True)
            pad = b.get("pad")
            pad = pad.to(device) if pad is not None else None
            with torch.autocast(device.type, enabled=use_amp):
                if not model.temporal:
                    out, aux = model(x), None
                elif is_quast:
                    out, aux = model(x, b["t"].to(device), pad, return_aux=True)
                else:
                    out, aux = model(x, b["t"].to(device), pad), None
            loss, parts = loss_fn(out, y, aux, pad)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        pred = predict(model, val_ds, device, cfg["data"]["num_classes"], cfg["eval"]["batch_size"], amp=tc["amp"])
        sc = evaluate(pred, ctx.eval_labels[ps["val"]], cfg["data"]["num_classes"])["scores"]
        rec = {"epoch": ep, **{f"train_{k}": v / max(nb, 1) for k, v in agg.items()},
               "val_mIoU": sc["mIoU"], "val_macro_F1": sc["macro_F1"], "lr": opt.param_groups[0]["lr"],
               "epoch_s": time.time() - te}
        hist.append(rec)
        with open(run_dir / "log.jsonl", "a", encoding="utf8") as f:
            f.write(json.dumps(rec) + "\n")
        if sc["mIoU"] > best:
            best, best_ep, bad = sc["mIoU"], ep, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad += 1
        log(f"[{run_dir.parent.name}/{run_dir.name}] ep {ep:3d} loss {rec.get('train_ce', 0):.3f}/{rec.get('train_dice', 0):.3f} "
            f"val mIoU {sc['mIoU']:.4f} (best {best:.4f} @ {best_ep})")
        if bad >= tc["patience"]:
            break

    info.update({"best_val_mIoU": best, "best_epoch": best_ep, "epochs_run": len(hist),
                 "stopped_early": len(hist) < tc["max_epochs"], "train_time_s": time.time() - t0})
    (run_dir / "summary.json").write_text(json.dumps(info, indent=1))
    return info


def load_trained(run_dir: Path, device=None):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf8"))
    device = get_device(device)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()
    return cfg, model, device
