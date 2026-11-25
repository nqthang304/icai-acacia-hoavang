"""Tạo mô hình từ config: cfg['model'] = {name: ..., các tham số kiến trúc}."""
from __future__ import annotations

from .blocks import count_params, params_by_child
from .quast import QuaST
from .temporal import TSViT, UTAE, PlainTAtt
from .unet import UNet


def input_channels(cfg: dict) -> list[int]:
    """Danh sách chỉ số kênh đưa vào mô hình (đối chứng cảm biến chọn tập con)."""
    m = cfg["model"]
    if "channels" in m:
        out = []
        for a, b in m["channels"]:
            out += list(range(a, b))
        return out
    return list(range(*cfg["data"]["input_channels"]))


def build_model(cfg: dict):
    m = dict(cfg["model"])
    name = m.pop("name")
    m.pop("channels", None)
    m.pop("window", None)
    K = cfg["data"]["num_classes"]
    cin = len(input_channels(cfg))
    if name == "unet":
        return UNet(cin, K, **m)
    if name == "plain_tatt":
        return PlainTAtt(cin, K, **m)
    if name == "utae":
        return UTAE(cin, K, **m)
    if name == "tsvit":
        return TSViT(cin, K, **m)
    if name == "quast":
        d = cfg["data"]
        return QuaST(K, tuple(d["optical"]), tuple(d["sar"]), tuple(d["quality"]), **m)
    raise ValueError(name)


__all__ = ["build_model", "count_params", "params_by_child", "input_channels"]
