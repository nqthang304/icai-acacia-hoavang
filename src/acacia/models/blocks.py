"""Khối dùng chung cho các mô hình phân đoạn."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    """(conv3x3 -> BN -> ReLU) x 2, conv không bias như U-Net chuẩn."""

    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class Encoder(nn.Module):
    """Encoder kiểu U-Net: trả về danh sách đặc trưng ở các tỉ lệ 1, 1/2, 1/4, ..."""

    def __init__(self, cin: int, widths: list[int]):
        super().__init__()
        self.stem = DoubleConv(cin, widths[0])
        self.down = nn.ModuleList(nn.Sequential(nn.MaxPool2d(2), DoubleConv(a, b)) for a, b in zip(widths, widths[1:]))

    def forward(self, x):
        feats = [self.stem(x)]
        for d in self.down:
            feats.append(d(feats[-1]))
        return feats


class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.conv = DoubleConv(cout + cskip, cout)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], 1))


class Decoder(nn.Module):
    """Decoder U-Net: widths giống encoder (nông -> sâu); skip_widths mặc định = widths."""

    def __init__(self, widths: list[int], n_classes: int, skip_widths: list[int] | None = None):
        super().__init__()
        sw = skip_widths or widths
        self.ups = nn.ModuleList(Up(widths[i + 1], sw[i], widths[i]) for i in reversed(range(len(widths) - 1)))
        self.head = nn.Conv2d(widths[0], n_classes, 1)

    def forward(self, feats):
        x = feats[-1]
        for up, skip in zip(self.ups, reversed(feats[:-1])):
            x = up(x, skip)
        return self.head(x)


def sinusoid(t: torch.Tensor, d: int, period: float = 1000.0) -> torch.Tensor:
    """Mã hóa vị trí thời gian dạng sin/cos. t: [...] (chỉ số quý) -> [..., d]."""
    half = d // 2
    freq = torch.exp(-math.log(period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    ang = t.float()[..., None] * freq
    return torch.cat([ang.sin(), ang.cos()], -1)


class TemporalAttention(nn.Module):
    """Các khối Transformer hai chiều theo trục thời gian, áp dụng độc lập cho từng pixel.

    x: [B, T, C, h, w], pad: [B, T] bool (True = quý đệm, bị bỏ qua khi chú ý).
    """

    def __init__(self, dim: int, depth: int = 2, heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, int(dim * mlp_ratio), dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.dim = dim

    def forward(self, x, tpos, pad=None):
        B, T, C, h, w = x.shape
        z = x.permute(0, 3, 4, 1, 2).reshape(B * h * w, T, C)
        pe = sinusoid(tpos, C)                                  # [B, T, C]
        z = z + pe[:, None, None].expand(B, h, w, T, C).reshape(B * h * w, T, C)
        kpm = None if pad is None else pad[:, None, None].expand(B, h, w, T).reshape(B * h * w, T)
        z = self.blocks(z, src_key_padding_mask=kpm)
        return z.reshape(B, h, w, T, C).permute(0, 3, 4, 1, 2)


def fold(x):
    """[B, T, C, H, W] -> [B*T, C, H, W]"""
    return x.flatten(0, 1)


def unfold(x, B):
    return x.view(B, -1, *x.shape[1:])


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def params_by_child(m: nn.Module) -> dict[str, int]:
    return {n: count_params(c) for n, c in m.named_children()}
