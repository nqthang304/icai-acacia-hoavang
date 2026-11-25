"""Các mô hình chuỗi ảnh: PlainTAtt/WindowNet, U-TAE (xuất theo quý), TSViT (xuất theo quý).

Giao diện chung: forward(x [B,T,C,H,W], tpos [B,T], pad [B,T] bool) -> logits [B,T,K,H,W].
Tất cả đều xuất bản đồ cho từng quý vì bài báo đánh giá mIoU trên toàn chuỗi (mục IV.C).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Decoder, Encoder, TemporalAttention, fold, sinusoid, unfold


class PlainTAtt(nn.Module):
    """Backbone cố định cho quét độ dài ngữ cảnh và đối chứng cảm biến [thesis Bảng 3.2].

    Tài liệu không mô tả kiến trúc (phản biện #6). Thiết kế ở đây [ASSUMPTION]:
    encoder CNN dùng chung cho mọi quý -> 2 khối temporal self-attention hai chiều tại tỉ lệ 1/8
    -> decoder U-Net với skip của chính quý đó. Không hợp nhất cảm biến: các kênh được ghép thẳng.
    Với T < 32 (WindowNet-TAtt) cùng kiến trúc, chỉ khác độ dài cửa sổ đầu vào.
    """
    temporal = True

    def __init__(self, in_ch: int, n_classes: int, widths=(16, 32, 64, 128), att_depth: int = 2, heads: int = 4):
        super().__init__()
        widths = list(widths)
        self.encoder = Encoder(in_ch, widths)
        self.tatt = TemporalAttention(widths[-1], att_depth, heads)
        self.decoder = Decoder(widths, n_classes)

    def forward(self, x, tpos, pad=None):
        B = x.shape[0]
        feats = [unfold(f, B) for f in self.encoder(fold(x))]
        feats[-1] = self.tatt(feats[-1], tpos, pad)
        return unfold(self.decoder([fold(f) for f in feats]), B)


class UTAE(nn.Module):
    """U-TAE [Garnot & Landrieu 2021] sửa để xuất bản đồ cho từng quý.

    Bản gốc: L-TAE với master query gộp T quý thành 1 bản đồ. Ở đây mỗi quý t là một query,
    tạo ma trận chú ý A[h, t, s] tại tỉ lệ thấp nhất; A được nội suy song tuyến lên các tỉ lệ
    skip và dùng để tổng hợp skip theo nhóm kênh (giữ cơ chế "attention-guided skip" của U-TAE).
    """
    temporal = True

    def __init__(self, in_ch: int, n_classes: int, widths=(16, 32, 48, 96), heads: int = 4, d_k: int = 8,
                 mlp_hidden: int = 128):
        super().__init__()
        widths = list(widths)
        assert all(w % heads == 0 for w in widths)
        self.heads, self.d_k = heads, d_k
        self.encoder = Encoder(in_ch, widths)
        c = widths[-1]
        self.in_norm = nn.GroupNorm(heads, c)
        self.qk = nn.Linear(c, 2 * heads * d_k)
        self.out_mlp = nn.Sequential(nn.Linear(c, mlp_hidden), nn.ReLU(inplace=True), nn.Linear(mlp_hidden, c))
        self.out_norm = nn.GroupNorm(heads, c)
        self.decoder = Decoder(widths, n_classes)

    def _attention(self, f, tpos, pad):
        """f: [B,T,C,h,w] -> A [B,H,T,T,h,w]"""
        B, T, C, h, w = f.shape
        z = self.in_norm(fold(f)).view(B, T, C, h, w).permute(0, 3, 4, 1, 2) + sinusoid(tpos, C)[:, None, None]
        q, k = self.qk(z).view(B, h, w, T, 2, self.heads, self.d_k).unbind(4)      # [B,h,w,T,H,dk]
        logits = torch.einsum("bxytnd,bxysnd->bxynts", q, k) / math.sqrt(self.d_k)  # [B,h,w,H,T,T]
        if pad is not None:
            logits = logits.masked_fill(pad[:, None, None, None, None, :], float("-inf"))
        A = logits.softmax(-1)
        return A.permute(0, 3, 4, 5, 1, 2)  # [B,H,T,S,h,w]

    def _aggregate(self, f, A):
        """f: [B,S,C,H,W]; A: [B,heads,T,S,h,w] -> [B,T,C,H,W]"""
        B, S, C, Hh, Ww = f.shape
        if A.shape[-1] != Ww:
            a = A.flatten(0, 3)[:, None]                         # nội suy từng bản đồ chú ý
            a = F.interpolate(a, size=(Hh, Ww), mode="bilinear", align_corners=False)
            A = a.view(*A.shape[:4], Hh, Ww)
        fg = f.view(B, S, self.heads, C // self.heads, Hh, Ww)
        return torch.einsum("bntsxy,bsncxy->btncxy", A, fg).reshape(B, -1, C, Hh, Ww)

    def forward(self, x, tpos, pad=None):
        B = x.shape[0]
        feats = [unfold(f, B) for f in self.encoder(fold(x))]
        A = self._attention(feats[-1], tpos, pad)
        agg = [self._aggregate(f, A) for f in feats]
        top = agg[-1]
        t = self.out_mlp(top.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        agg[-1] = unfold(self.out_norm(fold(top + t)), B)
        return unfold(self.decoder([fold(f) for f in agg]), B)


class TSViT(nn.Module):
    """TSViT [Tarasiou 2023] dạng phân rã thời gian -> không gian, xuất bản đồ cho từng quý.

    Bản gốc dùng K token lớp ở encoder thời gian để ra 1 bản đồ; để đánh giá theo quý, token lớp
    được bỏ và mỗi token (quý, vị trí) được giải mã thành patch p x p x K [ASSUMPTION].
    """
    temporal = True

    def __init__(self, in_ch: int, n_classes: int, img: int = 64, patch: int = 4, dim: int = 128,
                 t_depth: int = 3, s_depth: int = 3, heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.p, self.n, self.K = patch, img // patch, n_classes
        self.embed = nn.Linear(in_ch * patch * patch, dim)
        mk = lambda d: nn.TransformerEncoder(  # noqa: E731
            nn.TransformerEncoderLayer(dim, heads, int(dim * mlp_ratio), 0.0, batch_first=True, norm_first=True),
            d, enable_nested_tensor=False)
        self.temporal_tf, self.spatial_tf = mk(t_depth), mk(s_depth)
        self.space_pos = nn.Parameter(torch.zeros(1, self.n * self.n, dim))
        nn.init.trunc_normal_(self.space_pos, std=0.02)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch * patch * n_classes)
        self.dim = dim

    def forward(self, x, tpos, pad=None):
        B, T, C, H, W = x.shape
        p, n, D = self.p, self.n, self.dim
        z = x.view(B, T, C, n, p, n, p).permute(0, 1, 3, 5, 2, 4, 6).reshape(B, T, n * n, C * p * p)
        z = self.embed(z) + sinusoid(tpos, D)[:, :, None]                       # [B,T,N,D]
        z = z.permute(0, 2, 1, 3).reshape(B * n * n, T, D)
        kpm = None if pad is None else pad[:, None].expand(B, n * n, T).reshape(B * n * n, T)
        z = self.temporal_tf(z, src_key_padding_mask=kpm)
        z = z.view(B, n * n, T, D).permute(0, 2, 1, 3).reshape(B * T, n * n, D) + self.space_pos
        z = self.norm(self.spatial_tf(z))
        y = self.head(z).view(B, T, n, n, self.K, p, p).permute(0, 1, 4, 2, 5, 3, 6)
        return y.reshape(B, T, self.K, H, W)
