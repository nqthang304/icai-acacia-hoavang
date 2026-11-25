"""QuaST: hợp nhất quang học - SAR có điều kiện theo chất lượng quan sát (bài báo mục IV.A, Hình 3).

  g_{t,s} = sigmoid(h_s([O_{t,s}; S_{t,s}; Q_{t,s}]))
  F_{t,s} = g ⊙ O + (1 - g) ⊙ S + B_s(Q_{t,s})                                   (1)

- O, S: đặc trưng từ hai encoder riêng (21 kênh quang học, 3 kênh SAR).
- q_t ∈ R^3: trung bình không gian của 3 kênh chất lượng, nhúng bằng MLP hai lớp -> Q_{t,s}.
- h_s: cổng conv 1x1 hai lớp; B_s: phép chiếu 1x1 học được.
- Hợp nhất ở tỉ lệ 16x16 và 8x8; sau đó 2 khối temporal attention hai chiều; decoder có skip.
- Skip quang học (64x64, 32x32) đi vòng qua nút hợp nhất (Hình 3).
- Monotonicity penalty (trọng số 0,05, tính trong loss) khuyến khích g nhỏ hơn khi chất lượng thấp hơn.

Biến thể đối chứng (bài báo mục IV.B): fusion = full | mean (trung bình hai encoder)
| gate_noq (cổng không điều kiện chất lượng). "Không penalty" = full với loss.mono_weight = 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import Decoder, Encoder, TemporalAttention, fold, unfold


class QualityFusion(nn.Module):
    def __init__(self, c: int, q_dim: int = 3, q_hidden: int = 32, mode: str = "full"):
        super().__init__()
        self.mode = mode
        if mode == "mean":
            return
        use_q = mode == "full"
        self.q_mlp = nn.Sequential(nn.Linear(q_dim, q_hidden), nn.ReLU(inplace=True), nn.Linear(q_hidden, c)) if use_q else None
        cin = 3 * c if use_q else 2 * c
        self.h = nn.Sequential(nn.Conv2d(cin, c, 1), nn.ReLU(inplace=True), nn.Conv2d(c, c, 1))
        self.b = nn.Linear(c, c) if use_q else None   # B_s: chiếu 1x1 của Q (hằng theo không gian)

    def forward(self, O, S, q):
        """O, S: [N, C, h, w]; q: [N, 3] -> F [N,C,h,w], g [N] (trung bình cổng, None nếu mode=mean)"""
        if self.mode == "mean":
            return 0.5 * (O + S), None
        parts = [O, S]
        if self.q_mlp is not None:
            Q = self.q_mlp(q)
            parts.append(Q[:, :, None, None].expand_as(O))
        g = torch.sigmoid(self.h(torch.cat(parts, 1)))
        Fz = g * O + (1 - g) * S
        if self.b is not None:
            Fz = Fz + self.b(Q)[:, :, None, None]
        return Fz, g.mean(dim=(1, 2, 3))


class QuaST(nn.Module):
    temporal = True

    def __init__(self, n_classes: int, optical=(0, 21), sar=(21, 24), quality=(24, 27),
                 widths=(16, 32, 64, 128), sar_widths=(8, 16, 64, 128), fusion: str = "full",
                 att_depth: int = 2, heads: int = 4, q_hidden: int = 32):
        super().__init__()
        widths, sar_widths = list(widths), list(sar_widths)
        assert widths[2:] == sar_widths[2:], "hai encoder phải cùng số kênh ở tỉ lệ hợp nhất (16x16, 8x8)"
        self.sl_o, self.sl_s, self.sl_q = slice(*optical), slice(*sar), slice(*quality)
        self.enc_optical = Encoder(optical[1] - optical[0], widths)
        self.enc_sar = Encoder(sar[1] - sar[0], sar_widths)
        self.fuse16 = QualityFusion(widths[2], quality[1] - quality[0], q_hidden, fusion)
        self.fuse8 = QualityFusion(widths[3], quality[1] - quality[0], q_hidden, fusion)
        self.tatt = TemporalAttention(widths[3], att_depth, heads)
        self.decoder = Decoder(widths, n_classes)

    def forward(self, x, tpos, pad=None, return_aux: bool = False):
        B = x.shape[0]
        xf = fold(x)
        fo = self.enc_optical(xf[:, self.sl_o])
        fs = self.enc_sar(xf[:, self.sl_s])
        q = xf[:, self.sl_q].mean(dim=(2, 3))                 # q_t: trung bình không gian
        f16, g16 = self.fuse16(fo[2], fs[2], q)
        f8, g8 = self.fuse8(fo[3], fs[3], q)
        f8 = fold(self.tatt(unfold(f8, B), tpos, pad))
        y = unfold(self.decoder([fo[0], fo[1], f16, f8]), B)
        if not return_aux:
            return y
        gates = None if g16 is None else torch.stack([g16, g8], -1).view(B, -1, 2)   # [B,T,2]
        return y, {"gates": gates, "q": q.view(B, -1, q.shape[-1])}

    def features16(self, x, tpos, pad=None):
        """Đặc trưng đã hợp nhất 16x16 sau khi qua decoder tầng đầu (dùng cho nhánh polygon)."""
        B = x.shape[0]
        xf = fold(x)
        fo, fs = self.enc_optical(xf[:, self.sl_o]), self.enc_sar(xf[:, self.sl_s])
        q = xf[:, self.sl_q].mean(dim=(2, 3))
        f16, _ = self.fuse16(fo[2], fs[2], q)
        f8, _ = self.fuse8(fo[3], fs[3], q)
        f8 = fold(self.tatt(unfold(f8, B), tpos, pad))
        z = self.decoder.ups[0](f8, f16)                     # [B*T, C16, 16, 16]
        return unfold(z, B)


def monotonicity_penalty(gates: torch.Tensor, q: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Phạt cặp (i, j) có chất lượng q_i < q_j nhưng độ tin cậy quang học g_i > g_j.

    gates: [B,T,S] trung bình cổng theo tỉ lệ; q: [B,T,3]. Chất lượng = S2 count chuẩn hóa (q[...,0])
    [ASSUMPTION: bài báo chỉ nói "quality-comparison pairs"]. Các cặp lấy trong cùng batch.
    """
    g = gates.flatten(0, 1)                      # [N, S]
    s = q[..., 0].flatten()                      # [N]
    if valid is not None:
        keep = valid.flatten()
        g, s = g[keep], s[keep]
    if g.shape[0] < 2:
        return gates.new_zeros(())
    lower = (s[:, None] < s[None, :]).float()    # i kém hơn j
    viol = torch.relu(g[:, None, :] - g[None, :, :]).mean(-1)
    return (viol * lower).sum() / lower.sum().clamp_min(1.0)
