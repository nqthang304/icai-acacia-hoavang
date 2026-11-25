"""U-Net một quý (T = 1) [Ronneberger 2015].

base=16 -> 1,95M, base=12 -> 1,10M, base=9 -> 0,62M tham số: khớp Bảng III của bài báo.
"""
import torch.nn as nn

from .blocks import Decoder, Encoder


class UNet(nn.Module):
    temporal = False

    def __init__(self, in_ch: int, n_classes: int, base: int = 16, depth: int = 5):
        super().__init__()
        widths = [base * 2 ** i for i in range(depth)]
        self.encoder = Encoder(in_ch, widths)
        self.decoder = Decoder(widths, n_classes)

    def forward(self, x, tpos=None, pad=None):
        """x: [B, C, H, W] -> [B, K, H, W]"""
        return self.decoder(self.encoder(x))
