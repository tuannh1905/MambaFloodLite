"""
Ablation Case 6: w/o Deep Supervision
Chỉ dùng output từ d1 (full resolution), bỏ reduce2/3/4 và fusion
Bỏ: reduce4, reduce3, reduce2, out (conv num_classes*4 → num_classes)
Giữ: reduce1 duy nhất làm output cuối
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .propose_model.model.encoder import TripleBranchEncoderBlock
from .propose_model.module.eesp_bottleneck import EESPBottleneck


# ── DecoderBlock không có CSAM (y hệt model.py) ───────────────
class DecoderBlock_NoCSAM(nn.Module):
    def __init__(self, in_c, skip_c, out_c, reduction=2, kernel_size=3):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        hidden = (in_c + skip_c) // reduction

        self.pw1  = nn.Conv2d(in_c + skip_c, hidden, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.pw2  = nn.Conv2d(hidden, out_c, kernel_size=1, bias=False)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.pw1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pw2(x)
        return x


class MambaUNet_NoDeepSup(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        self.initial_conv = nn.Conv2d(in_channels, 16, kernel_size=1, bias=False)

        self.e1 = TripleBranchEncoderBlock(16,  16,  kernel_size=3)
        self.e2 = TripleBranchEncoderBlock(16,  32,  kernel_size=3)
        self.e3 = TripleBranchEncoderBlock(32,  64,  kernel_size=3)
        self.e4 = TripleBranchEncoderBlock(64,  128, kernel_size=3)

        self.bottleneck = EESPBottleneck(128, 256, num_blocks=1, branches=4)

        self.d4 = DecoderBlock_NoCSAM(256, 128, 128)
        self.d3 = DecoderBlock_NoCSAM(128,  64,  64)
        self.d2 = DecoderBlock_NoCSAM( 64,  32,  32)
        self.d1 = DecoderBlock_NoCSAM( 32,  16,  16)

        # ⚠️ Chỉ giữ reduce1, bỏ reduce2/3/4 và self.out
        self.reduce1 = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.initial_conv(x)

        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)

        x  = self.bottleneck(x)

        d4 = self.d4(x,  skip4)
        d3 = self.d3(d4, skip3)
        d2 = self.d2(d3, skip2)
        d1 = self.d1(d2, skip1)

        # ⚠️ Chỉ dùng d1, không fusion multi-scale
        return self.reduce1(d1)


def build_model(num_classes=1):
    return MambaUNet_NoDeepSup(in_channels=3, num_classes=num_classes)