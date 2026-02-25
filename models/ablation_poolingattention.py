"""
Ablation Case 3: w/o Pooling Attention
Bỏ hoàn toàn Branch 3 (pooling attention)
channel_adjust input: 2*in_c thay vì 3*in_c
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .propose_model.module.axial_dw import AxialDW
from .propose_model.module.dual_vss_block import DualVSSBlock
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


# ── Encoder block bỏ Branch 3 ─────────────────────────────────
class TripleBranchEncoderBlock_NoPoolingAttn(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=7):
        super().__init__()

        # Branch 1: AxialDW + PW + BN + ReLU (giữ nguyên)
        self.branch1_axial = AxialDW(dim=in_c, mixer_kernel=(kernel_size, kernel_size))
        self.branch1_pw    = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.branch1_bn    = nn.BatchNorm2d(in_c)
        self.branch1_relu  = nn.ReLU(inplace=True)

        # Branch 2: AxialDW + PW + BN + ReLU + DualVSS (giữ nguyên)
        self.branch2_axial = AxialDW(dim=in_c, mixer_kernel=(kernel_size, kernel_size))
        self.branch2_pw    = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.branch2_bn    = nn.BatchNorm2d(in_c)
        self.branch2_relu  = nn.ReLU(inplace=True)
        self.branch2_vss   = DualVSSBlock(hidden_dim=in_c, d_state=8)

        # Branch 3: BỎ HOÀN TOÀN

        # ⚠️ Concat chỉ: b1 + b2 = 2*in_c → out_c
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(in_c * 2, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Branch 1
        b1 = self.branch1_axial(x)
        b1 = self.branch1_pw(b1)
        b1 = self.branch1_bn(b1)
        b1 = self.branch1_relu(b1)

        # Branch 2
        b2 = self.branch2_axial(x)
        b2 = self.branch2_pw(b2)
        b2 = self.branch2_bn(b2)
        b2 = self.branch2_relu(b2)
        b2 = b2.permute(0, 2, 3, 1)
        b2 = self.branch2_vss(b2)
        b2 = b2.permute(0, 3, 1, 2)

        # Branch 3 bị bỏ → chỉ concat b1 + b2
        skip = self.channel_adjust(torch.cat([b1, b2], dim=1))
        x    = self.pool(skip)
        return x, skip


# ── Full Model ─────────────────────────────────────────────────
class MambaUNet_NoPoolingAttn(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        self.initial_conv = nn.Conv2d(in_channels, 16, kernel_size=1, bias=False)

        self.e1 = TripleBranchEncoderBlock_NoPoolingAttn(16,  16,  kernel_size=3)
        self.e2 = TripleBranchEncoderBlock_NoPoolingAttn(16,  32,  kernel_size=3)
        self.e3 = TripleBranchEncoderBlock_NoPoolingAttn(32,  64,  kernel_size=3)
        self.e4 = TripleBranchEncoderBlock_NoPoolingAttn(64,  128, kernel_size=3)

        self.bottleneck = EESPBottleneck(128, 256, num_blocks=1, branches=4)

        self.d4 = DecoderBlock_NoCSAM(256, 128, 128)
        self.d3 = DecoderBlock_NoCSAM(128,  64,  64)
        self.d2 = DecoderBlock_NoCSAM( 64,  32,  32)
        self.d1 = DecoderBlock_NoCSAM( 32,  16,  16)

        self.reduce4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.reduce3 = nn.Conv2d( 64, num_classes, kernel_size=1)
        self.reduce2 = nn.Conv2d( 32, num_classes, kernel_size=1)
        self.reduce1 = nn.Conv2d( 16, num_classes, kernel_size=1)
        self.out     = nn.Conv2d(num_classes * 4, num_classes, kernel_size=1)

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

        out4 = F.interpolate(self.reduce4(d4), size=(H, W), mode='bilinear', align_corners=False)
        out3 = F.interpolate(self.reduce3(d3), size=(H, W), mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.reduce2(d2), size=(H, W), mode='bilinear', align_corners=False)
        out1 = self.reduce1(d1)

        return self.out(torch.cat([out1, out2, out3, out4], dim=1))


def build_model(num_classes=1):
    return MambaUNet_NoPoolingAttn(in_channels=3, num_classes=num_classes)