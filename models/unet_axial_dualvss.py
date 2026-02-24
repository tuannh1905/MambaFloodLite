import torch
import torch.nn as nn
from .unet import DoubleConv, UpSampling
from .propose_model.module.axial_dw import AxialDW
from .propose_model.module.dual_vss_block import DualVSSBlock


class AxialDWDualVSSEncoder(nn.Module):
    """AxialDW + DualVSS kết hợp"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.axial = AxialDW(dim=in_c, mixer_kernel=(3, 3))
        self.pw = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.vss = DualVSSBlock(hidden_dim=out_c, d_state=8)

    def forward(self, x):
        x = self.pool(x)
        x = self.axial(x)
        x = self.pw(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.vss(x)
        x = x.permute(0, 3, 1, 2)
        return x, x


class UNetAxialDWDualVSS(nn.Module):
    """
    Variant 3: UNet + AxialDW + DualVSS
    Encoder: AxialDW -> PW -> DualVSS
    Giữ nguyên: decoder, bottleneck
    """
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.num_classes = num_classes

        self.inc = DoubleConv(in_channels, features[0])

        self.down1 = AxialDWDualVSSEncoder(features[0], features[1])
        self.down2 = AxialDWDualVSSEncoder(features[1], features[2])
        self.down3 = AxialDWDualVSSEncoder(features[2], features[3])

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(features[3], features[3] * 2)
        )

        self.up1 = UpSampling(features[3] * 2, features[3])
        self.up2 = UpSampling(features[3], features[2])
        self.up3 = UpSampling(features[2], features[1])
        self.up4 = UpSampling(features[1], features[0])

        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)

        x2, skip1 = self.down1(x1)
        x3, skip2 = self.down2(x2)
        x4, skip3 = self.down3(x3)

        x5 = self.bottleneck(x4)

        x = self.up1(x5, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        x = self.up4(x, x1)

        return self.outc(x)


def build_model(num_classes=1):
    return UNetAxialDWDualVSS(in_channels=3, num_classes=num_classes)