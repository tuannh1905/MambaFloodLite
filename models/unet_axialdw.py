import torch
import torch.nn as nn
from .unet import DoubleConv, UpSampling
from .propose_model.module.axial_dw import AxialDW


class AxialDWEncoder(nn.Module):
    """Thay DoubleConv encoder bằng AxialDW + PW"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.axial = AxialDW(dim=in_c, mixer_kernel=(3, 3))
        self.pw = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.axial(x)
        x = self.pw(x)
        return x, x  # trả về (downsampled, skip)


class UNetAxialDW(nn.Module):
    """
    Variant 1: UNet + AxialDW
    Chỉ thay encoder DoubleConv -> AxialDW
    Giữ nguyên: decoder, bottleneck
    """
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.num_classes = num_classes

        # Initial conv giữ nguyên như UNet
        self.inc = DoubleConv(in_channels, features[0])

        # Encoder: thay DoubleConv -> AxialDW
        self.down1 = AxialDWEncoder(features[0], features[1])
        self.down2 = AxialDWEncoder(features[1], features[2])
        self.down3 = AxialDWEncoder(features[2], features[3])

        # Bottleneck: giữ nguyên như UNet
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(features[3], features[3] * 2)
        )

        # Decoder: giữ nguyên như UNet
        self.up1 = UpSampling(features[3] * 2, features[3])
        self.up2 = UpSampling(features[3], features[2])
        self.up3 = UpSampling(features[2], features[1])
        self.up4 = UpSampling(features[1], features[0])

        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)                    # (B, 64, H, W)

        x2, skip1 = self.down1(x1)          # (B, 128, H/2, W/2)
        x3, skip2 = self.down2(x2)          # (B, 256, H/4, W/4)
        x4, skip3 = self.down3(x3)          # (B, 512, H/8, W/8)

        x5 = self.bottleneck(x4)            # (B, 1024, H/16, W/16)

        x = self.up1(x5, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        x = self.up4(x, x1)

        return self.outc(x)


def build_model(num_classes=1):
    return UNetAxialDW(in_channels=3, num_classes=num_classes)