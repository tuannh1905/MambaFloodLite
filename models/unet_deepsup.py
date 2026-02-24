import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import DoubleConv
from .unet_axialdw_dualvss import AxialDWDualVSSEncoder
from .propose_model.module.eesp_bottleneck import EESPBottleneck
from .unet_csam import UpSamplingCSAM


class UNetDeepSup(nn.Module):
    """
    Variant 6: UNet + AxialDW + DualVSS + EESP + CSAM + Deep Supervision
    Thêm deep supervision từ các decoder level
    = Full Proposed Model
    """
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.num_classes = num_classes

        self.inc = DoubleConv(in_channels, features[0])

        self.down1 = AxialDWDualVSSEncoder(features[0], features[1])
        self.down2 = AxialDWDualVSSEncoder(features[1], features[2])
        self.down3 = AxialDWDualVSSEncoder(features[2], features[3])

        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = EESPBottleneck(
            in_channels=features[3],
            out_channels=features[3] * 2,
            num_blocks=1,
            branches=4
        )

        self.up1 = UpSamplingCSAM(features[3] * 2, features[3])
        self.up2 = UpSamplingCSAM(features[3], features[2])
        self.up3 = UpSamplingCSAM(features[2], features[1])
        self.up4 = UpSamplingCSAM(features[1], features[0])

        # Deep supervision heads
        self.reduce4 = nn.Conv2d(features[3], num_classes, kernel_size=1)
        self.reduce3 = nn.Conv2d(features[2], num_classes, kernel_size=1)
        self.reduce2 = nn.Conv2d(features[1], num_classes, kernel_size=1)
        self.reduce1 = nn.Conv2d(features[0], num_classes, kernel_size=1)

        # Final fusion
        self.out = nn.Conv2d(num_classes * 4, num_classes, kernel_size=1)

    def forward(self, x):
        _, _, H, W = x.shape
        x1 = self.inc(x)

        x2, skip1 = self.down1(x1)
        x3, skip2 = self.down2(x2)
        x4, skip3 = self.down3(x3)

        x5 = self.bottleneck(self.pool(x4))

        d4 = self.up1(x5, skip3)
        d3 = self.up2(d4, skip2)
        d2 = self.up3(d3, skip1)
        d1 = self.up4(d2, x1)

        # Deep supervision
        out4 = F.interpolate(self.reduce4(d4), size=(H, W), mode='bilinear', align_corners=False)
        out3 = F.interpolate(self.reduce3(d3), size=(H, W), mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.reduce2(d2), size=(H, W), mode='bilinear', align_corners=False)
        out1 = self.reduce1(d1)

        out = self.out(torch.cat([out1, out2, out3, out4], dim=1))
        return out


def build_model(num_classes=1):
    return UNetDeepSup(in_channels=3, num_classes=num_classes)