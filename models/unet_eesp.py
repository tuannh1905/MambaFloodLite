import torch
import torch.nn as nn
from .unet import DoubleConv, UpSampling
from .unet_axialdw_dualvss import AxialDWDualVSSEncoder
from .propose_model.module.eesp_bottleneck import EESPBottleneck


class UNetEESP(nn.Module):
    """
    Variant 4: UNet + AxialDW + DualVSS + EESP Bottleneck
    Thay DoubleConv bottleneck -> EESPBottleneck
    """
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.num_classes = num_classes

        self.inc = DoubleConv(in_channels, features[0])

        self.down1 = AxialDWDualVSSEncoder(features[0], features[1])
        self.down2 = AxialDWDualVSSEncoder(features[1], features[2])
        self.down3 = AxialDWDualVSSEncoder(features[2], features[3])

        # Bottleneck: thay DoubleConv -> EESP
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = EESPBottleneck(
            in_channels=features[3],
            out_channels=features[3] * 2,
            num_blocks=1,
            branches=4
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

        x5 = self.bottleneck(self.pool(x4))

        x = self.up1(x5, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        x = self.up4(x, x1)

        return self.outc(x)


def build_model(num_classes=1):
    return UNetEESP(in_channels=3, num_classes=num_classes)