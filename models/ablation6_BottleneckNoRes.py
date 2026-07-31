import torch
import torch.nn as nn

from models.fsenet import DepthwiseConvBN, EncoderStage, DecoderStage, AuxiliaryBoundaryHead


class SerialBottleneckNoResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw1 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw2 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw3 = DepthwiseConvBN(channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.dw1(x)
        d2 = self.dw2(d1)
        d3 = self.dw3(d2)
        return d1 + d2 + d3


class FSENetNoResidual(nn.Module):
    # Ablation 6: bottleneck without the residual shortcut (x + d1 + d2 + d3 -> d1 + d2 + d3).
    ENCODER_CHANNELS = (16, 32, 64, 128, 128)  # stem, e1, e2, e3, e4
    DECODER_CHANNELS = (128, 64, 32, 16)       # d4, d3, d2, d1

    def __init__(self, num_classes: int = 1, aux_hidden_channels: int = 32):
        super().__init__()
        c_stem, c1, c2, c3, c4 = self.ENCODER_CHANNELS

        self.stem = nn.Conv2d(3, c_stem, kernel_size=3, padding=1)

        self.e1 = EncoderStage(c_stem, c1)
        self.e2 = EncoderStage(c1, c2)
        self.e3 = EncoderStage(c2, c3)
        self.e4 = EncoderStage(c3, c4)

        self.aux_head = AuxiliaryBoundaryHead(c4, aux_hidden_channels, num_classes)

        self.bottleneck = SerialBottleneckNoResidual(c4)

        d4, d3, d2, d1 = self.DECODER_CHANNELS
        self.d4 = DecoderStage(in_channels=c4, skip_channels=c4, out_channels=d4)
        self.d3 = DecoderStage(in_channels=d4, skip_channels=c3, out_channels=d3)
        self.d2 = DecoderStage(in_channels=d3, skip_channels=c2, out_channels=d2)
        self.d1 = DecoderStage(in_channels=d2, skip_channels=c1, out_channels=d1)

        self.head = nn.Conv2d(d1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]

        x = self.stem(x)
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)

        aux_out = self.aux_head(s4, input_size) if self.training else None

        x = self.bottleneck(x)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        main_out = self.head(x)

        return (main_out, aux_out) if self.training else main_out


def build_model(num_classes: int = 1, **kwargs) -> FSENetNoResidual:
    return FSENetNoResidual(num_classes=num_classes, **kwargs)
