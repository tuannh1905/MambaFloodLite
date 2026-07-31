import torch
import torch.nn as nn

from fsenet import EncoderStage, SerialBottleneck, DecoderStage


class FSENetNoAux(nn.Module):
    # Ablation 4: FSENet without the auxiliary boundary head.
    ENCODER_CHANNELS = (16, 32, 64, 128, 128)  # stem, e1, e2, e3, e4
    DECODER_CHANNELS = (128, 64, 32, 16)       # d4, d3, d2, d1

    def __init__(self, num_classes: int = 1):
        super().__init__()
        c_stem, c1, c2, c3, c4 = self.ENCODER_CHANNELS

        self.stem = nn.Conv2d(3, c_stem, kernel_size=3, padding=1)

        self.e1 = EncoderStage(c_stem, c1)
        self.e2 = EncoderStage(c1, c2)
        self.e3 = EncoderStage(c2, c3)
        self.e4 = EncoderStage(c3, c4)

        self.bottleneck = SerialBottleneck(c4)

        d4, d3, d2, d1 = self.DECODER_CHANNELS
        self.d4 = DecoderStage(in_channels=c4, skip_channels=c4, out_channels=d4)
        self.d3 = DecoderStage(in_channels=d4, skip_channels=c3, out_channels=d3)
        self.d2 = DecoderStage(in_channels=d3, skip_channels=c2, out_channels=d2)
        self.d1 = DecoderStage(in_channels=d2, skip_channels=c1, out_channels=d1)

        self.head = nn.Conv2d(d1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)

        x = self.bottleneck(x)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        return self.head(x)


def build_model(num_classes: int = 1, **kwargs) -> FSENetNoAux:
    return FSENetNoAux(num_classes=num_classes, **kwargs)
