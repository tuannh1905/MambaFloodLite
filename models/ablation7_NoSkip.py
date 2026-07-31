import torch
import torch.nn as nn

from models.fsenet import DepthwiseConvBN, NearestUpsampleRefine, EncoderStage, SerialBottleneck, AuxiliaryBoundaryHead


class DecoderStageNoSkip(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = NearestUpsampleRefine(in_channels)
        hidden_channels = max(out_channels // 4, 4)

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            DepthwiseConvBN(hidden_channels, kernel_size=5),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_feat = self.upsample(x)
        return self.act(self.refine(up_feat) + self.shortcut(up_feat))


class FSENetNoSkip(nn.Module):
    # Ablation 7: decoder receives no encoder skip connections.
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

        self.bottleneck = SerialBottleneck(c4)

        d4, d3, d2, d1 = self.DECODER_CHANNELS
        self.d4 = DecoderStageNoSkip(in_channels=c4, out_channels=d4)
        self.d3 = DecoderStageNoSkip(in_channels=d4, out_channels=d3)
        self.d2 = DecoderStageNoSkip(in_channels=d3, out_channels=d2)
        self.d1 = DecoderStageNoSkip(in_channels=d2, out_channels=d1)

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
        x = self.d4(x)
        x = self.d3(x)
        x = self.d2(x)
        x = self.d1(x)
        main_out = self.head(x)

        return (main_out, aux_out) if self.training else main_out


def build_model(num_classes: int = 1, **kwargs) -> FSENetNoSkip:
    return FSENetNoSkip(num_classes=num_classes, **kwargs)
