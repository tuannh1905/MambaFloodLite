import torch
import torch.nn as nn

from models.fsenet import SerialDepthwiseBlock, SerialBottleneck, DecoderStage, AuxiliaryBoundaryHead


class EncoderStageAvgPool(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = SerialDepthwiseBlock(in_channels)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.act = nn.ReLU6(inplace=True)

        self.channels_match = in_channels == out_channels
        if not self.channels_match:
            extra_channels = out_channels - in_channels
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, extra_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(extra_channels),
            )
            self.expand_pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.block(x)

        if self.channels_match:
            return self.act(self.pool(feat)), feat

        extra_feat = self.expand(feat)
        skip = torch.cat([feat, extra_feat], dim=1)
        out = self.act(torch.cat([self.pool(feat), self.expand_pool(extra_feat)], dim=1))
        return out, skip


class FSENetAvgPool(nn.Module):
    # Ablation 15: average-pooling downsampling instead of max-pooling.
    ENCODER_CHANNELS = (32, 64, 128, 128, 128)  # stem, e1, e2, e3, e4
    DECODER_CHANNELS = (128, 64, 32, 16)        # d4, d3, d2, d1

    def __init__(self, num_classes: int = 1, aux_hidden_channels: int = 64):
        super().__init__()
        c_stem, c1, c2, c3, c4 = self.ENCODER_CHANNELS

        self.stem = nn.Conv2d(3, c_stem, kernel_size=3, padding=1)

        self.e1 = EncoderStageAvgPool(c_stem, c1)
        self.e2 = EncoderStageAvgPool(c1, c2)
        self.e3 = EncoderStageAvgPool(c2, c3)
        self.e4 = EncoderStageAvgPool(c3, c4)

        self.aux_head = AuxiliaryBoundaryHead(c4, aux_hidden_channels, num_classes)

        self.bottleneck = SerialBottleneck(c4)

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


def build_model(num_classes: int = 1, **kwargs) -> FSENetAvgPool:
    return FSENetAvgPool(num_classes=num_classes, **kwargs)
