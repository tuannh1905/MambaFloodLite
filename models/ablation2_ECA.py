import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fsenet import DepthwiseConvBN, NearestUpsampleRefine, AuxiliaryBoundaryHead


class ECAModule(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, beta: int = 1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + beta) / gamma))
        kernel_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SerialDepthwiseBlockECA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw1 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw2 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw3 = DepthwiseConvBN(channels, kernel_size=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.eca = ECAModule(channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw1(x)
        out = self.dw2(out)
        out = self.dw3(out)
        out = self.fuse(out)
        out = self.eca(out)
        return self.act(out + x)


class EncoderStageECA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = SerialDepthwiseBlockECA(in_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.act = nn.ReLU6(inplace=True)

        self.channels_match = in_channels == out_channels
        if not self.channels_match:
            extra_channels = out_channels - in_channels
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, extra_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(extra_channels),
            )
            self.expand_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.block(x)

        if self.channels_match:
            return self.act(self.pool(feat)), feat

        extra_feat = self.expand(feat)
        skip = torch.cat([feat, extra_feat], dim=1)
        out = self.act(torch.cat([self.pool(feat), self.expand_pool(extra_feat)], dim=1))
        return out, skip


class SerialBottleneckECA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw1 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw2 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw3 = DepthwiseConvBN(channels, kernel_size=3)
        self.eca = ECAModule(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.dw1(x)
        d2 = self.dw2(d1)
        d3 = self.dw3(d2)
        return x + self.eca(d1 + d2 + d3)


class DecoderStageECA(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = NearestUpsampleRefine(in_channels)

        concat_channels = in_channels + skip_channels
        hidden_channels = max(out_channels // 4, 4)

        self.refine = nn.Sequential(
            nn.Conv2d(concat_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            DepthwiseConvBN(hidden_channels, kernel_size=5),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.eca = ECAModule(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.upsample(x), skip], dim=1)
        refined = self.eca(self.refine(fused))
        return self.act(refined + self.shortcut(fused))


class FSENetECA(nn.Module):
    # Ablation 2: FSENet + Efficient Channel Attention (ECA) on every stage.
    ENCODER_CHANNELS = (32, 64, 128, 128, 128)
    DECODER_CHANNELS = (128, 64, 32, 16)       # d4, d3, d2, d1

    def __init__(self, num_classes: int = 1, aux_hidden_channels: int = 32):
        super().__init__()
        c_stem, c1, c2, c3, c4 = self.ENCODER_CHANNELS

        self.stem = nn.Conv2d(3, c_stem, kernel_size=3, padding=1)

        self.e1 = EncoderStageECA(c_stem, c1)
        self.e2 = EncoderStageECA(c1, c2)
        self.e3 = EncoderStageECA(c2, c3)
        self.e4 = EncoderStageECA(c3, c4)

        self.aux_head = AuxiliaryBoundaryHead(c4, aux_hidden_channels, num_classes)

        self.bottleneck = SerialBottleneckECA(c4)

        d4, d3, d2, d1 = self.DECODER_CHANNELS
        self.d4 = DecoderStageECA(in_channels=c4, skip_channels=c4, out_channels=d4)
        self.d3 = DecoderStageECA(in_channels=d4, skip_channels=c3, out_channels=d3)
        self.d2 = DecoderStageECA(in_channels=d3, skip_channels=c2, out_channels=d2)
        self.d1 = DecoderStageECA(in_channels=d2, skip_channels=c1, out_channels=d1)

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


def build_model(num_classes: int = 1, **kwargs) -> FSENetECA:
    return FSENetECA(num_classes=num_classes, **kwargs)
