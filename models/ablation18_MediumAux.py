import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fsenet import DepthwiseConvBN


class NearestUpsampleRefineReLU6(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.refine = DepthwiseConvBN(channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.upsample(x))


class SerialDepthwiseBlockReLU6(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw1 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw2 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw3 = DepthwiseConvBN(channels, kernel_size=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw1(x)
        out = self.dw2(out)
        out = self.dw3(out)
        out = self.fuse(out)
        return self.act(out + x)


class EncoderStageReLU6(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = SerialDepthwiseBlockReLU6(in_channels)
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


class SerialBottleneckReLU6(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw1 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw2 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw3 = DepthwiseConvBN(channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.dw1(x)
        d2 = self.dw2(d1)
        d3 = self.dw3(d2)
        return x + d1 + d2 + d3


class DecoderStageReLU6(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = NearestUpsampleRefineReLU6(in_channels)

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
        self.shortcut = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.upsample(x), skip], dim=1)
        return self.act(self.refine(fused) + self.shortcut(fused))


class AuxiliaryBoundaryHeadReLU6(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
        out = self.head(x)
        return F.interpolate(out, size=output_size, mode="bilinear", align_corners=False)


class FSENetMediumAux(nn.Module):
    # Ablation 22: FSENet với aux head lấy từ skip stage 3 (s3) thay vì stage 4 (s4).
    ENCODER_CHANNELS = (32, 64, 128, 128, 128)  # stem, e1, e2, e3, e4
    DECODER_CHANNELS = (128, 64, 32, 16)       # d4, d3, d2, d1

    def __init__(self, num_classes: int = 1, aux_hidden_channels: int = 64):
        super().__init__()
        c_stem, c1, c2, c3, c4 = self.ENCODER_CHANNELS

        self.stem = nn.Conv2d(3, c_stem, kernel_size=3, padding=1)

        self.e1 = EncoderStageReLU6(c_stem, c1)
        self.e2 = EncoderStageReLU6(c1, c2)
        self.e3 = EncoderStageReLU6(c2, c3)
        self.e4 = EncoderStageReLU6(c3, c4)

        # aux head lấy đặc trưng biên từ s3 (medium-level), không phải s4 như bản gốc
        self.aux_head = AuxiliaryBoundaryHeadReLU6(c3, aux_hidden_channels, num_classes)

        self.bottleneck = SerialBottleneckReLU6(c4)

        d4, d3, d2, d1 = self.DECODER_CHANNELS
        self.d4 = DecoderStageReLU6(in_channels=c4, skip_channels=c4, out_channels=d4)
        self.d3 = DecoderStageReLU6(in_channels=d4, skip_channels=c3, out_channels=d3)
        self.d2 = DecoderStageReLU6(in_channels=d3, skip_channels=c2, out_channels=d2)
        self.d1 = DecoderStageReLU6(in_channels=d2, skip_channels=c1, out_channels=d1)

        self.head = nn.Conv2d(d1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]

        x = self.stem(x)
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)

        aux_out = self.aux_head(s3, input_size) if self.training else None

        x = self.bottleneck(x)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        main_out = self.head(x)

        return (main_out, aux_out) if self.training else main_out


def build_model(num_classes: int = 1, **kwargs) -> FSENetMediumAux:
    return FSENetMediumAux(num_classes=num_classes, **kwargs)
