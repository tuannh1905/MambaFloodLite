import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fsenet import DepthwiseConvBN, NearestUpsampleRefine, AuxiliaryBoundaryHead


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden_channels = max(4, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(out)


class CBAMModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class SerialDepthwiseBlockCBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw1 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw2 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw3 = DepthwiseConvBN(channels, kernel_size=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.cbam = CBAMModule(channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw1(x)
        out = self.dw2(out)
        out = self.dw3(out)
        out = self.fuse(out)
        out = self.cbam(out)
        return self.act(out + x)


class EncoderStageCBAM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = SerialDepthwiseBlockCBAM(in_channels)
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


class SerialBottleneckCBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw1 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw2 = DepthwiseConvBN(channels, kernel_size=3)
        self.dw3 = DepthwiseConvBN(channels, kernel_size=3)
        self.cbam = CBAMModule(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.dw1(x)
        d2 = self.dw2(d1)
        d3 = self.dw3(d2)
        return x + self.cbam(d1 + d2 + d3)


class DecoderStageCBAM(nn.Module):
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
        self.cbam = CBAMModule(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.upsample(x), skip], dim=1)
        refined = self.cbam(self.refine(fused))
        return self.act(refined + self.shortcut(fused))


class FSENetCBAM(nn.Module):
    # Ablation 3: FSENet + CBAM (channel + spatial attention) on every stage.
    ENCODER_CHANNELS = (32, 64, 128, 128, 128)  # stem, e1, e2, e3, e4
    DECODER_CHANNELS = (128, 64, 32, 16)       # d4, d3, d2, d1

    def __init__(self, num_classes: int = 1, aux_hidden_channels: int = 64):
        super().__init__()
        c_stem, c1, c2, c3, c4 = self.ENCODER_CHANNELS

        self.stem = nn.Conv2d(3, c_stem, kernel_size=3, padding=1)

        self.e1 = EncoderStageCBAM(c_stem, c1)
        self.e2 = EncoderStageCBAM(c1, c2)
        self.e3 = EncoderStageCBAM(c2, c3)
        self.e4 = EncoderStageCBAM(c3, c4)

        self.aux_head = AuxiliaryBoundaryHead(c4, aux_hidden_channels, num_classes)

        self.bottleneck = SerialBottleneckCBAM(c4)

        d4, d3, d2, d1 = self.DECODER_CHANNELS
        self.d4 = DecoderStageCBAM(in_channels=c4, skip_channels=c4, out_channels=d4)
        self.d3 = DecoderStageCBAM(in_channels=d4, skip_channels=c3, out_channels=d3)
        self.d2 = DecoderStageCBAM(in_channels=d3, skip_channels=c2, out_channels=d2)
        self.d1 = DecoderStageCBAM(in_channels=d2, skip_channels=c1, out_channels=d1)

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


def build_model(num_classes: int = 1, **kwargs) -> FSENetCBAM:
    return FSENetCBAM(num_classes=num_classes, **kwargs)
