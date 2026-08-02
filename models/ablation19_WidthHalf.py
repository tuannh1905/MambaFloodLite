import torch
import torch.nn as nn

from models.fsenet import EncoderStage, SerialBottleneck, DecoderStage, AuxiliaryBoundaryHead


def make_divisible(value: float, divisor: int = 8) -> int:
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class FSENetWidthScale(nn.Module):
    # Ablation 23: FSENet with all channel widths scaled by `width_mult`
    # relative to the base FSENet schedule (32-64-128-128-128).
    BASE_ENCODER_CHANNELS = (32, 64, 128, 128, 128)  # stem, e1, e2, e3, e4

    def __init__(self, num_classes: int = 1, width_mult: float = 0.75):
        super().__init__()
        c_stem, c1, c2, c3, c4 = (make_divisible(c * width_mult) for c in self.BASE_ENCODER_CHANNELS)

        self.stem = nn.Conv2d(3, c_stem, kernel_size=3, padding=1)

        self.e1 = EncoderStage(c_stem, c1)
        self.e2 = EncoderStage(c1, c2)
        self.e3 = EncoderStage(c2, c3)
        self.e4 = EncoderStage(c3, c4)

        aux_hidden_channels = max(c4 // 2, 8)
        self.aux_head = AuxiliaryBoundaryHead(c4, aux_hidden_channels, num_classes)

        self.bottleneck = SerialBottleneck(c4)

        d4_out = c4
        d3_out = c3
        d2_out = c1
        d1_out = max(c_stem // 2, 8)
        self.d4 = DecoderStage(in_channels=c4, skip_channels=c4, out_channels=d4_out)
        self.d3 = DecoderStage(in_channels=d4_out, skip_channels=c3, out_channels=d3_out)
        self.d2 = DecoderStage(in_channels=d3_out, skip_channels=c2, out_channels=d2_out)
        self.d1 = DecoderStage(in_channels=d2_out, skip_channels=c1, out_channels=d1_out)

        self.head = nn.Conv2d(d1_out, num_classes, kernel_size=1)

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


def build_model(num_classes: int = 1, **kwargs) -> FSENetWidthScale:
    return FSENetWidthScale(num_classes=num_classes, **kwargs)
