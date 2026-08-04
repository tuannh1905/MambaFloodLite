import torch
import torch.nn as nn

from models.fsenet import EncoderStage, SerialBottleneck, DecoderStage, AuxiliaryBoundaryHead


class FSENetAuxAll(nn.Module):
    # Aux ablation: one auxiliary boundary head attached to every encoder
    # skip feature (s1, s2, s3, s4) simultaneously -- full deep supervision
    # instead of a single head at one stage. All aux outputs share the
    # same aux_weight in the trainer's loss loop (outputs[1:]).
    ENCODER_CHANNELS = (32, 64, 128, 128, 128)  # stem, e1, e2, e3, e4
    DECODER_CHANNELS = (128, 64, 32, 16)        # d4, d3, d2, d1

    def __init__(self, num_classes: int = 1, aux_hidden_channels: int = 64):
        super().__init__()
        c_stem, c1, c2, c3, c4 = self.ENCODER_CHANNELS

        self.stem = nn.Conv2d(3, c_stem, kernel_size=3, padding=1)

        self.e1 = EncoderStage(c_stem, c1)
        self.e2 = EncoderStage(c1, c2)
        self.e3 = EncoderStage(c2, c3)
        self.e4 = EncoderStage(c3, c4)

        self.aux_head_s1 = AuxiliaryBoundaryHead(c1, aux_hidden_channels, num_classes)
        self.aux_head_s2 = AuxiliaryBoundaryHead(c2, aux_hidden_channels, num_classes)
        self.aux_head_s3 = AuxiliaryBoundaryHead(c3, aux_hidden_channels, num_classes)
        self.aux_head_s4 = AuxiliaryBoundaryHead(c4, aux_hidden_channels, num_classes)

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

        if self.training:
            aux_out_s1 = self.aux_head_s1(s1, input_size)
            aux_out_s2 = self.aux_head_s2(s2, input_size)
            aux_out_s3 = self.aux_head_s3(s3, input_size)
            aux_out_s4 = self.aux_head_s4(s4, input_size)

        x = self.bottleneck(x)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        main_out = self.head(x)

        if self.training:
            return main_out, aux_out_s1, aux_out_s2, aux_out_s3, aux_out_s4
        return main_out


def build_model(num_classes: int = 1, **kwargs) -> FSENetAuxAll:
    return FSENetAuxAll(num_classes=num_classes, **kwargs)
