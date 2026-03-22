import torch
import torch.nn as nn


class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        pad_h = (h + (h - 1) * (dilation - 1)) // 2
        pad_w = (w + (w - 1) * (dilation - 1)) // 2
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding=(pad_h, 0), groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding=(0, pad_w), groups=dim, dilation=dilation)

    def forward(self, x):
        return x + self.dw_h(x) + self.dw_w(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        conv_out = out_c - in_c
        self.dw       = AxialDW(in_c, mixer_kernel=mixer_kernel)
        self.bn       = nn.BatchNorm2d(in_c)
        self.pw       = nn.Conv2d(in_c, conv_out, kernel_size=1)
        self.down_pool = nn.MaxPool2d((2, 2))
        self.down_pw   = nn.MaxPool2d((2, 2))
        self.bn2      = nn.BatchNorm2d(out_c)
        self.act      = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        pool = self.down_pool(skip)
        conv = self.down_pw(self.pw(skip))
        x    = self.act(self.bn2(torch.cat([pool, conv], dim=1)))
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        gc = max(out_c // 4, 4)
        self.up      = nn.Upsample(scale_factor=2)
        self.pw      = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn      = nn.BatchNorm2d(out_c)
        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1)
        self.dw      = AxialDW(gc, mixer_kernel=mixer_kernel)
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1)
        self.act     = nn.GELU()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.bn(self.pw(x))
        x = self.act(self.pw_up(self.dw(self.pw_down(x))) + x)
        return x


class BottleNeckBlock(nn.Module):
    def __init__(self, dim, max_dim=64):
        super().__init__()
        gc = min(dim // 4, max_dim // 4)
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)
        self.bn  = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], dim=1)
        x = self.act(self.pw2(self.bn(x)))
        return x


class ULiteModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # 4 stage thay vì 5, channel max 128
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32,  mixer_kernel=(7, 7))
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=(7, 7))
        self.e3 = EncoderBlock(64,  128, mixer_kernel=(7, 7))
        self.e4 = EncoderBlock(128, 128, mixer_kernel=(7, 7))

        self.b4 = BottleNeckBlock(128, max_dim=64)

        self.d4 = DecoderBlock(128, 128, mixer_kernel=(7, 7))
        self.d3 = DecoderBlock(128, 64,  mixer_kernel=(7, 7))
        self.d2 = DecoderBlock(64,  32,  mixer_kernel=(7, 7))
        self.d1 = DecoderBlock(32,  16,  mixer_kernel=(7, 7))

        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)

        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)

        x = self.b4(x)

        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)

        x = self.conv_out(x)
        return x


def build_model(num_classes=1):
    return ULiteModel(num_classes=num_classes)