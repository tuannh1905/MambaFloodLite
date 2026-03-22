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


# ── Thay đổi 1: ENet-style encoder ──────────────────────────────
# Bỏ pw projection (in_c → out_c), thay bằng maxpool branch + conv branch
# rồi concat → channel tăng miễn phí, không cần 1x1 conv tăng channel
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        conv_out = out_c - in_c          # branch conv chỉ cần phần còn thiếu
        self.dw  = AxialDW(in_c, mixer_kernel=mixer_kernel)
        self.bn  = nn.BatchNorm2d(in_c)
        self.pw  = nn.Conv2d(in_c, conv_out, kernel_size=1)   # nhỏ hơn nhiều
        self.down_conv = nn.MaxPool2d((2, 2))                  # branch pool
        self.down_pw   = nn.MaxPool2d((2, 2))                  # pool skip trước pw
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))          # [B, in_c, H, W]
        pool = self.down_conv(skip)          # [B, in_c, H/2, W/2]  ← maxpool branch
        conv = self.down_pw(self.pw(skip))   # [B, conv_out, H/2, W/2] ← conv branch
        x    = self.act(self.bn2(torch.cat([pool, conv], dim=1)))  # [B, out_c, H/2, W/2]
        return x, skip


# ── Thay đổi 2: Bottleneck 1/4 trong DecoderBlock ───────────────
# Thêm pw_down/pw_up bọc AxialDW, giảm channels khi chạy DW
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        gc = out_c // 4                              # bottleneck ratio 1/4
        self.up      = nn.Upsample(scale_factor=2)
        self.pw      = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn      = nn.BatchNorm2d(out_c)
        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1)       # project xuống
        self.dw      = AxialDW(gc, mixer_kernel=mixer_kernel)    # DW trên gc thay vì out_c
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1)       # project lên lại
        self.bn2     = nn.BatchNorm2d(out_c)
        self.act     = nn.GELU()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.bn(self.pw(x))
        x = self.act(self.pw_up(self.dw(self.pw_down(x))) + x)  # residual
        return x


# ── Thay đổi 3: Giảm channel width tại bottleneck ───────────────
# dim=512 → max_dim=128, bottleneck không còn phình to
class BottleNeckBlock(nn.Module):
    def __init__(self, dim, max_dim=128):
        super().__init__()
        gc = min(dim // 4, max_dim // 4)   # gc tối đa 32 thay vì 128
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

        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding=3)

        self.e1 = EncoderBlock(16,  32)
        self.e2 = EncoderBlock(32,  64)
        self.e3 = EncoderBlock(64,  128)
        self.e4 = EncoderBlock(128, 256)
        self.e5 = EncoderBlock(256, 512)

        self.b5 = BottleNeckBlock(512, max_dim=128)   # thay đổi 3

        self.d5 = DecoderBlock(512, 256)
        self.d4 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64,  32)
        self.d1 = DecoderBlock(32,  16)

        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)

        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        x = self.b5(x)

        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)

        x = self.conv_out(x)
        return x


def build_model(num_classes=1):
    return ULiteModel(num_classes=num_classes)