import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(act_type):
    return nn.ReLU6(inplace=True)

class SquareDW(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
    def forward(self, x):
        return self.bn(self.dw(x))

class NearestUpsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return self.refine(self.up(x))

class Straight3x3Block(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw1 = SquareDW(dim)
        self.dw2 = SquareDW(dim)
        self.dw3 = SquareDW(dim)
        self.pw_fuse = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)
    def forward(self, x):
        return self.act(self.bn_fuse(self.pw_fuse(self.dw3(self.dw2(self.dw1(x))))) + x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, act_type='relu6'):
        super().__init__()
        self.block = Straight3x3Block(in_c, act_type)
        self.down_pool = nn.MaxPool2d((2, 2))
        self.same_channels = (in_c == out_c)
        if not self.same_channels:
            self.pw = nn.Sequential(nn.Conv2d(in_c, out_c - in_c, 1, bias=False), nn.BatchNorm2d(out_c - in_c))
            self.down_pw = nn.MaxPool2d((2, 2))
        self.act = get_activation(act_type)
    def forward(self, x):
        feat = self.block(x)
        if self.same_channels:
            return self.act(self.down_pool(feat)), feat
        else:
            feat_pw = self.pw(feat)
            skip = torch.cat([feat, feat_pw], dim=1)
            out = self.act(torch.cat([self.down_pool(feat), self.down_pw(feat_pw)], dim=1))
            return out, skip

class ConcatDecoderBlock_NoAttn(nn.Module):
    def __init__(self, in_c, skip_c, out_c, act_type='relu6'):
        super().__init__()
        self.up = NearestUpsample(in_c)
        gc = max(out_c // 4, 4)
        self.refine = nn.Sequential(
            nn.Conv2d(in_c + skip_c, gc, 1, bias=False),
            nn.BatchNorm2d(gc),
            get_activation(act_type),
            SquareDW(gc, 5),
            nn.Conv2d(gc, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(nn.Conv2d(in_c + skip_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
        self.act = get_activation(act_type)
    def forward(self, x, skip):
        fused = torch.cat([self.up(x), skip], dim=1)
        return self.act(self.refine(fused) + self.shortcut(fused))

class SerialBottleneck_NoAttn(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw1 = SquareDW(dim)
        self.dw2 = SquareDW(dim)
        self.dw3 = SquareDW(dim)
    def forward(self, x):
        d1 = self.dw1(x)
        d2 = self.dw2(d1)
        d3 = self.dw3(d2)
        return x + d1 + d2 + d3

def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Ablation23_WidthHalf(nn.Module):
    def __init__(self, num_classes=1, input_size=128, width_mult=0.75):
        super().__init__()
        def c(x):
            return _make_divisible(int(x * width_mult))
        c0  = c(32)
        c1  = c(64)
        c2  = c(128)
        c3  = c(128)
        c4  = c(128)
        self.conv_in = nn.Conv2d(3, c0, kernel_size=3, padding=1)
        self.e1 = EncoderBlock(c0, c1)
        self.e2 = EncoderBlock(c1, c2)
        self.e3 = EncoderBlock(c2, c3)
        self.e4 = EncoderBlock(c3, c4)
        aux_dim = max(c4 // 2, 8)
        self.aux_head = nn.Sequential(
            nn.Conv2d(c4, aux_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(aux_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(aux_dim, num_classes, 1)
        )
        self.bottleneck = SerialBottleneck_NoAttn(c4)
        self.d4 = ConcatDecoderBlock_NoAttn(c4, c4, c4)
        self.d3 = ConcatDecoderBlock_NoAttn(c4, c3, c3)
        self.d2 = ConcatDecoderBlock_NoAttn(c3, c2, c1)
        self.d1 = ConcatDecoderBlock_NoAttn(c1, c1, max(c0 // 2, 8))
        self.conv_out = nn.Conv2d(max(c0 // 2, 8), num_classes, 1)

    def forward(self, x):
        input_shape = x.shape[2:]
        x = self.conv_in(x)
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        aux_out = None
        if self.training:
            aux_out = self.aux_head(s4)
            aux_out = F.interpolate(aux_out, size=input_shape, mode='bilinear', align_corners=False)
        x = self.bottleneck(x)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        main_out = self.conv_out(x)
        if self.training:
            return main_out, aux_out
        return main_out

def build_model(num_classes=1, input_size=128):
    return Ablation23_WidthHalf(num_classes=num_classes, input_size=input_size)
