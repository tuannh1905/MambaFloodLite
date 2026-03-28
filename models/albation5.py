import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. MODULES LÕI (Không có Attention)
# ==============================================================================

class AxialDW_Original(nn.Module):
    """Lõi AxialDW chuẩn nguyên bản (Có Identity Shortcut)"""
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation, bias=False)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation, bias=False)

    def forward(self, x):
        return x + self.dw_h(x) + self.dw_w(x)


class DetailGuidance(nn.Module):
    """Nhánh Lọc chi tiết (DG Shortcut)"""
    def __init__(self, dim):
        super().__init__()
        self.dg_dw_h = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding='same', groups=dim, bias=False)
        self.dg_dw_w = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding='same', groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        edges = self.dg_dw_h(x) + self.dg_dw_w(x)
        return self.bn(x + edges)


class Axial_PFCU_DG_NoAttn(nn.Module):
    """
    TRƯỜNG HỢP 2.1: Khối Encoder Hoàn chỉnh (PFCU + DG) nhưng KHÔNG Attention.
    Đây là Baseline để so sánh với các kỹ thuật Dung hợp và Bối cảnh phía sau.
    """
    def __init__(self, dim, mixer_kernel=(7, 7)):
        super().__init__()
        
        # Nhánh Bối cảnh (PFCU)
        self.branch_r1 = AxialDW_Original(dim, mixer_kernel, dilation=1)
        self.branch_r2 = AxialDW_Original(dim, mixer_kernel, dilation=2)
        self.branch_r5 = AxialDW_Original(dim, mixer_kernel, dilation=5)
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        
        # Nhánh Chi tiết (DG Shortcut)
        self.dg_shortcut = DetailGuidance(dim)
        
        # Kích hoạt PReLU
        self.act = nn.PReLU(dim)

    def forward(self, x):
        b1 = self.branch_r1(x)
        b2 = self.branch_r2(x)
        b5 = self.branch_r5(x)
        fused_context = self.bn_fuse(self.pw_fuse(b1 + b2 + b5))
        
        guided_details = self.dg_shortcut(x)
        
        return self.act(fused_context + guided_details)


# ==============================================================================
# 2. ENCODER (PFCU+DG) | BOTTLENECK & DECODER (THÔ SƠ - CAT)
# ==============================================================================

class EncoderBlock_2_1(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.dw = Axial_PFCU_DG_NoAttn(in_c, mixer_kernel=mixer_kernel)
        self.bn = nn.BatchNorm2d(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw = nn.Conv2d(in_c, conv_out, kernel_size=1, bias=False)
            self.down_pw = nn.MaxPool2d((2, 2))

        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        skip = self.bn(self.dw(x))
        pool = self.down_pool(skip)

        if self.same_channels:
            x = self.act(self.bn2(pool))
        else:
            conv = self.down_pw(self.pw(skip))
            x = self.act(self.bn2(torch.cat([pool, conv], dim=1)))

        return x, skip


class BottleneckBlock_Original(nn.Module):
    """Bottleneck nguyên bản (Không SPP)"""
    def __init__(self, dim):
        super().__init__()
        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1, bias=False)
        self.dw1 = AxialDW_Original(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW_Original(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW_Original(gc, mixer_kernel=(3, 3), dilation=3)
        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1, bias=False)
        self.act = nn.PReLU(dim)

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], dim=1)
        x = self.act(self.pw2(self.bn(x)))
        return x


class DecoderBlock_Simple(nn.Module):
    """Decoder dung hợp thô sơ bằng ghép nối (Cat), không UAFM"""
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW_Original(out_c, mixer_kernel=mixer_kernel)
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1, bias=False)
        self.act = nn.PReLU(out_c)

    def forward(self, x, skip):
        x = self.up(x)
        
        # Đảm bảo an toàn kích thước
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        # Ghép nối thô sơ (Cat)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x


# ==============================================================================
# 3. MẠNG CHÍNH (ABLATION 2.1 - BASELINE KỊCH BẢN 2)
# ==============================================================================

class ULiteModel_Ablation_2_1(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (7, 7)

        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')
        
        # Khối Encoder hiện tại (PFCU + DG)
        self.e1 = EncoderBlock_2_1(16, 32, mixer_kernel=mk)
        self.e2 = EncoderBlock_2_1(32, 64, mixer_kernel=mk)
        self.e3 = EncoderBlock_2_1(64, 128, mixer_kernel=mk)
        self.e4 = EncoderBlock_2_1(128, 256, mixer_kernel=mk)
        self.e5 = EncoderBlock_2_1(256, 512, mixer_kernel=mk)

        # Bottleneck và Decoder thô sơ
        self.b5 = BottleneckBlock_Original(512)

        self.d5 = DecoderBlock_Simple(512, 256, mixer_kernel=mk)
        self.d4 = DecoderBlock_Simple(256, 128, mixer_kernel=mk)
        self.d3 = DecoderBlock_Simple(128, 64, mixer_kernel=mk)
        self.d2 = DecoderBlock_Simple(64, 32, mixer_kernel=mk)
        self.d1 = DecoderBlock_Simple(32, 16, mixer_kernel=mk)
        
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
        
        return self.conv_out(x)

def build_model(num_classes=1):
    return ULiteModel_Ablation_2_1(num_classes=num_classes)