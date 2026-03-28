import torch
import torch.nn as nn

# ==============================================================================
# 1. MODULES LÕI CHO TRƯỜNG HỢP 1.4 (PFCU + DG)
# ==============================================================================

class AxialDW_Original(nn.Module):
    """Lõi AxialDW chuẩn nguyên bản (Dùng cho Bottleneck và Decoder)"""
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
    TRƯỜNG HỢP 1.4: Khối Encoder Hoàn chỉnh (PFCU + DG).
    Kết hợp cả trường thụ cảm rộng (Dilation 1,2,5) và ranh giới sắc nét (DG Shortcut).
    KHÔNG sử dụng Attention để cô lập biến số kiến trúc.
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
        
        # Kích hoạt chuẩn MCU
        self.act = nn.PReLU(dim)

    def forward(self, x):
        # Gom bối cảnh từ 3 quy mô
        b1 = self.branch_r1(x)
        b2 = self.branch_r2(x)
        b5 = self.branch_r5(x)
        fused_context = self.bn_fuse(self.pw_fuse(b1 + b2 + b5))
        
        # Lọc ranh giới
        guided_details = self.dg_shortcut(x)
        
        # Dung hợp Bối cảnh + Chi tiết
        return self.act(fused_context + guided_details)


# ==============================================================================
# 2. ENCODER (PFCU+DG) | BOTTLENECK & DECODER (NGUYÊN BẢN)
# ==============================================================================

class EncoderBlock_1_4(nn.Module):
    """Khối Encoder Case 1.4: Lõi PFCU+DG, Kích hoạt PReLU"""
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        # Áp dụng lõi hoàn chỉnh của Kịch bản 1
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
    """Bottleneck nguyên bản 100%"""
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


class DecoderBlock_Original(nn.Module):
    """Decoder nguyên bản 100%"""
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
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x


# ==============================================================================
# 3. MẠNG CHÍNH (ABLATION 1.4 - PFCU + DG)
# ==============================================================================

class ULiteModel_Ablation_1_4(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (7, 7)

        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')
        
        self.e1 = EncoderBlock_1_4(16, 32, mixer_kernel=mk)
        self.e2 = EncoderBlock_1_4(32, 64, mixer_kernel=mk)
        self.e3 = EncoderBlock_1_4(64, 128, mixer_kernel=mk)
        self.e4 = EncoderBlock_1_4(128, 256, mixer_kernel=mk)
        self.e5 = EncoderBlock_1_4(256, 512, mixer_kernel=mk)

        self.b5 = BottleneckBlock_Original(512)

        self.d5 = DecoderBlock_Original(512, 256, mixer_kernel=mk)
        self.d4 = DecoderBlock_Original(256, 128, mixer_kernel=mk)
        self.d3 = DecoderBlock_Original(128, 64, mixer_kernel=mk)
        self.d2 = DecoderBlock_Original(64, 32, mixer_kernel=mk)
        self.d1 = DecoderBlock_Original(32, 16, mixer_kernel=mk)
        
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
    return ULiteModel_Ablation_1_4(num_classes=num_classes)