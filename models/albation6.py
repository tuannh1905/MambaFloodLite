import torch
import torch.nn as nn

# ==============================================================================
# 1. CORE MODULES (Cải tiến cho Encoder - Không có Attention)
# ==============================================================================

class DetailGuidance(nn.Module):
    """
    Nhánh Lọc chi tiết (Detail Guidance) bằng Axial 3x3 để thay thế Shortcut Identity.
    Chỉ dùng trong khối trích xuất (Axial_PFCU_DG_NoAttn) của Encoder.
    """
    def __init__(self, dim):
        super().__init__()
        self.dg_dw_h = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding='same', groups=dim, bias=False)
        self.dg_dw_w = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding='same', groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        edges = self.dg_dw_h(x) + self.dg_dw_w(x)
        return self.bn(x + edges)

class AxialDW_OriginalTemplate(nn.Module):
    """
    Lõi AxialDW chuẩn template gốc (Sử dụng padding='same', Identity shortcut).
    Cần thiết cho khối Bottleneck gốc và Decoder gốc.
    """
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        # Identity shortcut thô sơ (+) theo template gốc
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

class Axial_PFCU_DG_NoAttn(nn.Module):
    """
    Khối song song bối cảnh (PFCU - 1, 2, 5) kết hợp Detail Guidance.
    Được thiết kế để thay thế lớp AxialDW đơn trong Encoder.
    - SỬ DỤNG: PReLU (Tối ưu MCU)
    - KHÔNG CÓ: Coordinate Attention
    """
    def __init__(self, dim, mixer_kernel=(7, 7)):
        super().__init__()
        self.branch_r1 = AxialDW_OriginalTemplate(dim, mixer_kernel, dilation=1)
        self.branch_r2 = AxialDW_OriginalTemplate(dim, mixer_kernel, dilation=2)
        self.branch_r5 = AxialDW_OriginalTemplate(dim, mixer_kernel, dilation=5)
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        
        self.dg_shortcut = DetailGuidance(dim)
        
        # Hàm kích hoạt PReLU thay cho GELU
        self.act = nn.PReLU(dim)

    def forward(self, x):
        b1 = self.branch_r1(x)
        b2 = self.branch_r2(x)
        b5 = self.branch_r5(x)
        
        fused_context = self.bn_fuse(self.pw_fuse(b1 + b2 + b5))
        guided_details = self.dg_shortcut(x)
        
        return self.act(fused_context + guided_details)

# ==============================================================================
# 2. ENCODER (Cải tiến)
# ==============================================================================

class EncoderBlock_ImprovedEncoder(nn.Module):
    """Khối Encoder chuẩn template cũ, nhưng lõi DW được cải tiến"""
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        # THAY THẾ AxialDW đơn lẻ bằng khối PFCU + DG (Không Attention)
        self.dw = Axial_PFCU_DG_NoAttn(in_c, mixer_kernel=mixer_kernel)
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2, 2))
        
        # PReLU thay cho GELU
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip

# ==============================================================================
# 3. DECODER & BOTTLENECK (Nguyên bản chuẩn template cũ)
# ==============================================================================

class BottleneckBlock_Original(nn.Module):
    """Bottleneck nguyên bản 100% chuẩn template cũ"""
    def __init__(self, dim):
        super().__init__()
        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW_OriginalTemplate(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW_OriginalTemplate(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW_OriginalTemplate(gc, mixer_kernel=(3, 3), dilation=3)
        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        
        # PReLU thay cho GELU
        self.act = nn.PReLU(dim)

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], dim=1)
        x = self.act(self.pw2(self.bn(x)))
        return x

class DecoderBlock_Original(nn.Module):
    """Decoder nguyên bản 100% chuẩn template cũ (Dùng torch.cat)"""
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW_OriginalTemplate(out_c, mixer_kernel=mixer_kernel)
        
        # PReLU thay cho GELU
        self.act = nn.PReLU(out_c)
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        # Nối thô sơ chuẩn gốc
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x

# ==============================================================================
# 4. MẠNG CHÍNH (ULITE + IMPROVED ENCODER ONLY)
# ==============================================================================

class ULiteModel_ImprovedEncoder_Only(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # InConv chuẩn gốc
        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')
        
        # 5 Tầng Encoder (CẢI TIẾN)
        mk = (7, 7) 
        self.e1 = EncoderBlock_ImprovedEncoder(16, 32, mixer_kernel=mk)
        self.e2 = EncoderBlock_ImprovedEncoder(32, 64, mixer_kernel=mk)
        self.e3 = EncoderBlock_ImprovedEncoder(64, 128, mixer_kernel=mk)
        self.e4 = EncoderBlock_ImprovedEncoder(128, 256, mixer_kernel=mk)
        self.e5 = EncoderBlock_ImprovedEncoder(256, 512, mixer_kernel=mk)

        # Bottleneck (NGUYÊN BẢN)
        self.b5 = BottleneckBlock_Original(512)

        # 5 Tầng Decoder (NGUYÊN BẢN)
        self.d5 = DecoderBlock_Original(512, 256, mixer_kernel=mk)
        self.d4 = DecoderBlock_Original(256, 128, mixer_kernel=mk)
        self.d3 = DecoderBlock_Original(128, 64, mixer_kernel=mk)
        self.d2 = DecoderBlock_Original(64, 32, mixer_kernel=mk)
        self.d1 = DecoderBlock_Original(32, 16, mixer_kernel=mk)
        
        # OutConv chuẩn gốc
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
    return ULiteModel_ImprovedEncoder_Only(num_classes=num_classes)