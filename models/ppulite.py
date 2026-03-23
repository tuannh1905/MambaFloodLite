import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. CORE MODULES TỪ U-LITE (Backbone & Bottleneck)
# ==============================================================================
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        return x + self.dw_h(x) + self.dw_w(x)

class EncoderBlock(nn.Module):
    """Encoding then downsampling (Giữ nguyên từ U-Lite)"""
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel=(7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2, 2))
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip

class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution (Giữ nguyên từ U-Lite)"""
    def __init__(self, dim):
        super().__init__()
        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)
        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        return x

# ==============================================================================
# 2. MODULES LAI TẠO TỪ PP-LITESEG (Skip Connection & Decoder)
# ==============================================================================
class SpatialAttentionModule(nn.Module):
    """Trích xuất Attention Không gian để giữ ranh giới nước sắc nét"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size=1, bias=False)

    def forward(self, x_up, x_low):
        mean_up = torch.mean(x_up, dim=1, keepdim=True)
        max_up, _ = torch.max(x_up, dim=1, keepdim=True)
        mean_low = torch.mean(x_low, dim=1, keepdim=True)
        max_low, _ = torch.max(x_low, dim=1, keepdim=True)
        x = self.conv(torch.cat([mean_up, max_up, mean_low, max_low], dim=1))
        return torch.sigmoid(x)

class UAFM_Spatial(nn.Module):
    """Unified Attention Fusion Module - Thay thế cho phép nối torch.cat cồng kềnh"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.attention = SpatialAttentionModule()

    def forward(self, x_high, x_low):
        size = x_low.size()[2:]
        x_low = self.conv(x_low)
        x_high_up = F.interpolate(x_high, size, mode='bilinear', align_corners=True)
        alpha = self.attention(x_high_up, x_low)
        # Dung hợp có trọng số thay vì nối chập
        return alpha * x_high_up + (1 - alpha) * x_low

class FLD_DecoderBlock(nn.Module):
    """Decoder ép giảm tham số theo chuẩn FLD của PP-LiteSeg"""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        # Bước 1: Ép giảm số kênh của đặc trưng mức cao (Channel Reduction)
        self.reduce_high = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        ) if in_c != out_c else nn.Identity()

        # Bước 2: Dung hợp qua UAFM
        self.uafm = UAFM_Spatial(skip_c, out_c)
        
        # Bước 3: Refinement nhẹ (Sử dụng Depthwise Separable để giữ độ "Lite")
        self.refine = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )

    def forward(self, x_high, x_low):
        x_high = self.reduce_high(x_high)
        x_fused = self.uafm(x_high, x_low)
        return self.refine(x_fused)

# ==============================================================================
# 3. RÁP THÀNH MẠNG HOÀN CHỈNH (PP-ULITE)
# ==============================================================================
class PP_ULite(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        """Encoder (Giữ nguyên xương sống U-Lite)"""
        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)
        self.e5 = EncoderBlock(256, 512)

        """Bottle Neck"""
        self.b5 = BottleNeckBlock(512)

        """Decoder (Chuyển sang chuẩn UAFM + FLD)"""
        # Lưu ý: skip5 là 256, skip4 là 128, skip3 là 64... (Dựa vào kiến trúc EncoderBlock)
        self.d5 = FLD_DecoderBlock(in_c=512, skip_c=256, out_c=256)
        self.d4 = FLD_DecoderBlock(in_c=256, skip_c=128, out_c=128)
        self.d3 = FLD_DecoderBlock(in_c=128, skip_c=64, out_c=64)
        self.d2 = FLD_DecoderBlock(in_c=64, skip_c=32, out_c=32)
        self.d1 = FLD_DecoderBlock(in_c=32, skip_c=16, out_c=16)
        
        # Logits output (chuẩn Template)
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        """Encoder"""
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        """BottleNeck"""
        x = self.b5(x)         

        """Decoder với Spatial Attention"""
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        
        return self.conv_out(x)

def build_model(num_classes=1):
    return PP_ULite(num_classes=num_classes)