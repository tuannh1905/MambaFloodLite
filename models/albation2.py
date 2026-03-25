import torch
import torch.nn as nn

# ==============================================================================
# 1. NHÁNH DETAIL GUIDANCE (Trọng tâm của Mẫu Ablation 3)
# ==============================================================================
class DetailGuidance(nn.Module):
    """
    Nhánh lọc ranh giới bằng Axial 3x3.
    Thay thế cho đường Identity (cộng thẳng x) truyền thống để không mang theo nhiễu nền.
    """
    def __init__(self, dim):
        super().__init__()
        self.dg_dw_h = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding='same', groups=dim, bias=False)
        self.dg_dw_w = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding='same', groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        edges = self.dg_dw_h(x) + self.dg_dw_w(x)
        return self.bn(x + edges)

class AxialDW_DG(nn.Module):
    """
    Lõi AxialDW tiêu chuẩn (Không chia nhánh Dilation) 
    nhưng Shortcut được thay bằng Detail Guidance.
    """
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        
        # Nhánh chính (Main Path): Tích chập Axial kích thước lớn (VD: 7x7)
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)
        
        # Nhánh phụ (Shortcut): Detail Guidance
        self.dg_shortcut = DetailGuidance(dim)

    def forward(self, x):
        # Thay vì: return x + self.dw_h(x) + self.dw_w(x)
        # Ta dùng:
        return self.dg_shortcut(x) + self.dw_h(x) + self.dw_w(x)

# ==============================================================================
# 2. ENCODER, DECODER & BOTTLENECK (Khung chuẩn ULite gốc)
# ==============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        # Sử dụng AxialDW có Detail Guidance
        self.dw = AxialDW_DG(in_c, mixer_kernel=mixer_kernel)
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2, 2))
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        # Sử dụng AxialDW có Detail Guidance
        self.dw = AxialDW_DG(out_c, mixer_kernel=mixer_kernel)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        # Khung chuẩn: Dùng torch.cat thay vì UAFM
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x
    
class BottleNeckBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        # Khung chuẩn của ULite Bottleneck: 3 nhánh dilation 1, 2, 3
        # Đã thay lớp bên trong thành AxialDW_DG
        self.dw1 = AxialDW_DG(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW_DG(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW_DG(gc, mixer_kernel=(3, 3), dilation=3)

        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], dim=1)
        x = self.act(self.pw2(self.bn(x)))
        return x

# ==============================================================================
# 3. MẠNG CHÍNH (ABLATION 3: ULITE + DG SHORTCUT)
# ==============================================================================
class ULiteModel_Ablation3(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')
        
        self.e1 = EncoderBlock(16, 32, mixer_kernel=(7, 7))
        self.e2 = EncoderBlock(32, 64, mixer_kernel=(7, 7))
        self.e3 = EncoderBlock(64, 128, mixer_kernel=(7, 7))
        self.e4 = EncoderBlock(128, 256, mixer_kernel=(7, 7))
        self.e5 = EncoderBlock(256, 512, mixer_kernel=(7, 7))

        self.b5 = BottleNeckBlock(512)

        self.d5 = DecoderBlock(512, 256, mixer_kernel=(7, 7))
        self.d4 = DecoderBlock(256, 128, mixer_kernel=(7, 7))
        self.d3 = DecoderBlock(128, 64, mixer_kernel=(7, 7))
        self.d2 = DecoderBlock(64, 32, mixer_kernel=(7, 7))
        self.d1 = DecoderBlock(32, 16, mixer_kernel=(7, 7))
        
        # Raw logits
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

# ==============================================================================
# 4. HÀM TỰ ĐỘNG BUILD MODEL
# ==============================================================================
def build_model(num_classes=1):
    return ULiteModel_Ablation3(num_classes=num_classes)