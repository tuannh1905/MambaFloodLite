import torch
import torch.nn as nn

# ==============================================================================
# 1. CORE MODULES
# ==============================================================================
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        # Tính toán padding để đảm bảo output luôn cùng shape với mọi dilation
        pad_h = (h + (h - 1) * (dilation - 1)) // 2
        pad_w = (w + (w - 1) * (dilation - 1)) // 2
        
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding=(pad_h, 0), groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding=(0, pad_w), groups=dim, dilation=dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

# Khối thay đổi cốt lõi cho Ablation Study 2
class Axial_PFCU_Identity(nn.Module):
    """
    Parallel Factorized Convolutional Unit (PFCU) 
    - 3 nhánh Dilation: 1, 2, 5 (Mở rộng bối cảnh)
    - Shortcut: Identity thuần túy (Cộng thẳng x, không có Lọc chi tiết)
    """
    def __init__(self, dim, mixer_kernel=(7, 7)):
        super().__init__()
        # Nhánh đa quy mô
        self.branch_r1 = AxialDW(dim, mixer_kernel, dilation=1)
        self.branch_r2 = AxialDW(dim, mixer_kernel, dilation=2)
        self.branch_r5 = AxialDW(dim, mixer_kernel, dilation=5)
        
        # Pointwise Conv để trộn thông tin từ 3 nhánh
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        b1 = self.branch_r1(x)
        b2 = self.branch_r2(x)
        b5 = self.branch_r5(x)
        
        # Gộp thông tin bối cảnh
        fused = self.bn_fuse(self.pw_fuse(b1 + b2 + b5))
        
        # Shortcut: Identity (Cộng thẳng input x)
        return self.act(fused + x)

# ==============================================================================
# 2. ENCODER, DECODER & BOTTLENECK (Thay thế DW bằng PFCU)
# ==============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        # Thay thế AxialDW đơn thuần bằng khối PFCU đa quy mô
        self.pfcu = Axial_PFCU_Identity(in_c, mixer_kernel=mixer_kernel)
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2, 2))
        self.act = nn.GELU()

    def forward(self, x):
        # Trích xuất bối cảnh bằng PFCU trước khi MaxPool
        skip = self.bn(self.pfcu(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        # Sử dụng PFCU để khôi phục chi tiết tốt hơn AxialDW đơn
        self.pfcu = Axial_PFCU_Identity(out_c, mixer_kernel=mixer_kernel)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.pfcu(self.bn(self.pw(x)))))
        return x
    
class BottleNeckBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        # Giữ nguyên Bottleneck gốc vì nó vốn đã chia nhánh Dilation
        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)

        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], dim=1)
        x = self.act(self.pw2(self.bn(x)))
        return x

# ==============================================================================
# 3. MẠNG CHÍNH (ABLATION 2: ULITE + PFCU_IDENTITY)
# ==============================================================================
class ULiteModel_Ablation2(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')
        
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)
        self.e5 = EncoderBlock(256, 512)

        self.b5 = BottleNeckBlock(512)

        self.d5 = DecoderBlock(512, 256)
        self.d4 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)
        self.d1 = DecoderBlock(32, 16)
        
        # Raw logits output
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
    return ULiteModel_Ablation2(num_classes=num_classes)