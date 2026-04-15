import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# LƯU Ý TƯƠNG THÍCH ONNX / TFLITE / MCU - PICO-UNET ZERO-COPY:
# 1. ECA OPTIMIZED: Dùng Conv2d 1x1, tuyệt đối không dùng permute/reshape.
# 2. ENCODER NO-CONCAT: Dùng Conv 1x1 để nhân đôi kênh, xóa bỏ lệnh torch.cat.
# 3. DECODER NO-CONCAT: Ép kênh -> Upsample -> MicroAGFusion (Cộng). Giảm 50% FLOPs.
# 4. LARGE KERNEL BOTTLE-NECK: 2 khối DW 7x7 nối tiếp, xóa bỏ hoàn toàn SPP.
# 5. ASYMMETRIC ENCODER: e1, e2 chạy Linear (chống Peak RAM); e3, e4 chạy MultiScale.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES (ECA OPTIMIZED)
# ==============================================================================
class ECABlock_MCU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid_channels = max(8, channels // 4)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        )
        self.hardsigmoid = nn.Hardsigmoid() 

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)              
        y = self.hardsigmoid(self.conv(y))                     
        return x * y

# ==============================================================================
# 2. DECODER FUSION (MICRO ATTENTION GATE)
# ==============================================================================
class MicroAGFusion(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_c, 1, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x_up, x_skip):
        # Tạo mặt nạ không gian từ sự đồng thuận để dập nhiễu
        alpha = self.gate(x_up + x_skip) 
        x_skip_gated = x_skip * alpha
        # Cộng và trộn (0 Concat)
        return self.fuse_conv(x_up + x_skip_gated)

# ==============================================================================
# 3. PFCU BLOCKS: LINEAR (Ép RAM) & MULTI-SCALE (Tăng mIoU)
# ==============================================================================
class SquareDW(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim) 

    def forward(self, x):
        return self.bn(self.dw(x)) 

class DetailGuidance(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim) 
        
    def forward(self, x):
        return x + self.bn(self.dw(x))

class Linear_PFCU(nn.Module):
    """ Tầng nông: Chỉ chạy 1 nhánh 7x7 để ép chết Peak RAM """
    def __init__(self, dim):
        super().__init__()
        self.large_kernel = SquareDW(dim, kernel_size=7)
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim) 
        self.dg_shortcut = DetailGuidance(dim)
        self.eca = ECABlock_MCU(dim) 
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        context = self.large_kernel(x)
        fused_context = self.bn_fuse(self.pw_fuse(context))
        guided_details = self.dg_shortcut(x)
        return self.eca(self.act(fused_context + guided_details))

class MultiScale_PFCU_DG(nn.Module):
    """ Tầng sâu: Phân nhánh 3-5-7 để tối đa mIoU """
    def __init__(self, dim):
        super().__init__()
        self.branch_3 = SquareDW(dim, kernel_size=3)
        self.branch_5 = SquareDW(dim, kernel_size=5)
        self.branch_7 = SquareDW(dim, kernel_size=7)
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim) 
        self.dg_shortcut = DetailGuidance(dim)
        self.eca = ECABlock_MCU(dim) 
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        b3, b5, b7 = self.branch_3(x), self.branch_5(x), self.branch_7(x)
        fused_context = self.bn_fuse(self.pw_fuse(b3 + b5 + b7))
        guided_details = self.dg_shortcut(x)
        return self.eca(self.act(fused_context + guided_details))

# ==============================================================================
# 4. ENCODER, DECODER & LINEAR BOTTLE-NECK (ZERO-COPY)
# ==============================================================================
class EncoderBlock_Linear(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pfcu_dg = Linear_PFCU(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))
        
        # ✓ ĐÃ TỐI ƯU: Dùng Conv 1x1 nhân kênh thay vì torch.cat
        self.pw = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        skip = self.pfcu_dg(x) 
        out = self.pw(self.down_pool(skip))
        return out, skip

class EncoderBlock_MultiScale(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pfcu_dg = MultiScale_PFCU_DG(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))
        
        # ✓ ĐÃ TỐI ƯU: Dùng Conv 1x1 nhân kênh thay vì torch.cat
        self.pw = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        skip = self.pfcu_dg(x) 
        out = self.pw(self.down_pool(skip))
        return out, skip

class LightDecoderBlock_AG(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        # ✓ BÍ QUYẾT TỐI ƯU FLOPs: Ép số kênh (Ví dụ 256 -> 128) TRƯỚC khi phóng to
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, skip_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_c),
            nn.ReLU6(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # MicroAGFusion dùng phép cộng, không Concat
        self.fusion = MicroAGFusion(in_c=skip_c, out_c=out_c) 

        gc = max(out_c // 4, 4)
        self.pw_down = nn.Sequential(nn.Conv2d(out_c, gc, 1, bias=False), nn.BatchNorm2d(gc), nn.ReLU6(inplace=True))
        self.refine_spatial = SquareDW(gc, kernel_size=5)
        self.pw_up  = nn.Sequential(nn.Conv2d(gc, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU6(inplace=True))

    def forward(self, x, skip):
        x = self.reduce(x)  # Ép kênh trước
        x = self.up(x)      # Phóng to sau (tiết kiệm 50% RAM/FLOPs)
        fused = self.fusion(x, skip)
        feat = self.pw_down(fused)
        feat = self.refine_spatial(feat)
        return self.pw_up(feat)

class LinearBottleneck_LargeKernel(nn.Module):
    """ ✓ ĐÃ TỐI ƯU: Đập bỏ 4 nhánh của SPP, dùng 2 nhánh 7x7 siêu tốc """
    def __init__(self, dim):
        super().__init__()
        self.dw1 = SquareDW(dim, kernel_size=7)
        self.dw2 = SquareDW(dim, kernel_size=7)
        self.eca = ECABlock_MCU(dim)
        self.pw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        out = self.dw1(x)
        out = self.dw2(out)
        out = self.eca(out)
        return self.pw(out) + x 

# ==============================================================================
# 5. MẠNG CHÍNH (PICOUNET EDGE FINAL)
# ==============================================================================
class PicoUNet_Edge(nn.Module):
    def __init__(self, num_classes=1, input_size=256):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"PicoUNet yêu cầu input_size chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        # Asymmetric Encoder
        self.e1 = EncoderBlock_Linear(16,  32)      # Nông: Linear
        self.e2 = EncoderBlock_Linear(32,  64)      # Nông: Linear
        self.e3 = EncoderBlock_MultiScale(64,  128) # Sâu: MultiScale
        self.e4 = EncoderBlock_MultiScale(128, 256) # Sâu: MultiScale

        # Linear Bottleneck (Không SPP)
        self.b4 = LinearBottleneck_LargeKernel(256)

        # Decoder No-Concat: Tham số (in_c, skip_c, out_c)
        self.d4 = LightDecoderBlock_AG(256, 128, 128)
        self.d3 = LightDecoderBlock_AG(128, 64,  64)
        self.d2 = LightDecoderBlock_AG(64,  32,  32)
        self.d1 = LightDecoderBlock_AG(32,  16,  16)

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
        
        return self.conv_out(x)

def build_model(num_classes=1, input_size=256):
    return PicoUNet_Edge(num_classes=num_classes, input_size=input_size)