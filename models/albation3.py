import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# LƯU Ý TƯƠNG THÍCH ONNX / TORCH.FX / MCU:
# - MULTI-SCALE SQUARE DW: Dùng (3x3, 5x5, 7x7) thay chập chéo để tối ưu NPU.
# - NEAREST UPSAMPLE: Loại bỏ Bilinear.
# - ✓ ĐÃ CẬP NHẬT: Thay Sigmoid bằng Hardsigmoid để NPU dịch bit (siêu nhanh).
# - ✓ ĐÃ CẬP NHẬT: Thay AdaptiveAvgPool2d(1) bằng torch.mean() chống lỗi biên dịch.
# - ✓ ĐÃ CẬP NHẬT: Dùng biến 'B' explicit trong ECABlock để chốt Static Tensor Arena.
# - ✓ ĐÃ CẬP NHẬT: Dùng permute() thay vì reshape() trong ECABlock.
# ==============================================================================

# ==============================================================================
# ABLATION STUDY 2: BỎ KHỐI SPP TĨNH (No SPP)
# - CHỈNH SỬA: Thay khối BottleNeck_Static bằng Inverted Residual tiêu chuẩn.
# - GIỮ LẠI: Multi-Scale & Detail Guidance (Encoder), Simple Fusion & DW 5x5 (Decoder).
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES (ECA)
# ==============================================================================
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), padding=(0, k_size//2), bias=False)
        self.hardsigmoid = nn.Hardsigmoid() 

    def forward(self, x):
        B, C, _, _ = x.shape 
        y = torch.mean(x, dim=[2, 3], keepdim=True)              
        y = y.permute(0, 2, 3, 1) # An toàn cho ONNX
        y = self.hardsigmoid(self.conv(y))                     
        y = y.permute(0, 3, 1, 2) 
        return x * y

# ==============================================================================
# 2. NEAREST UPSAMPLE & SIMPLE FUSION (BASELINE MỚI)
# ==============================================================================
class NearestUpsample(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.up     = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn     = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.refine(self.up(x)))

class SimpleConcatFusion(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x_up, x_skip):
        fused = torch.cat([x_up, x_skip], dim=1)
        return self.fuse_conv(fused)

# ==============================================================================
# 3. SQUARE-PFCU-DG BLOCK (TRỞ LẠI BẢN GỐC ĐA TỶ LỆ)
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

class MultiScale_PFCU_DG(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.branch_3 = SquareDW(dim, kernel_size=3)
        self.branch_5 = SquareDW(dim, kernel_size=5)
        self.branch_7 = SquareDW(dim, kernel_size=7)
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim) 
        
        self.dg_shortcut = DetailGuidance(dim)
        self.eca = ECABlock(dim) 
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        b3, b5, b7 = self.branch_3(x), self.branch_5(x), self.branch_7(x)
        fused_context = self.bn_fuse(self.pw_fuse(b3 + b5 + b7))
        guided_details = self.dg_shortcut(x)
        return self.eca(self.act(fused_context + guided_details))

# ==============================================================================
# 4. ENCODER & DECODER
# ==============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.pfcu_dg   = MultiScale_PFCU_DG(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw      = nn.Conv2d(in_c, conv_out, kernel_size=1, bias=False)
            self.bn_pw   = nn.BatchNorm2d(conv_out) 
            self.down_pw = nn.MaxPool2d((2, 2))

        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        feat = self.pfcu_dg(x) 

        if self.same_channels:
            return self.act(self.down_pool(feat)), feat
        else:
            feat_pw = self.bn_pw(self.pw(feat)) 
            skip = torch.cat([feat, feat_pw], dim=1) 
            
            pool_feat = self.down_pool(feat)
            pool_pw   = self.down_pw(feat_pw)
            x = self.act(torch.cat([pool_feat, pool_pw], dim=1))
            return x, skip

class LightDecoderBlock_NoUAFM(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        gc = max(out_c // 4, 4)

        self.up   = NearestUpsample(in_c, scale_factor=2)
        self.fusion = SimpleConcatFusion(in_c=in_c, skip_c=in_c, out_c=out_c)

        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        self.bn_down = nn.BatchNorm2d(gc)
        
        self.refine_spatial = SquareDW(gc, kernel_size=5)
        self.eca = ECABlock(gc) 
        
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        self.bn_up  = nn.BatchNorm2d(out_c)
        
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        fused = self.fusion(x, skip)
        
        feat = self.act(self.bn_down(self.pw_down(fused)))
        feat = self.refine_spatial(feat)
        feat = self.eca(feat)
        
        out = self.bn_up(self.pw_up(feat))
        return self.act(out + fused)

# ==============================================================================
# 5. KHỐI BOTTLENECK ABLATION (KHÔNG CÓ SPP)
# ==============================================================================
class BottleNeckBlock_NoSPP(nn.Module):
    """
    ✓ ABLATION BOTTLENECK: Bỏ hoàn toàn Static Multi-Scale SPP.
    Thay thế bằng Inverted Residual tiêu chuẩn: Expand -> DW 5x5 -> ECA -> Squeeze -> Add.
    """
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden_dim = dim * expansion # Phình to kênh (Ví dụ: 256 -> 512)

        # 1. Expand (Tăng kênh)
        self.expand = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        # 2. Spatial Refine (Trộn không gian bằng kernel lớn)
        self.spatial = SquareDW(hidden_dim, kernel_size=5)
        
        # 3. Channel Attention (Lọc nhiễu)
        self.eca = ECABlock(hidden_dim)
        
        # 4. Squeeze (Ép kênh về lại ban đầu - Dùng Linear Activation)
        self.squeeze = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.spatial(out)
        out = self.eca(out)
        out = self.squeeze(out)
        return out + x # Cộng bù Residual

# ==============================================================================
# 6. MẠNG CHÍNH: ABLATION KHÔNG CÓ SPP BOTTLE-NECK
# ==============================================================================
class PicoUNet_Ablation_NoSPP(nn.Module):
    def __init__(self, num_classes=1, input_size=256):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"PicoUNet yêu cầu input_size chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32)
        self.e2 = EncoderBlock(32,  64)
        self.e3 = EncoderBlock(64,  128)
        self.e4 = EncoderBlock(128, 256)

        # ✓ ĐÃ THAY THẾ: Dùng Bottleneck Inverted Residual (Không có SPP)
        self.b4 = BottleNeckBlock_NoSPP(256, expansion=2)

        self.d4 = LightDecoderBlock_NoUAFM(256, 128)
        self.d3 = LightDecoderBlock_NoUAFM(128, 64)
        self.d2 = LightDecoderBlock_NoUAFM(64,  32)
        self.d1 = LightDecoderBlock_NoUAFM(32,  16)

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
    return PicoUNet_Ablation_NoSPP(num_classes=num_classes, input_size=input_size)