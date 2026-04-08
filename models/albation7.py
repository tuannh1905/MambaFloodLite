import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# LƯU Ý TƯƠNG THÍCH ONNX / TORCH.FX / MCU:
# - MULTI-SCALE SQUARE DW: Dùng (3x3, 5x5, 7x7) thay chập chéo để tối ưu NPU.
# - NEAREST UPSAMPLE: Loại bỏ Bilinear.
# - ✓ ĐÃ CẬP NHẬT: Thay Sigmoid bằng Hardsigmoid để NPU dịch bit (siêu nhanh).
# - ✓ ĐÃ CẬP NHẬT: Thay AdaptiveAvgPool2d(1) bằng torch.mean() chống lỗi biên dịch.
# - ✓ ĐÃ CẬP NHẬT: Xóa bỏ abs() trong TinyUAFM_v2, thay bằng torch.max() (CBAM style).
# - ✓ ĐÃ CẬP NHẬT: Dùng biến 'B' explicit trong ECABlock để chốt Static Tensor Arena.
# - ✓ ĐÃ CẬP NHẬT (NEW 1): Dùng permute() thay vì reshape() trong ECABlock.
# - ✓ ĐÃ CẬP NHẬT (NEW 2): Dùng LightDecoderBlock siêu nhẹ ở Decoder.
# ==============================================================================

# ==============================================================================
# ABLATION STUDY: BỎ STATIC SPP BOTTLE-NECK
# - CHỈNH SỬA: Thay BottleNeckBlock_Static bằng StandardBottleneck (2 lớp Conv 3x3).
# - GIỮ LẠI: MultiScale_PFCU_DG, TinyUAFM_v2, LightDecoderBlock.
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
# 2. NEAREST UPSAMPLE & TINY-UAFM_V2
# ==============================================================================
class NearestUpsample(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.up     = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn     = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.refine(self.up(x)))

class TinyUAFM_v2(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.reduce_up   = nn.Conv2d(in_c,   out_c, 1, bias=False) if in_c   != out_c else nn.Identity()
        self.reduce_skip = nn.Conv2d(skip_c, out_c, 1, bias=False) if skip_c != out_c else nn.Identity()
        
        self.alpha_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), 
            nn.Hardsigmoid() 
        )
        self.beta_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), 
            nn.Hardsigmoid() 
        )
        self.channel_att = ECABlock(out_c)

    def forward(self, x_up, x_skip):
        x_up   = self.reduce_up(x_up)
        x_skip = self.reduce_skip(x_skip)
        
        spatial_input = torch.cat([
            torch.mean(x_up, dim=1, keepdim=True),
            torch.max(x_skip, dim=1, keepdim=True)[0] 
        ], dim=1)
        
        alpha = self.alpha_conv(spatial_input)
        beta  = self.beta_conv(spatial_input)
        
        out = x_up * alpha + x_skip * beta
        return self.channel_att(out)

# ==============================================================================
# 3. SQUARE-PFCU-DG BLOCK (MCU NATIVE)
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
# 4. ENCODER, DECODER CHUẨN & BOTTLE-NECK ABLATION
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

class LightDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        gc = max(out_c // 4, 4)

        self.up   = NearestUpsample(in_c, scale_factor=2)
        self.uafm = TinyUAFM_v2(in_c=in_c, skip_c=in_c, out_c=out_c)

        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        self.bn_down = nn.BatchNorm2d(gc)
        
        self.refine_spatial = SquareDW(gc, kernel_size=5)
        self.eca = ECABlock(gc) 
        
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        self.bn_up  = nn.BatchNorm2d(out_c)
        
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        fused = self.uafm(x, skip)
        
        feat = self.act(self.bn_down(self.pw_down(fused)))
        feat = self.refine_spatial(feat)
        feat = self.eca(feat)
        
        out = self.bn_up(self.pw_up(feat))
        return self.act(out + fused)

class StandardBottleneck(nn.Module):
    """
    ✓ ABLATION: Thay thế Static SPP bằng 2 lớp ConvBNReLU 3x3 nối tiếp.
    Mất đi Global Context Receptive Field.
    """
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# ==============================================================================
# 5. MẠNG CHÍNH
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

        # ✓ ĐÃ ĐỔI SANG BẢN ABLATION: StandardBottleneck
        self.b4 = StandardBottleneck(256)

        self.d4 = LightDecoderBlock(256, 128)
        self.d3 = LightDecoderBlock(128, 64)
        self.d2 = LightDecoderBlock(64,  32)
        self.d1 = LightDecoderBlock(32,  16)

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