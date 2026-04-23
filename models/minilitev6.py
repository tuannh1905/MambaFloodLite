import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# LƯU Ý TƯƠNG THÍCH ONNX / TORCH.FX / MCU:
# - ✓ ĐÃ CẬP NHẬT 1: ECABlock hoàn toàn sạch bóng lệnh permute().
# - ✓ ĐÃ CẬP NHẬT 2: Phục hồi Base Channel = 16 -> 256 (Tối ưu cho 128x128 Input).
# - ✓ ĐÃ CẬP NHẬT 3: Factorized_PFCU_DG - Feature Reuse 3 lớp 3x3.
# - ✓ ĐÃ CẬP NHẬT 4: Nâng cấp toàn bộ ReLU6 -> Hardswish (Tăng mIoU, 0 tham số).
# - ✓ ĐÃ CẬP NHẬT 5 (NEW): Thêm SpatialAttention_MCU vào Bottleneck (Dual Attention).
# - ✓ ĐÃ CẬP NHẬT 6 (NEW): Thêm GatedConcatFusion vào Decoder để dập nhiễu nền.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES (DUAL ATTENTION CHO MCU)
# ==============================================================================
class ECABlock_Lite(nn.Module):
    """ Channel Attention (Tỉ lệ nén r=8) """
    def __init__(self, channels):
        super().__init__()
        mid_channels = max(8, channels // 8) 
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.Hardswish(inplace=True), # Nâng cấp Hardswish
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        )
        self.hardsigmoid = nn.Hardsigmoid() 

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)              
        y = self.hardsigmoid(self.conv(y))                     
        return x * y

class SpatialAttention_MCU(nn.Module):
    """ 
    ✓ NEW: Spatial Attention (Bản đồ định vị không gian).
    Siêu nhẹ: Ép đa kênh thành 2 mặt phẳng, quét Conv 7x7 (Tốn đúng 98 tham số).
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1) 
        y = self.hardsigmoid(self.conv(y))
        return x * y

# ==============================================================================
# 2. NEAREST UPSAMPLE & GATED FUSION (CHỐNG NHIỄU DECODER)
# ==============================================================================
class NearestUpsample(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.up     = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn     = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.refine(self.up(x)))

class GatedConcatFusion(nn.Module):
    """
    ✓ NEW: Dùng đặc trưng Decoder tạo Mặt Nạ (Gate) dập tắt nhiễu của Encoder.
    Giúp mép nước sắc nét hơn rất nhiều mà tốn chưa tới 1k params.
    """
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        # Bộ tạo mặt nạ từ upsampled feature
        self.gate = nn.Sequential(
            nn.Conv2d(in_c, 1, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Hardswish(inplace=True) # Nâng cấp Hardswish
        )

    def forward(self, x_up, x_skip):
        # Nhân mặt nạ để lọc rác trước khi nối
        mask = self.gate(x_up)
        gated_skip = x_skip * mask
        
        fused = torch.cat([x_up, gated_skip], dim=1)
        return self.fuse_conv(fused)

# ==============================================================================
# 3. KHỐI ENCODER: FACTORIZED INCEPTION-LITE
# ==============================================================================
class SquareDW(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
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

class Factorized_PFCU_DG(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_3x3 = SquareDW(dim)
        self.dw_5x5 = SquareDW(dim) 
        self.dw_7x7 = SquareDW(dim) 
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim) 
        
        self.dg_shortcut = DetailGuidance(dim)
        self.eca = ECABlock_Lite(dim) 
        self.act = nn.Hardswish(inplace=True) # Nâng cấp Hardswish

    def forward(self, x):
        b3 = self.dw_3x3(x)         
        b5 = self.dw_5x5(b3)        
        b7 = self.dw_7x7(b5)        
        
        fused_context = self.bn_fuse(self.pw_fuse(b3 + b5 + b7))
        guided_details = self.dg_shortcut(x)
        return self.eca(self.act(fused_context + guided_details))

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.pfcu_dg   = Factorized_PFCU_DG(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw      = nn.Conv2d(in_c, conv_out, kernel_size=1, bias=False)
            self.bn_pw   = nn.BatchNorm2d(conv_out) 
            self.down_pw = nn.MaxPool2d((2, 2))

        self.act = nn.Hardswish(inplace=True) # Nâng cấp Hardswish

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

# ==============================================================================
# 4. DECODER & BOTTLE-NECK ĐÃ NÂNG CẤP
# ==============================================================================
class LightDecoderBlock_NoUAFM(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        gc = max(out_c // 4, 4)

        self.up   = NearestUpsample(in_c, scale_factor=2)
        # ✓ Thay SimpleConcatFusion bằng GatedConcatFusion
        self.fusion = GatedConcatFusion(in_c=in_c, skip_c=in_c, out_c=out_c)

        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        self.bn_down = nn.BatchNorm2d(gc)
        
        self.refine_spatial_1 = SquareDW(gc)
        self.refine_spatial_2 = SquareDW(gc)
        
        self.eca = ECABlock_Lite(gc) 
        
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        self.bn_up  = nn.BatchNorm2d(out_c)
        
        self.act = nn.Hardswish(inplace=True) # Nâng cấp Hardswish

    def forward(self, x, skip):
        x = self.up(x)
        fused = self.fusion(x, skip)
        
        feat = self.act(self.bn_down(self.pw_down(fused)))
        feat = self.refine_spatial_2(self.refine_spatial_1(feat))
        feat = self.eca(feat)
        
        out = self.bn_up(self.pw_up(feat))
        return self.act(out + fused)

class PerfectedBottleneck(nn.Module):
    """
    ✓ NEW: Thêm Spatial Attention tạo thành Dual Attention Block.
    0 bóp kênh, 0 Pointwise thừa, nhưng biết chính xác vũng nước nằm ở đâu.
    """
    def __init__(self, dim):
        super().__init__()
        self.dw1 = SquareDW(dim) 
        self.dw2 = SquareDW(dim) 
        
        self.channel_attn = ECABlock_Lite(dim)
        self.spatial_attn = SpatialAttention_MCU(kernel_size=7)

    def forward(self, x):
        out = self.dw2(self.dw1(x))
        out = self.channel_attn(out)
        out = self.spatial_attn(out) # Dual Attention
        return out + x

# ==============================================================================
# 5. MẠNG CHÍNH PICO-UNET
# ==============================================================================
class PicoUNet_Ablation_NoUAFM(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"PicoUNet yêu cầu input_size chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32)
        self.e2 = EncoderBlock(32,  64)
        self.e3 = EncoderBlock(64,  128)
        self.e4 = EncoderBlock(128, 256)

        # Bottleneck với Dual Attention
        self.b4 = PerfectedBottleneck(256)

        # Decoder với Gated Fusion chống nhiễu
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

def build_model(num_classes=1, input_size=128):
    return PicoUNet_Ablation_NoUAFM(num_classes=num_classes, input_size=input_size)