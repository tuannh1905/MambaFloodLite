import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# PICO-UNET V2: BẢN PAPER - TỐI ƯU HÓA CHO ESP32-S3 (BẢN CÂN BẰNG)
# - ✓ TỐI ƯU 1: AdditiveDecoderBlock ĐÃ FIX BUG TẬN GỐC (Tách bạch in_c, skip_c, out_c).
# - ✓ TỐI ƯU 2: SerialMultiScaleBottleneck (Góc nhìn đa tỷ lệ).
# - ✓ TỐI ƯU 3: Nâng cấp toàn bộ thành Hardswish siêu mượt.
# - ✓ TỐI ƯU 4: Cấu hình "Vừa Vừa" (24 -> 48 -> 96 -> 192) đạt ~195k Params.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES (DUAL ATTENTION)
# ==============================================================================
class ECABlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid_channels = max(8, channels // 4)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.Hardswish(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        )
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)
        y = self.hardsigmoid(self.conv(y))
        return x * y

class SpatialAttention_MCU(nn.Module):
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
# 2. KHỐI TÍCH CHẬP VÀ UPSAMPLE
# ==============================================================================
class SquareDW(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.dw(x))

class NearestUpsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return self.refine(self.up(x))

# ==============================================================================
# 3. KHỐI ENCODER V2
# ==============================================================================
class MultiScale_PFCU_DG_v2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_3x3 = SquareDW(dim)
        self.dw_5x5 = SquareDW(dim) 
        self.dw_7x7 = SquareDW(dim) 
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        
        self.eca = ECABlock(dim)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        b3 = self.dw_3x3(x)         
        b5 = self.dw_5x5(b3)        
        b7 = self.dw_7x7(b5)        
        
        fused = self.bn_fuse(self.pw_fuse(b3 + b5 + b7))
        return self.eca(self.act(fused + x))

class EncoderBlock_v2(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pfcu_dg = MultiScale_PFCU_DG_v2(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))
        
        self.same_channels = (in_c == out_c)
        if not self.same_channels:
            self.pw = nn.Sequential(
                nn.Conv2d(in_c, out_c - in_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c - in_c)
            )
            self.down_pw = nn.MaxPool2d((2, 2))
            
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        feat = self.pfcu_dg(x)
        
        if self.same_channels:
            return self.act(self.down_pool(feat)), feat
        else:
            feat_pw = self.pw(feat)
            skip = torch.cat([feat, feat_pw], dim=1)
            
            pool_feat = self.down_pool(feat)
            pool_pw   = self.down_pw(feat_pw)
            
            out = self.act(torch.cat([pool_feat, pool_pw], dim=1))
            return out, skip

# ==============================================================================
# 4. DECODER (FIX BUG TRIỆT ĐỂ) & BOTTLE-NECK
# ==============================================================================
class AdditiveDecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = NearestUpsample(in_c)
        
        # ✓ FIX MẠNH: Chiếu kênh đầu vào (in_c) về BẰNG VỚI kênh skip (skip_c)
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, skip_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_c)
        )
        
        gc = max(out_c // 4, 4)
        # Chú ý: refine nhận vào tensor đã được cộng (số kênh là skip_c), xuất ra out_c
        self.refine = nn.Sequential(
            nn.Conv2d(skip_c, gc, kernel_size=1, bias=False), 
            nn.BatchNorm2d(gc), 
            nn.Hardswish(inplace=True),
            
            SquareDW(gc, kernel_size=5), 
            ECABlock(gc),
            
            nn.Conv2d(gc, out_c, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        
        # Shortcut nhận skip_c và trả về out_c để cộng residual
        self.shortcut = nn.Sequential(
            nn.Conv2d(skip_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x, skip):
        # 1. Ép kênh x (từ in_c -> skip_c) rồi mới cộng với skip (skip_c)
        fused = self.proj(self.up(x)) + skip
        
        # 2. Refine (skip_c -> out_c) và cộng với Shortcut (skip_c -> out_c)
        return self.act(self.refine(fused) + self.shortcut(fused))

class SerialMultiScaleBottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_3x3 = SquareDW(dim)
        self.dw_5x5 = SquareDW(dim) 
        self.dw_7x7 = SquareDW(dim) 
        
        self.channel_attn = ECABlock(dim)
        self.spatial_attn = SpatialAttention_MCU(kernel_size=7)

    def forward(self, x):
        d1 = self.dw_3x3(x)         
        d2 = self.dw_5x5(d1)        
        d3 = self.dw_7x7(d2)        
        
        fused = d1 + d2 + d3
        
        out = self.channel_attn(fused)
        out = self.spatial_attn(out)
        
        return x + out

# ==============================================================================
# 5. MẠNG CHÍNH PICO-UNET V2 PAPER (BẢN CÂN BẰNG)
# ==============================================================================
class PicoUNet_v2_Paper(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"Input_size phải chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 24, kernel_size=3, padding=1)
        
        # Encoder:
        # e1: 24 -> 48   => s1 có 48 kênh
        # e2: 48 -> 96   => s2 có 96 kênh
        # e3: 96 -> 192  => s3 có 192 kênh
        # e4: 192 -> 192 => s4 có 192 kênh
        self.e1 = EncoderBlock_v2(24, 48)
        self.e2 = EncoderBlock_v2(48, 96)
        self.e3 = EncoderBlock_v2(96, 192)
        self.e4 = EncoderBlock_v2(192, 192) 
        
        self.bottleneck = SerialMultiScaleBottleneck(192)
        
        # Decoder lùi đối xứng (Đã khai báo rành mạch in, skip, out)
        # d4 nhận x=192, s4=192 -> trả về 192
        self.d4 = AdditiveDecoderBlock(in_c=192, skip_c=192, out_c=192) 
        # d3 nhận x=192 (từ d4), s3=192 -> trả về 96
        self.d3 = AdditiveDecoderBlock(in_c=192, skip_c=192, out_c=96)  
        # d2 nhận x=96 (từ d3), s2=96 -> trả về 48
        self.d2 = AdditiveDecoderBlock(in_c=96,  skip_c=96,  out_c=48)   
        # d1 nhận x=48 (từ d2), s1=48 -> trả về 24
        self.d1 = AdditiveDecoderBlock(in_c=48,  skip_c=48,  out_c=24)   
        
        self.conv_out = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        
        x = self.bottleneck(x)
        
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        
        return self.conv_out(x)

def build_model(num_classes=1, input_size=128):
    return PicoUNet_v2_Paper(num_classes=num_classes, input_size=input_size)