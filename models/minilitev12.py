import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 0. CUSTOM ACTIVATIONS
# ==============================================================================
class CustomHardsigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x):
        return self.relu6(x + 3.0) / 6.0

class CustomHardswish(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardsigmoid = CustomHardsigmoid()
    def forward(self, x):
        return x * self.hardsigmoid(x)

def get_activation(act_type):
    if act_type == 'hswish': return CustomHardswish()
    return nn.ReLU6(inplace=True)

# ==============================================================================
# 1. ATTENTION MODULES (TỪ V4)
# ==============================================================================
class ECABlock(nn.Module):
    def __init__(self, channels, act_type='hswish'):
        super().__init__()
        mid_channels = max(8, channels // 4)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            get_activation(act_type),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        )
        self.hardsigmoid = CustomHardsigmoid()

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1) 
        return x * self.hardsigmoid(self.conv(y))

class SpatialAttention_MCU(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.hardsigmoid = CustomHardsigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1) 
        return x * self.hardsigmoid(self.conv(y))

# ==============================================================================
# 2. KHỐI TÍCH CHẬP VÀ UPSAMPLE
# ==============================================================================
class SquareDW(nn.Module):
    def __init__(self, dim, kernel_size=3, act_type='relu6'):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type) # Đã bọc act để chống suy biến

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))

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
# 3. ENCODER BLOCKS (V4 CHUẨN MỰC)
# ==============================================================================
class DualScale_PFCU_DG(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw_3x3 = SquareDW(dim, act_type=act_type)
        self.dw_5x5 = SquareDW(dim, act_type=act_type) 
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        b3 = self.dw_3x3(x)        
        b5 = self.dw_5x5(b3)        
        return self.act(self.bn_fuse(self.pw_fuse(b3 + b5)) + x)

class MultiScale_PFCU_DG(nn.Module):
    def __init__(self, dim, act_type='hswish'):
        super().__init__()
        self.dw_3x3 = SquareDW(dim, act_type=act_type)
        self.dw_5x5 = SquareDW(dim, act_type=act_type) 
        self.dw_7x7 = SquareDW(dim, act_type=act_type) 
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        b3 = self.dw_3x3(x)        
        b5 = self.dw_5x5(b3)        
        b7 = self.dw_7x7(b5)        
        return self.act(self.bn_fuse(self.pw_fuse(b3 + b5 + b7)) + x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, is_deep=False, act_type='relu6'):
        super().__init__()
        if is_deep: self.pfcu_dg = MultiScale_PFCU_DG(in_c, act_type)
        else:       self.pfcu_dg = DualScale_PFCU_DG(in_c, act_type)
            
        self.down_pool = nn.MaxPool2d((2, 2))
        self.same_channels = (in_c == out_c)
        if not self.same_channels:
            self.pw = nn.Sequential(
                nn.Conv2d(in_c, out_c - in_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c - in_c)
            )
            self.down_pw = nn.MaxPool2d((2, 2))
        self.act = get_activation(act_type)

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
# 4. DECODER & BOTTLE-NECK (V4 CHUẨN MỰC)
# ==============================================================================
class AdditiveDecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, act_type='hswish'):
        super().__init__()
        self.up = NearestUpsample(in_c)
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, skip_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_c)
        )
        gc = max(out_c // 4, 4)
        self.refine = nn.Sequential(
            nn.Conv2d(skip_c, gc, kernel_size=1, bias=False), 
            nn.BatchNorm2d(gc), 
            get_activation(act_type),
            SquareDW(gc, kernel_size=5, act_type=act_type), 
            ECABlock(gc, act_type), 
            nn.Conv2d(gc, out_c, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(skip_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act = get_activation(act_type)

    def forward(self, x, skip):
        fused = self.proj(self.up(x)) + skip
        return self.act(self.refine(fused) + self.shortcut(fused))

class SerialMultiScaleBottleneck(nn.Module):
    def __init__(self, dim, act_type='hswish'):
        super().__init__()
        self.dw_3x3 = SquareDW(dim, act_type=act_type)
        self.dw_5x5 = SquareDW(dim, act_type=act_type) 
        self.dw_7x7 = SquareDW(dim, act_type=act_type) 
        self.channel_attn = ECABlock(dim, act_type)
        self.spatial_attn = SpatialAttention_MCU(kernel_size=3)

    def forward(self, x):
        d1 = self.dw_3x3(x)        
        d2 = self.dw_5x5(d1)        
        d3 = self.dw_7x7(d2)        
        fused = d1 + d2 + d3
        out = self.channel_attn(fused)
        return x + self.spatial_attn(out)

# ==============================================================================
# 5. PICO-UNET V19 (FAST-STEM EDITION)
# ==============================================================================
class PicoUNet_v19_FastStem(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        self.input_size = input_size
        
        # ✓ VŨ KHÍ TỐI THƯỢNG 1: FAST STEM
        # Ép ảnh 128x128 -> 64x64 ngay từ đầu vào. GFLOPs giảm 4 lần!
        self.conv_in = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            get_activation('relu6')
        )
        
        # Trunk: 24 -> 48 -> 96 -> 128 (Giữ V4 nhưng gọn gàng hơn)
        self.e1 = EncoderBlock(24, 48,  is_deep=False, act_type='relu6')   # skip ở 64x64
        self.e2 = EncoderBlock(48, 96,  is_deep=False, act_type='relu6')   # skip ở 32x32
        self.e3 = EncoderBlock(96, 128, is_deep=True, act_type='hswish')   # skip ở 16x16
        self.e4 = EncoderBlock(128, 128, is_deep=True, act_type='hswish')  # skip ở 8x8
        
        self.bottleneck = SerialMultiScaleBottleneck(128, act_type='hswish') # Chạy ở 4x4
        
        # Decoder
        self.d4 = AdditiveDecoderBlock(in_c=128, skip_c=128, out_c=96, act_type='hswish') 
        self.d3 = AdditiveDecoderBlock(in_c=96,  skip_c=128, out_c=48, act_type='hswish')  
        self.d2 = AdditiveDecoderBlock(in_c=48,  skip_c=96,  out_c=24, act_type='hswish')   
        self.d1 = AdditiveDecoderBlock(in_c=24,  skip_c=48,  out_c=16, act_type='hswish')  
        
        # ✓ VŨ KHÍ TỐI THƯỢNG 2: MẶT NẠ 64x64
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: [B, 3, 128, 128]
        x = self.conv_in(x) # x shape: [B, 24, 64, 64]
        
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        
        x = self.bottleneck(x)
        
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        
        mask_64 = self.conv_out(x) # Mask sinh ra ở 64x64
        
        # ✓ VŨ KHÍ TỐI THƯỢNG 3: NỘI SUY (0 FLOPs)
        # Phóng to mask từ 64x64 trả về 128x128
        out = F.interpolate(mask_64, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        return out

def build_model(num_classes=1, input_size=128):
    return PicoUNet_v19_FastStem(num_classes=num_classes, input_size=input_size)