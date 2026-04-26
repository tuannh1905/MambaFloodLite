import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 0. CUSTOM ACTIVATIONS CHO ONNX OPSET 11 (VACCINE)
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
    if act_type == 'hswish':
        return CustomHardswish()
    return nn.ReLU6(inplace=True)

# ==============================================================================
# PICO-UNET V4: BẢN "SMART MUSCLE" (~300K PARAMS - LOW FLOPS)
# - ✓ ADDITIVE FUSION: Giữ nguyên để cứu RAM.
# - ✓ CUT ATTENTION: Bỏ hoàn toàn ECA ở Encoder để giảm Overhead băng thông.
# - ✓ DUAL-SCALE ENCODER: E1, E2 chỉ dùng 3x3 + 5x5 để giảm GFLOPs ở độ phân giải cao.
# - ✓ BƠM PARAMS: Cấu hình kênh 32 -> 64 -> 128 -> 192.
# - ✓ FIX BUG: Khai báo chuẩn xác skip_c cho Decoder V4.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES
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
        # VACCINE 2: Dùng AdaptiveAvgPool thay vì torch.mean
        y = F.adaptive_avg_pool2d(x, 1) 
        y = self.hardsigmoid(self.conv(y))
        return x * y

class SpatialAttention_MCU(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        # Đã thay thế bằng CustomHardsigmoid
        self.hardsigmoid = CustomHardsigmoid()

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
# 3. KHỐI ENCODER (CHIA LÀM 2 LOẠI: DUAL-SCALE VÀ MULTI-SCALE)
# ==============================================================================
class DualScale_PFCU_DG(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw_3x3 = SquareDW(dim)
        self.dw_5x5 = SquareDW(dim) 
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        b3 = self.dw_3x3(x)        
        b5 = self.dw_5x5(b3)        
        
        fused = self.bn_fuse(self.pw_fuse(b3 + b5))
        return self.act(fused + x)

class MultiScale_PFCU_DG(nn.Module):
    def __init__(self, dim, act_type='hswish'):
        super().__init__()
        self.dw_3x3 = SquareDW(dim)
        self.dw_5x5 = SquareDW(dim) 
        self.dw_7x7 = SquareDW(dim) 
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        b3 = self.dw_3x3(x)        
        b5 = self.dw_5x5(b3)        
        b7 = self.dw_7x7(b5)        
        
        fused = self.bn_fuse(self.pw_fuse(b3 + b5 + b7))
        return self.act(fused + x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, is_deep=False, act_type='relu6'):
        super().__init__()
        if is_deep:
            self.pfcu_dg = MultiScale_PFCU_DG(in_c, act_type)
        else:
            self.pfcu_dg = DualScale_PFCU_DG(in_c, act_type)
            
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
# 4. DECODER (ADDITIVE) & BOTTLE-NECK
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
            
            SquareDW(gc, kernel_size=5), 
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
        self.dw_3x3 = SquareDW(dim)
        self.dw_5x5 = SquareDW(dim) 
        self.dw_7x7 = SquareDW(dim) 
        
        self.channel_attn = ECABlock(dim, act_type)
        self.spatial_attn = SpatialAttention_MCU(kernel_size=3)

    def forward(self, x):
        d1 = self.dw_3x3(x)        
        d2 = self.dw_5x5(d1)        
        d3 = self.dw_7x7(d2)        
        
        fused = d1 + d2 + d3
        
        out = self.channel_attn(fused)
        out = self.spatial_attn(out)
        
        return x + out

# ==============================================================================
# 5. MẠNG CHÍNH PICO-UNET V4 (SMART MUSCLE)
# ==============================================================================
class PicoUNet_v4_Edge(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"Input_size phải chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        self.e1 = EncoderBlock(32, 64,  is_deep=False, act_type='relu6')   
        self.e2 = EncoderBlock(64, 128, is_deep=False, act_type='relu6')   
        self.e3 = EncoderBlock(128, 192, is_deep=True, act_type='hswish') 
        self.e4 = EncoderBlock(192, 192, is_deep=True, act_type='hswish') 
        
        self.bottleneck = SerialMultiScaleBottleneck(192, act_type='hswish')
        
        self.d4 = AdditiveDecoderBlock(in_c=192, skip_c=192, out_c=128, act_type='hswish') 
        self.d3 = AdditiveDecoderBlock(in_c=128, skip_c=192, out_c=64,  act_type='hswish')  
        self.d2 = AdditiveDecoderBlock(in_c=64,  skip_c=128, out_c=32,  act_type='hswish')   
        self.d1 = AdditiveDecoderBlock(in_c=32,  skip_c=64,  out_c=16,  act_type='hswish')   
        
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

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
    return PicoUNet_v4_Edge(num_classes=num_classes, input_size=input_size)