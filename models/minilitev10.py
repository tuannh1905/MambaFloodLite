import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# PICO-UNET V2: BẢN PAPER - TỐI ƯU HÓA CHO ESP32-S3 (BẢN TỐC ĐỘ CAO)
# - ✓ CHIẾN THUẬT 2: Spatial Attention giảm kernel xuống 3x3 siêu nhẹ.
# - ✓ CHIẾN THUẬT 3: Hybrid Activation (e1, e2 dùng ReLU6; e3, e4, Decoder dùng Hardswish).
# - ✓ CHIẾN THUẬT 4: Đơn giản hóa Decoder bằng Concat Fusion truyền thống.
# - ✓ CHIẾN THUẬT 5 (NEW): Bóp kênh đáy mạng về 128 (24 -> 48 -> 96 -> 128) để tối đa FPS.
# - ✓ CHIẾN THUẬT 6 (ONNX FIX): Tự định nghĩa Hardsigmoid/Hardswish để ép xuất Opset 11.
# ==============================================================================

# ==============================================================================
# 0. CUSTOM ACTIVATIONS (DÀNH RIÊNG CHO ONNX EXPORT OPSET 11)
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

# Hàm hỗ trợ chọn Activation linh hoạt
def get_activation(act_type):
    if act_type == 'hswish':
        # Dùng hàm Custom để không bị lỗi version_converter
        return CustomHardswish()
    return nn.ReLU6(inplace=True)

# ==============================================================================
# 1. ATTENTION MODULE
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
        self.hardsigmoid = nn.Hardsigmoid()
    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)
        y = self.hardsigmoid(self.conv(y))
        return x * y

# 2. SpatialAttention_MCU: Dùng amax để tránh lỗi indices
class SpatialAttention_MCU(nn.Module):
    def __init__(self, kernel_size=3):
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
# 3. KHỐI ENCODER V2 (HỖ TRỢ HYBRID ACTIVATION)
# ==============================================================================
class MultiScale_PFCU_DG_v2(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw_3x3 = SquareDW(dim)
        self.dw_5x5 = SquareDW(dim) 
        self.dw_7x7 = SquareDW(dim) 
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        
        self.eca = ECABlock(dim, act_type)
        self.act = get_activation(act_type)

    def forward(self, x):
        b3 = self.dw_3x3(x)         
        b5 = self.dw_5x5(b3)        
        b7 = self.dw_7x7(b5)        
        
        fused = self.bn_fuse(self.pw_fuse(b3 + b5 + b7))
        return self.eca(self.act(fused + x))

class EncoderBlock_v2(nn.Module):
    def __init__(self, in_c, out_c, act_type='relu6'):
        super().__init__()
        self.pfcu_dg = MultiScale_PFCU_DG_v2(in_c, act_type)
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
# 4. DECODER (SIMPLE CONCAT) & BOTTLE-NECK
# ==============================================================================
class SimpleConcatFusion(nn.Module):
    def __init__(self, in_c, skip_c, out_c, act_type='hswish'):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c), 
            get_activation(act_type)
        )

    def forward(self, x, skip):
        return self.fuse(torch.cat([x, skip], dim=1))

class LightDecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, act_type='hswish'):
        super().__init__()
        self.up = NearestUpsample(in_c)
        self.fusion = SimpleConcatFusion(in_c, skip_c, out_c, act_type)
        
        gc = max(out_c // 4, 4)
        self.refine = nn.Sequential(
            nn.Conv2d(out_c, gc, kernel_size=1, bias=False), 
            nn.BatchNorm2d(gc), 
            get_activation(act_type),
            
            SquareDW(gc, kernel_size=5), 
            ECABlock(gc, act_type),
            
            nn.Conv2d(gc, out_c, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        self.act = get_activation(act_type)

    def forward(self, x, skip):
        up = self.up(x)
        fused = self.fusion(up, skip)
        return self.act(self.refine(fused) + fused) 

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
# 5. MẠNG CHÍNH PICO-UNET V2 PAPER (BẢN TỐC ĐỘ CAO - MAX 128 CHANNELS)
# ==============================================================================
class PicoUNet_v2_Paper(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"Input_size phải chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 24, kernel_size=3, padding=1)
        
        # Encoder: 24 -> 48 -> 96 -> 128 -> 128
        self.e1 = EncoderBlock_v2(24, 48, act_type='relu6')   # skip1: 48
        self.e2 = EncoderBlock_v2(48, 96, act_type='relu6')   # skip2: 96
        self.e3 = EncoderBlock_v2(96, 128, act_type='hswish') # skip3: 128
        self.e4 = EncoderBlock_v2(128, 128, act_type='hswish')# skip4: 128
        
        # Bottleneck max tại 128 kênh
        self.bottleneck = SerialMultiScaleBottleneck(128, act_type='hswish')
        
        # Decoder (Khai báo chính xác In, Skip, Out)
        self.d4 = LightDecoderBlock(in_c=128, skip_c=128, out_c=128, act_type='hswish') 
        self.d3 = LightDecoderBlock(in_c=128, skip_c=128, out_c=96,  act_type='hswish')  
        self.d2 = LightDecoderBlock(in_c=96,  skip_c=96,  out_c=48,  act_type='hswish')   
        self.d1 = LightDecoderBlock(in_c=48,  skip_c=48,  out_c=24,  act_type='hswish')   
        
        # Đầu ra 1 class (flood mask)
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