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
# 3. KHỐI ENCODER (SINGLE-SCALE VÀ MULTI-SCALE)
# ==============================================================================
class SingleScale_PFCU_DG(nn.Module):
    # Dùng cho Tầng Nông: 1 nhánh 3x3 để tiết kiệm MACs
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw_3x3 = SquareDW(dim, kernel_size=3)
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        b3 = self.dw_3x3(x)        
        fused = self.bn_fuse(self.pw_fuse(b3))
        return self.act(fused + x)

class MultiScale_PFCU_DG(nn.Module):
    # Dùng cho Tầng Sâu: Xếp chồng 3x3 để ESP-DL chạy Fast-path Vector
    def __init__(self, dim, act_type='hswish'):
        super().__init__()
        self.dw_3x3_step1 = SquareDW(dim, kernel_size=3)
        self.dw_3x3_step2 = SquareDW(dim, kernel_size=3) 
        self.dw_3x3_step3 = SquareDW(dim, kernel_size=3) 
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        rf_3x3 = self.dw_3x3_step1(x)        
        rf_5x5 = self.dw_3x3_step2(rf_3x3)        
        rf_7x7 = self.dw_3x3_step3(rf_5x5)        
        
        fused = self.bn_fuse(self.pw_fuse(rf_3x3 + rf_5x5 + rf_7x7))
        return self.act(fused + x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, scale_type='single', act_type='relu6'):
        super().__init__()
        
        if scale_type == 'multi':
            self.pfcu_dg = MultiScale_PFCU_DG(in_c, act_type)
        else:
            self.pfcu_dg = SingleScale_PFCU_DG(in_c, act_type)
            
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
# 4. BOTTLENECK (LIGHTWEIGHT ORTHOGONAL) & DECODER (ADDITIVE)
# ==============================================================================
class LightweightOrthogonalBottleneck(nn.Module):
    def __init__(self, dim, act_type='hswish', reduction=4):
        super().__init__()
        
        self.local_dw = SquareDW(dim, kernel_size=3)
        
        # Squeeze: Bóp kênh (vd 192 -> 48) để giảm params điểm X-Y
        mid_dim = max(16, dim // reduction)
        self.squeeze = nn.Sequential(
            nn.Conv2d(dim, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            get_activation(act_type)
        )
        
        self.proj_h = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim)
        )
        self.proj_w = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim)
        )
        
        # Expand: Phóng to lại số kênh ban đầu
        self.expand = nn.Sequential(
            nn.Conv2d(mid_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        
        self.channel_attn = ECABlock(dim, act_type)
        self.act = get_activation(act_type)

    def forward(self, x):
        feat_local = self.local_dw(x)
        
        x_sq = self.squeeze(x)
        
        pool_h = torch.mean(x_sq, dim=3, keepdim=True)
        feat_h = self.proj_h(pool_h)
        
        pool_w = torch.mean(x_sq, dim=2, keepdim=True)
        feat_w = self.proj_w(pool_w)
        
        feat_global = feat_h + feat_w
        feat_global = self.act(self.expand(feat_global))
        
        fused = feat_local + feat_global
        fused = self.channel_attn(fused)
        
        return x + fused

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

# ==============================================================================
# 5. MẠNG CHÍNH PICO-UNET V4 (HARDWARE-AWARE SOTA)
# ==============================================================================
class PicoUNet_v4_Proposed(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"Input_size phải chia hết cho 16.")

        # Khởi tạo: In -> 16
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # Tầng Nông: Single-Scale (Giảm MACs)
        self.e1 = EncoderBlock(16, 32,  scale_type='single', act_type='relu6')   
        self.e2 = EncoderBlock(32, 64,  scale_type='single', act_type='relu6')   
        
        # Tầng Sâu: Multi-Scale (Tăng Receptive Field)
        self.e3 = EncoderBlock(64, 128, scale_type='multi',  act_type='hswish') 
        self.e4 = EncoderBlock(128, 192, scale_type='multi', act_type='hswish') 
        
        # Đáy: Lightweight Orthogonal Bottleneck (192)
        self.bottleneck = LightweightOrthogonalBottleneck(192, act_type='hswish', reduction=4)
        
        # Decoder: Additive Fusion (Cứu RAM)
        self.d4 = AdditiveDecoderBlock(in_c=192, skip_c=192, out_c=128, act_type='hswish') 
        self.d3 = AdditiveDecoderBlock(in_c=128, skip_c=128, out_c=64,  act_type='hswish')  
        self.d2 = AdditiveDecoderBlock(in_c=64,  skip_c=64,  out_c=32,  act_type='hswish')   
        self.d1 = AdditiveDecoderBlock(in_c=32,  skip_c=32,  out_c=16,  act_type='hswish')   
        
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
    return PicoUNet_v4_Proposed(num_classes=num_classes, input_size=input_size)