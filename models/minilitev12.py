import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 0. CUSTOM ACTIVATIONS CHO ONNX OPSET 11
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
# PICO-UNET V22: THE FOCUS V4 EDITION
# - ✓ FOCUS STEM (SPACE-TO-DEPTH): Gom 4 pixel thành kênh, giảm 75% GFLOPs không mất dữ liệu.
# - ✓ LINEAR V4 CORE: Dùng lại Dual/Multi-Scale Linear Encoder của V4.
# - ✓ SPPM BOTTLE-NECK: Giữ nguyên khối Kim tự tháp không gian.
# - ✓ HALF-RES HEAD: Vẽ mask ở 64x64 và nội suy lên 128x128.
# ==============================================================================

# ==============================================================================
# 1. FOCUS MODULE (VŨ KHÍ MỚI) & ATTENTION
# ==============================================================================
class Focus_MCU(nn.Module):
    """ Space-to-Depth: Gom 4 pixel kề nhau thành kênh """
    def __init__(self, in_channels=3, out_channels=32, act_type='relu6'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(act_type)
        )

    def forward(self, x):
        # x shape: [B, 3, 128, 128]
        patch_top_left  = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left  = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        
        # Ghép lại: [B, 12, 64, 64]
        x_unshuffled = torch.cat(
            [patch_top_left, patch_bot_left, patch_top_right, patch_bot_right], 
            dim=1
        )
        return self.conv(x_unshuffled)

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
        y = self.hardsigmoid(self.conv(y))
        return x * y

class SpatialAttention_MCU(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
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
    # TRỞ VỀ CHÂN LÝ: Không dùng Activation để tạo Linear Manifold
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
# 3. ENCODER BLOCKS (V4 CHUẨN MỰC + STRIDE=2)
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
            
        self.same_channels = (in_c == out_c)
        if not self.same_channels:
            self.pw = nn.Sequential(
                nn.Conv2d(in_c, out_c - in_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c - in_c)
            )
            
        # Learnable Downsample Stride=2
        self.down_dw = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act_out = get_activation(act_type)

    def forward(self, x):
        feat = self.pfcu_dg(x)
        
        if self.same_channels:
            skip = feat
        else:
            feat_pw = self.pw(feat)
            skip = torch.cat([feat, feat_pw], dim=1)
            
        out = self.act_out(self.down_dw(skip))
        return out, skip

# ==============================================================================
# 4. DECODER V4 & BOTTLE-NECK SPPM
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

class PoolBranch(nn.Module):
    def __init__(self, in_c, out_c, pool_size, act_type):
        super().__init__()
        if pool_size == 1:
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
            
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            get_activation(act_type)
        )

    def forward(self, x, target_size):
        pooled = self.conv(self.pool(x))
        return F.interpolate(pooled, size=target_size, mode='nearest')

class SPPMBottleneck(nn.Module):
    def __init__(self, dim, act_type='hswish'):
        super().__init__()
        branch_c = max(32, dim // 4)
        self.pool1 = PoolBranch(dim, branch_c, pool_size=1, act_type=act_type)
        self.pool2 = PoolBranch(dim, branch_c, pool_size=2, act_type=act_type)
        self.pool4 = PoolBranch(dim, branch_c, pool_size=4, act_type=act_type)
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_c, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            get_activation(act_type)
        )
        self.channel_attn = ECABlock(dim, act_type)
        self.spatial_attn = SpatialAttention_MCU(kernel_size=3)

    def forward(self, x):
        target_size = x.shape[2:]
        p1 = self.pool1(x, target_size)
        p2 = self.pool2(x, target_size)
        p4 = self.pool4(x, target_size)
        
        ppm_out = self.fuse(p1 + p2 + p4)
        out = x + ppm_out
        
        out = self.channel_attn(out)
        out = self.spatial_attn(out)
        return out

# ==============================================================================
# 5. MẠNG CHÍNH PICO-UNET V22 (FOCUS V4)
# ==============================================================================
class PicoUNet_v22_Focus(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        self.input_size = input_size
        if input_size % 16 != 0:
            raise ValueError(f"Input_size phải chia hết cho 16.")

        # ✓ VŨ KHÍ TỐI THƯỢNG: Cắt 75% MACs ngay từ vòng gửi xe
        self.conv_in = Focus_MCU(in_channels=3, out_channels=32, act_type='relu6')
        
        # Mạch chính V4 (Chạy bắt đầu từ 64x64)
        self.e1 = EncoderBlock(32, 64,  is_deep=False, act_type='relu6')   
        self.e2 = EncoderBlock(64, 128, is_deep=False, act_type='relu6')   
        self.e3 = EncoderBlock(128, 192, is_deep=True, act_type='hswish') 
        self.e4 = EncoderBlock(192, 192, is_deep=True, act_type='hswish') 
        
        self.bottleneck = SPPMBottleneck(192, act_type='hswish')
        
        self.d4 = AdditiveDecoderBlock(in_c=192, skip_c=192, out_c=128, act_type='hswish') 
        self.d3 = AdditiveDecoderBlock(in_c=128, skip_c=192, out_c=64,  act_type='hswish')  
        self.d2 = AdditiveDecoderBlock(in_c=64,  skip_c=128, out_c=32,  act_type='hswish')   
        self.d1 = AdditiveDecoderBlock(in_c=32,  skip_c=64,  out_c=16,  act_type='hswish')   
        
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: [B, 3, 128, 128]
        x = self.conv_in(x) # x shape: [B, 32, 64, 64]
        
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        
        x = self.bottleneck(x)
        
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        
        mask_64 = self.conv_out(x)
        
        # ✓ Khôi phục kích thước ảnh 128x128
        out = F.interpolate(mask_64, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        return out

def build_model(num_classes=1, input_size=128):
    return PicoUNet_v22_Focus(num_classes=num_classes, input_size=input_size)