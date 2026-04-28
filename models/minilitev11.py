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
# PICO-UNET V12: THE SEMANTIC BRIDGE EDITION
# - ✓ DUAL BOTTLENECK: Dùng 2 khối ContextBottleneck liên tiếp để tăng suy luận.
# - ✓ WIDER CONTEXT: branch_c tăng lên 48 (kênh ngữ cảnh phong phú hơn).
# - ✓ DEEP SKIPS: s3, s4 tăng lên 96 kênh (Semantic Bandwidth lớn hơn).
# - ✓ ASYMMETRIC DECODER: d4, d3 phình to (112, 56) để xử lý logic, d2, d1 giữ mỏng (24) để cứu FPS.
# ==============================================================================

# ==============================================================================
# 1. CONTEXT ENHANCEMENT GATE (CE-GATE)
# ==============================================================================
class CEGate(nn.Module):
    def __init__(self, channels, act_type='hswish'):
        super().__init__()
        mid_channels = max(8, channels // 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            get_activation(act_type),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            CustomHardsigmoid()
        )

    def forward(self, x):
        gate = self.mlp(self.pool(x))
        return x * gate

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
    def __init__(self, channels=None):
        super().__init__()
        pass

    def forward(self, x, scale_factor=2, size=None):
        if size is not None:
            return F.interpolate(x, size=size, mode='nearest')
        return F.interpolate(x, scale_factor=scale_factor, mode='nearest')

# ==============================================================================
# 3. COMPRESSED ENCODER BLOCKS
# ==============================================================================
class DualScale_PFCU_DG(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw1 = SquareDW(dim, kernel_size=3)
        self.dw2 = SquareDW(dim, kernel_size=3) 
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        b3 = self.dw1(x)        
        b5 = self.dw2(b3)        
        fused = self.bn_fuse(self.pw_fuse(b3 + b5))
        return self.act(fused + x)

class MultiScale_PFCU_DG(nn.Module):
    def __init__(self, dim, act_type='hswish'):
        super().__init__()
        self.dw1 = SquareDW(dim, kernel_size=3)
        self.dw2 = SquareDW(dim, kernel_size=3) 
        self.dw3 = SquareDW(dim, kernel_size=3) 
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        b3 = self.dw1(x)        
        b5 = self.dw2(b3)        
        b7 = self.dw3(b5)        
        fused = self.bn_fuse(self.pw_fuse(b3 + b5 + b7))
        return self.act(fused + x)

class CompressedEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, skip_c, is_deep=False, act_type='relu6'):
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
        
        self.compress_skip = nn.Sequential(
            nn.Conv2d(out_c, skip_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_c),
            get_activation(act_type)
        )

    def forward(self, x):
        feat = self.pfcu_dg(x)
        
        if self.same_channels:
            skip_raw = feat
            out = self.act(self.down_pool(feat))
        else:
            feat_pw = self.pw(feat)
            skip_raw = torch.cat([feat, feat_pw], dim=1)
            
            pool_feat = self.down_pool(feat)
            pool_pw   = self.down_pw(feat_pw)
            out = self.act(torch.cat([pool_feat, pool_pw], dim=1))
            
        skip_compressed = self.compress_skip(skip_raw)
        return out, skip_compressed

# ==============================================================================
# 4. BOTTLENECK ĐA THANG (PPM-LITE) VÀ OUT-SPACE DECODER
# ==============================================================================
class PoolBranch(nn.Module):
    def __init__(self, in_c, branch_c, pool_size, act_type):
        super().__init__()
        if pool_size == 'global':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
            
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, branch_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_c),
            get_activation(act_type)
        )
        
    def forward(self, x, target_size):
        p = self.conv(self.pool(x))
        return F.interpolate(p, size=target_size, mode='nearest')

class ContextBottleneck_MCU(nn.Module):
    # ✓ MỚI: Thêm tham số branch_c để dễ dàng điều chỉnh độ rộng ngữ cảnh
    def __init__(self, dim, branch_c=32, act_type='hswish'):
        super().__init__()
        
        self.local = nn.Sequential(
            SquareDW(dim, kernel_size=3),
            nn.Conv2d(dim, branch_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_c),
            get_activation(act_type)
        )
        
        self.pool2 = PoolBranch(dim, branch_c, pool_size=2, act_type=act_type)
        self.pool4 = PoolBranch(dim, branch_c, pool_size=4, act_type=act_type)
        self.pool_g = PoolBranch(dim, branch_c, pool_size='global', act_type=act_type)
        
        # Fuse 4 luồng branch_c thành 1 luồng dim
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_c, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            get_activation(act_type)
        )
        self.ce_gate = CEGate(dim, act_type)

    def forward(self, x):
        target_size = x.shape[2:]
        
        l  = self.local(x)
        p2 = self.pool2(x, target_size)
        p4 = self.pool4(x, target_size)
        pg = self.pool_g(x, target_size)
        
        fused = l + p2 + p4 + pg
        out = self.fuse(fused)
        out = self.ce_gate(out)
        
        return x + out

class OutSpaceDecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, act_type='hswish'):
        super().__init__()
        self.up_proj = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        
        gc = max(out_c // 2, 8) 
        
        self.refine = nn.Sequential(
            nn.Conv2d(out_c, gc, kernel_size=1, bias=False), 
            nn.BatchNorm2d(gc), 
            get_activation(act_type),
            
            SquareDW(gc, kernel_size=5), 
            CEGate(gc, act_type), 
            
            nn.Conv2d(gc, out_c, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        self.act = get_activation(act_type)

    def forward(self, x, skip):
        up_x = F.interpolate(x, scale_factor=2, mode='nearest')
        fused = self.up_proj(up_x) + self.skip_proj(skip)
        return self.act(self.refine(fused) + fused)

# ==============================================================================
# 5. MẠNG CHÍNH PICO-UNET V12 (THE SEMANTIC BRIDGE)
# ==============================================================================
class PicoUNet_v12_Edge(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"Input_size phải chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 24, kernel_size=3, padding=1)
        
        # ✓ TRUNK & COMPRESSED SKIPS: 
        # s1=24, s2=48, s3=96 (Tăng mạnh), s4=96 (Tăng mạnh)
        self.e1 = CompressedEncoderBlock(24, 48, skip_c=24, is_deep=False, act_type='relu6')   
        self.e2 = CompressedEncoderBlock(48, 96, skip_c=48, is_deep=False, act_type='relu6')   
        self.e3 = CompressedEncoderBlock(96, 128, skip_c=96, is_deep=True, act_type='hswish')  
        self.e4 = CompressedEncoderBlock(128, 128, skip_c=96, is_deep=True, act_type='hswish') 
        
        # ✓ DUAL BOTTLENECK: 2 lớp liên tiếp, mở rộng branch_c lên 48
        self.bottleneck = nn.Sequential(
            ContextBottleneck_MCU(128, branch_c=48, act_type='hswish'),
            ContextBottleneck_MCU(128, branch_c=48, act_type='hswish')
        )
        
        # ✓ ASYMMETRIC DECODER: Bơm máu ở d4, d3. Giữ mỏng d2, d1.
        self.d4 = OutSpaceDecoderBlock(in_c=128, skip_c=96, out_c=112, act_type='hswish') 
        self.d3 = OutSpaceDecoderBlock(in_c=112, skip_c=96, out_c=56,  act_type='hswish')  
        self.d2 = OutSpaceDecoderBlock(in_c=56,  skip_c=48, out_c=24,  act_type='relu6')   
        self.d1 = OutSpaceDecoderBlock(in_c=24,  skip_c=24, out_c=24,  act_type='relu6')   
        
        # ✓ SMOOTHING HEAD
        self.head_smooth = nn.Sequential(
            SquareDW(24, kernel_size=3),
            get_activation('relu6')
        )
        
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
        
        x = self.head_smooth(x)
        return self.conv_out(x)

def build_model(num_classes=1, input_size=128):
    return PicoUNet_v12_Edge(num_classes=num_classes, input_size=input_size)