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
# PICO-UNET V15: THE ULTIMATE LRASPP EDITION
# - ✓ LRASPP-LIKE BOTTLENECK: Kết hợp Local (3x3 -> 5x5) và Global (GAP).
# - ✓ THICKER SHALLOW SKIPS: s1(16), s2(24) giúp giữ chi tiết sắc nét hơn.
# - ✓ THICKER GUIDED REFINEMENT: d1, d2 được bơm thêm mid_c (16, 20).
# ==============================================================================

# ==============================================================================
# 1. KHỐI TÍCH CHẬP VÀ UPSAMPLE
# ==============================================================================
class SquareDW(nn.Module):
    def __init__(self, dim, kernel_size=3, act_type='relu6'):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))

class NearestUpsample(nn.Module):
    def __init__(self, channels=None):
        super().__init__()

    def forward(self, x, scale_factor=2, size=None):
        if size is not None:
            return F.interpolate(x, size=size, mode='nearest')
        return F.interpolate(x, scale_factor=scale_factor, mode='nearest')

# ==============================================================================
# 2. ENCODER BLOCKS (Detail vs Semantic)
# ==============================================================================
class DualScale_PFCU_DG(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw1 = SquareDW(dim, kernel_size=3, act_type=act_type)
        self.dw2 = SquareDW(dim, kernel_size=3, act_type=act_type) 
        
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
        self.dw1 = SquareDW(dim, kernel_size=3, act_type=act_type)
        self.dw2 = SquareDW(dim, kernel_size=3, act_type=act_type) 
        self.dw3 = SquareDW(dim, kernel_size=3, act_type=act_type) 
        
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
            self.pw_down = nn.Sequential(
                nn.Conv2d(in_c, out_c - in_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c - in_c)
            )
            self.down_pw = nn.MaxPool2d((2, 2))
            
        self.act_out = get_activation(act_type)
        
        self.compress_skip = nn.Sequential(
            nn.Conv2d(out_c, skip_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_c),
            get_activation(act_type)
        )

    def forward(self, x):
        feat = self.pfcu_dg(x)
        
        if self.same_channels:
            skip_raw = feat
            out = self.act_out(self.down_pool(feat))
        else:
            feat_pw = self.pw_down(feat)
            skip_raw = torch.cat([feat, feat_pw], dim=1)
            
            pool_feat = self.down_pool(feat)
            pool_pw   = self.down_pw(feat_pw)
            out = self.act_out(torch.cat([pool_feat, pool_pw], dim=1))
            
        skip_compressed = self.compress_skip(skip_raw)
        return out, skip_compressed

# ==============================================================================
# 3. NEW BOTTLENECK (BÓP GAP GỌN NHẸ)
# ==============================================================================
class BottleneckWithGAP(nn.Module):
    def __init__(self, dim=128, act_type='hswish'):
        super().__init__()
        # V14 SquareDW đã tích hợp sẵn Act+BN, nối tiếp tạo RF=7
        self.dw1 = SquareDW(dim, kernel_size=3, act_type=act_type)  
        self.dw2 = SquareDW(dim, kernel_size=5, act_type=act_type)  
        
        # GAP thay cho lớp 5x5 thứ 3. RF: vô cực
        self.gap_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            get_activation(act_type)
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            get_activation(act_type)
        )

    def forward(self, x):
        local = self.dw2(self.dw1(x))
        # GAP: context toàn cục, PyTorch sẽ tự động broadcast (1x1) lên (HxW) khi cộng
        global_ctx = self.gap_proj(x)
        return self.fuse(local + global_ctx)

# ==============================================================================
# 4. DECODER BLOCKS
# ==============================================================================
class SemanticDecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, act_type='hswish'):
        super().__init__()
        self.up = NearestUpsample()
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
            
            SquareDW(gc, kernel_size=5, act_type=act_type), 
            nn.Conv2d(gc, out_c, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        self.act = get_activation(act_type)

    def forward(self, x, skip):
        fused = self.up_proj(self.up(x)) + self.skip_proj(skip)
        return self.act(self.refine(fused) + fused)

class GuidedDetailRefinementBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, mid_c=12, act_type='relu6'):
        super().__init__()
        self.up = NearestUpsample()
        
        self.low_sem = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c)
        )
        self.skip_det = nn.Sequential(
            nn.Conv2d(skip_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c)
        )
        
        self.gate = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=1, bias=False),
            CustomHardsigmoid()
        )
        
        self.dw = SquareDW(mid_c, kernel_size=3, act_type=act_type)
        
        self.to_out = nn.Sequential(
            nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act = get_activation(act_type)

    def forward(self, x, skip):
        up_x = self.up(x)
        sem = self.low_sem(up_x)
        det = self.skip_det(skip)
        
        fused = sem + det * self.gate(sem)
        fused = self.dw(fused)
        out = self.to_out(fused)
        
        return self.act(out + self.shortcut(up_x))

# ==============================================================================
# 5. MẠNG CHÍNH PICO-UNET V15 (FINAL DEPLOY)
# ==============================================================================
class PicoUNet_v15_Edge(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"Input_size phải chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # ✓ FIX 1: Tăng Skip_c ở e1, e2
        self.e1 = CompressedEncoderBlock(16, 32, skip_c=16,  is_deep=False, act_type='relu6')   
        self.e2 = CompressedEncoderBlock(32, 64, skip_c=24, is_deep=False, act_type='relu6')   
        self.e3 = CompressedEncoderBlock(64, 128, skip_c=48, is_deep=True, act_type='hswish')  
        self.e4 = CompressedEncoderBlock(128, 128, skip_c=64, is_deep=True, act_type='hswish') 
        
        # ✓ FIX 2: Tích hợp LRASPP Bottleneck
        self.bottleneck = BottleneckWithGAP(128, act_type='hswish')
        
        # Decoder
        self.d4 = SemanticDecoderBlock(in_c=128, skip_c=64, out_c=96, act_type='hswish') 
        self.d3 = SemanticDecoderBlock(in_c=96,  skip_c=48, out_c=64, act_type='hswish')  
        
        # ✓ FIX 1 (tt): Cập nhật Decoder nhận chuẩn skip_c và mid_c mới
        self.d2 = GuidedDetailRefinementBlock(in_c=64, skip_c=24, out_c=32, mid_c=20, act_type='relu6')   
        self.d1 = GuidedDetailRefinementBlock(in_c=32, skip_c=16,  out_c=16, mid_c=16, act_type='relu6')   
        
        self.head_smooth = nn.Sequential(
            SquareDW(16, kernel_size=3, act_type='relu6'),
            get_activation('relu6')
        )
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
        
        x = self.head_smooth(x)
        return self.conv_out(x)

def build_model(num_classes=1, input_size=128):
    return PicoUNet_v15_Edge(num_classes=num_classes, input_size=input_size)