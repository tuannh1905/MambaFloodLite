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
# PICO-UNET V13: THE GUIDED REFINEMENT EDITION
# - ✓ D1/D2 CHỈ LÀ THỢ XÂY: Dùng Guided Detail Refinement siêu mỏng (mid_c=12).
# - ✓ SKIP PHÂN CẤP: s1(8), s2(16) [Rất nén] | s3(48), s4(64) [Mang Semantic].
# - ✓ E1/E2 DOWN-SAMPLE SỚM: Bóp ảnh nhỏ nhanh chóng, cắt MACs triệt để.
# ==============================================================================

# ==============================================================================
# 1. KHỐI TÍCH CHẬP VÀ UPSAMPLE
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

    def forward(self, x, scale_factor=2, size=None):
        if size is not None:
            return F.interpolate(x, size=size, mode='nearest')
        return F.interpolate(x, scale_factor=scale_factor, mode='nearest')

# ==============================================================================
# 2. ENCODER BLOCKS (Detail vs Semantic)
# ==============================================================================
class CompressedEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, skip_c, is_deep=False, act_type='relu6'):
        super().__init__()
        # Kỹ thuật Factorized Conv
        self.dw1 = SquareDW(in_c, kernel_size=3)
        self.dw2 = SquareDW(in_c, kernel_size=3)
        self.is_deep = is_deep
        
        if is_deep:
            self.dw3 = SquareDW(in_c, kernel_size=3)
            
        self.pw_fuse = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(in_c)
        self.act_fuse = get_activation(act_type)

        self.down_pool = nn.MaxPool2d((2, 2))
        
        self.same_channels = (in_c == out_c)
        if not self.same_channels:
            self.pw_down = nn.Sequential(
                nn.Conv2d(in_c, out_c - in_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c - in_c)
            )
            self.down_pw = nn.MaxPool2d((2, 2))
            
        self.act_out = get_activation(act_type)
        
        # ✓ Lớp Nén Skip-Connection (Bóp gắt ở tầng nông, lỏng ở tầng sâu)
        self.compress_skip = nn.Sequential(
            nn.Conv2d(out_c, skip_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_c),
            get_activation(act_type)
        )

    def forward(self, x):
        d1 = self.dw1(x)
        d2 = self.dw2(d1)
        if self.is_deep:
            d3 = self.dw3(d2)
            feat = self.act_fuse(self.bn_fuse(self.pw_fuse(d1 + d2 + d3)) + x)
        else:
            feat = self.act_fuse(self.bn_fuse(self.pw_fuse(d1 + d2)) + x)
        
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
# 3. DECODER BLOCKS (Semantic vs Detail Refinement)
# ==============================================================================

class SemanticDecoderBlock(nn.Module):
    """ ✓ Dùng cho d4, d3: Gánh vác Semantic Fusion (Não bộ) """
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
            
            SquareDW(gc, kernel_size=5), # Gọt biên
            nn.Conv2d(gc, out_c, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        self.act = get_activation(act_type)

    def forward(self, x, skip):
        fused = self.up_proj(self.up(x)) + self.skip_proj(skip)
        return self.act(self.refine(fused) + fused)


class GuidedDetailRefinementBlock(nn.Module):
    """ ✓ MỚI! Dùng cho d2, d1: Tầng nông (Chỉ gọt biên và texture) siêu rẻ """
    def __init__(self, in_c, skip_c, out_c, mid_c=12, act_type='relu6'):
        super().__init__()
        self.up = NearestUpsample()
        
        # Semantic đi lên bị ép vào không gian guide cực hẹp (mid_c)
        self.low_sem = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c)
        )
        # Detail đi ngang bị ép vào mid_c
        self.skip_det = nn.Sequential(
            nn.Conv2d(skip_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c)
        )
        
        # ✓ GATE: Dùng Semantic để quyết định xem có nhận Detail từ Skip hay không
        self.gate = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=1, bias=False),
            CustomHardsigmoid()
        )
        
        # Gọt không gian 3x3
        self.dw = SquareDW(mid_c, kernel_size=3)
        self.act_mid = get_activation(act_type)
        
        # Phóng lên out_c để truyền lên
        self.to_out = nn.Sequential(
            nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        
        # Shortcut residual
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act = get_activation(act_type)

    def forward(self, x, skip):
        up_x = self.up(x)
        
        sem = self.low_sem(up_x)
        det = self.skip_det(skip)
        
        # Gating: Detail chỉ được đi qua nếu Semantic cho phép
        fused = sem + det * self.gate(sem)
        
        fused = self.act_mid(self.dw(fused))
        out = self.to_out(fused)
        
        return self.act(out + self.shortcut(up_x))

# ==============================================================================
# 4. MẠNG CHÍNH PICO-UNET V13 (THE GUIDED REFINEMENT)
# ==============================================================================
class PicoUNet_v13_Edge(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"Input_size phải chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # ✓ TRUNK: 16 -> 32 -> 64 -> 128 (Mỏng ở nông, Dày ở sâu)
        # ✓ COMPRESSED SKIPS: s1(8) [Rất nén], s2(16), s3(48), s4(64) [Ít nén]
        self.e1 = CompressedEncoderBlock(16, 32, skip_c=8,  is_deep=False, act_type='relu6')   
        self.e2 = CompressedEncoderBlock(32, 64, skip_c=16, is_deep=False, act_type='relu6')   
        self.e3 = CompressedEncoderBlock(64, 128, skip_c=48, is_deep=True, act_type='hswish')  
        self.e4 = CompressedEncoderBlock(128, 128, skip_c=64, is_deep=True, act_type='hswish') 
        
        # Bottleneck (Giữ nguyên cấu trúc mỏng nhẹ 128->128 của V4)
        self.bottleneck = nn.Sequential(
            SquareDW(128, kernel_size=3),
            SquareDW(128, kernel_size=3),
            SquareDW(128, kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            get_activation('hswish')
        )
        
        # ✓ D4, D3 (Semantic Decoder): Trộn ngữ nghĩa
        self.d4 = SemanticDecoderBlock(in_c=128, skip_c=64, out_c=96, act_type='hswish') 
        self.d3 = SemanticDecoderBlock(in_c=96,  skip_c=48, out_c=64, act_type='hswish')  
        
        # ✓ D2, D1 (Guided Detail Refinement): Thợ xây siêu rẻ (mid_c=16 và mid_c=12)
        self.d2 = GuidedDetailRefinementBlock(in_c=64, skip_c=16, out_c=32, mid_c=16, act_type='relu6')   
        self.d1 = GuidedDetailRefinementBlock(in_c=32, skip_c=8,  out_c=16, mid_c=12, act_type='relu6')   
        
        # Head Smoothing
        self.head_smooth = nn.Sequential(
            SquareDW(16, kernel_size=3),
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
    return PicoUNet_v13_Edge(num_classes=num_classes, input_size=input_size)