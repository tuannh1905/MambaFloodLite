import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# LƯU Ý TƯƠNG THÍCH ONNX / TORCH.FX / MCU:
# - BOTTLE-NECK ĐƯỜNG THẲNG: Dùng DW 7x7, xóa bỏ hoàn toàn SPP và nhánh.
# - MICRO-AG FUSION: Attention Gate siêu nhẹ cắt rác cho Encoder trước khi cộng.
# - ASYMMETRIC ENCODER (NEW): e1 chạy đường thẳng 7x7, e2-e4 phân nhánh.
# - ASYMMETRIC DECODER (NEW): d1, d2 lột sạch CBAM/Residual để ép RAM; d3, d4 giữ nguyên.
# - LARGE KERNEL STEM (NEW): Đầu vào dùng Conv 7x7 mở rộng tầm nhìn toàn cục.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES (DUAL ATTENTION TỐI ƯU CHO MCU)
# ==============================================================================
class MicroSEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid_channels = max(8, channels // 4)
        self.se = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Hardsigmoid() 
        )

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)
        return x * self.se(y)

class MicroSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.hardsigmoid(y)

class EdgeCBAM(nn.Module):
    def __init__(self, channels, spatial_kernel=7):
        super().__init__()
        self.channel_att = MicroSEBlock(channels)
        self.spatial_att = MicroSpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# ==============================================================================
# 2. NEAREST UPSAMPLE & MICRO ATTENTION GATE FUSION
# ==============================================================================
class NearestUpsample(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.up     = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn     = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.refine(self.up(x)))

class MicroAGFusion(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_c, 1, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x_up, x_skip):
        alpha = self.gate(x_up + x_skip) 
        x_skip_gated = x_skip * alpha
        return self.fuse_conv(x_up + x_skip_gated)

# ==============================================================================
# 3. PFCU BLOCKS: LINEAR (Ép RAM) & MULTI-SCALE (Tăng mIoU)
# ==============================================================================
class SquareDW(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)
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

class Linear_PFCU(nn.Module):
    """ Dành cho tầng e1 (độ phân giải khổng lồ): Không phân nhánh, chỉ dùng MicroSE """
    def __init__(self, dim):
        super().__init__()
        self.large_kernel = SquareDW(dim, kernel_size=7)
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim) 
        self.dg_shortcut = DetailGuidance(dim)
        self.attention = MicroSEBlock(dim) # Chỉ dùng SE để đỡ tốn RAM không gian
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        context = self.large_kernel(x)
        fused_context = self.bn_fuse(self.pw_fuse(context))
        guided_details = self.dg_shortcut(x)
        return self.attention(self.act(fused_context + guided_details))

class MultiScale_PFCU_DG(nn.Module):
    """ Dành cho tầng e2, e3, e4 (độ phân giải nhỏ): Phân nhánh, dùng EdgeCBAM """
    def __init__(self, dim):
        super().__init__()
        self.branch_3 = SquareDW(dim, kernel_size=3)
        self.branch_5 = SquareDW(dim, kernel_size=5)
        self.branch_7 = SquareDW(dim, kernel_size=7)
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim) 
        self.dg_shortcut = DetailGuidance(dim)
        self.attention = EdgeCBAM(dim) 
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        b3, b5, b7 = self.branch_3(x), self.branch_5(x), self.branch_7(x)
        fused_context = self.bn_fuse(self.pw_fuse(b3 + b5 + b7))
        guided_details = self.dg_shortcut(x)
        return self.attention(self.act(fused_context + guided_details))

# ==============================================================================
# 4. ENCODER ASYMMETRIC, DECODER ASYMMETRIC & LINEAR BOTTLE-NECK
# ==============================================================================
class EncoderBlock_Linear(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.pfcu_dg = Linear_PFCU(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw = nn.Conv2d(in_c, conv_out, kernel_size=1, bias=False)
            self.bn_pw = nn.BatchNorm2d(conv_out) 
            self.down_pw = nn.MaxPool2d((2, 2))
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        feat = self.pfcu_dg(x) 
        if self.same_channels:
            return self.act(self.down_pool(feat)), feat
        else:
            feat_pw = self.bn_pw(self.pw(feat)) 
            skip = torch.cat([feat, feat_pw], dim=1) 
            pool_feat = self.down_pool(feat)
            pool_pw   = self.down_pw(feat_pw)
            return self.act(torch.cat([pool_feat, pool_pw], dim=1)), skip

class EncoderBlock_MultiScale(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.pfcu_dg = MultiScale_PFCU_DG(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw = nn.Conv2d(in_c, conv_out, kernel_size=1, bias=False)
            self.bn_pw = nn.BatchNorm2d(conv_out) 
            self.down_pw = nn.MaxPool2d((2, 2))
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        feat = self.pfcu_dg(x) 
        if self.same_channels:
            return self.act(self.down_pool(feat)), feat
        else:
            feat_pw = self.bn_pw(self.pw(feat)) 
            skip = torch.cat([feat, feat_pw], dim=1) 
            pool_feat = self.down_pool(feat)
            pool_pw   = self.down_pw(feat_pw)
            return self.act(torch.cat([pool_feat, pool_pw], dim=1)), skip

class LightDecoderBlock_AG_Deep(nn.Module):
    """ Dùng cho tầng d4, d3: Giữ nguyên CBAM và Residual vì RAM còn rộng rãi """
    def __init__(self, in_c, out_c):
        super().__init__()
        gc = max(out_c // 4, 4)

        self.up = NearestUpsample(in_c, scale_factor=2)
        self.fusion = MicroAGFusion(in_c=in_c, out_c=out_c) 

        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        self.bn_down = nn.BatchNorm2d(gc)
        
        self.refine_spatial = SquareDW(gc, kernel_size=5)
        self.attention = EdgeCBAM(gc) 
        
        self.pw_up  = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        self.bn_up  = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        fused = self.fusion(x, skip)
        feat = self.act(self.bn_down(self.pw_down(fused)))
        feat = self.refine_spatial(feat)
        feat = self.attention(feat)
        out = self.bn_up(self.pw_up(feat))
        return self.act(out + fused)

class LightDecoderBlock_AG_Shallow(nn.Module):
    """ Dùng cho tầng d2, d1: Lột sạch CBAM và Residual để cứu Peak RAM """
    def __init__(self, in_c, out_c):
        super().__init__()
        gc = max(out_c // 4, 4)

        self.up = NearestUpsample(in_c, scale_factor=2)
        self.fusion = MicroAGFusion(in_c=in_c, out_c=out_c) 

        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        self.bn_down = nn.BatchNorm2d(gc)
        
        self.refine_spatial = SquareDW(gc, kernel_size=5)
        # Bỏ CBAM hoàn toàn
        
        self.pw_up  = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        self.bn_up  = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        fused = self.fusion(x, skip)
        feat = self.act(self.bn_down(self.pw_down(fused)))
        feat = self.refine_spatial(feat)
        out = self.bn_up(self.pw_up(feat))
        return self.act(out) # Bỏ cộng dư (fused)

class LinearBottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw1 = SquareDW(dim, kernel_size=7)
        self.attention = EdgeCBAM(dim)
        self.dw2 = SquareDW(dim, kernel_size=7)
        
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(dim)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.dw1(x)
        out = self.attention(out)
        out = self.dw2(out)
        out = self.act(self.bn_pw(self.pw(out)))
        return out + x 

# ==============================================================================
# 5. MẠNG CHÍNH (PicoUNet EDGE ASYMMETRIC)
# ==============================================================================
class PicoUNet_Edge(nn.Module):
    def __init__(self, num_classes=1, input_size=256):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"PicoUNet yêu cầu input_size chia hết cho 16.")

        # ✓ Stem 7x7 mở góc nhìn rộng
        self.conv_in = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        # ✓ Encoder Bất đối xứng
        self.e1 = EncoderBlock_Linear(16,  32)      # Linear (Chống Peak RAM)
        self.e2 = EncoderBlock_MultiScale(32,  64)  # Multi-Scale (Tăng mIoU)
        self.e3 = EncoderBlock_MultiScale(64,  128) # Multi-Scale
        self.e4 = EncoderBlock_MultiScale(128, 256) # Multi-Scale

        self.b4 = LinearBottleneck(256)

        # ✓ Decoder Bất đối xứng
        self.d4 = LightDecoderBlock_AG_Deep(256, 128)    # Giữ CBAM, Residual
        self.d3 = LightDecoderBlock_AG_Deep(128, 64)     # Giữ CBAM, Residual
        self.d2 = LightDecoderBlock_AG_Shallow(64,  32)  # Bỏ CBAM, Residual
        self.d1 = LightDecoderBlock_AG_Shallow(32,  16)  # Bỏ CBAM, Residual

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

def build_model(num_classes=1, input_size=256):
    return PicoUNet_Edge(num_classes=num_classes, input_size=input_size)