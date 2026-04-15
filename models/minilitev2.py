import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# LƯU Ý TƯƠNG THÍCH ONNX / TORCH.FX / MCU:
# - BOTTLE-NECK ĐƯỜNG THẲNG: Dùng DW 7x7, xóa bỏ hoàn toàn SPP và nhánh.
# - MICRO-AG FUSION (NEW): Attention Gate siêu nhẹ (chỉ tốn ~480 params toàn mạng)
#   để dùng Decoder (x_up) làm màng lọc cắt rác cho Encoder (x_skip) trước khi cộng.
# - EDGE-CBAM: Bổ sung Spatial Attention siêu nhẹ (~98 tham số) kết hợp Channel Attention.
# - NEAREST UPSAMPLE: Đảm bảo MCU nội suy ảnh tốc độ cao nhất.
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
# 2. NEAREST UPSAMPLE & MICRO ATTENTION GATE (Hướng 2)
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
    """
    Attention Gate Siêu Nhẹ (Micro Attention Gate).
    Dùng x_up để soi đường cho x_skip.
    Overhead tham số: Bằng đúng in_c (vài trăm tham số).
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # Tạo Bản đồ chú ý không gian (Spatial Gate) từ sự kết hợp của x_up và x_skip
        # Ép in_c kênh xuống 1 kênh duy nhất
        self.gate = nn.Sequential(
            nn.Conv2d(in_c, 1, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )
        
        # Khối trộn kênh cuối cùng
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x_up, x_skip):
        # 1. Tìm sự đồng thuận không gian giữa 2 luồng (Tìm ra chỗ nào có nước)
        alpha = self.gate(x_up + x_skip) # Shape: (B, 1, H, W)
        
        # 2. Dùng mặt nạ alpha cắt tỉa sạch sẽ rác của x_skip
        x_skip_gated = x_skip * alpha
        
        # 3. Cộng x_up với x_skip_gated (thay vì x_skip đầy rác) và trộn kênh
        return self.fuse_conv(x_up + x_skip_gated)

# ==============================================================================
# 3. SQUARE-PFCU-DG BLOCK (MCU NATIVE)
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

class MultiScale_PFCU_DG(nn.Module):
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
# 4. ENCODER, LIGHT-DECODER & LINEAR BOTTLE-NECK
# ==============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.pfcu_dg   = MultiScale_PFCU_DG(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw      = nn.Conv2d(in_c, conv_out, kernel_size=1, bias=False)
            self.bn_pw   = nn.BatchNorm2d(conv_out) 
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
            x = self.act(torch.cat([pool_feat, pool_pw], dim=1))
            return x, skip

class LightDecoderBlock_AG(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        gc = max(out_c // 4, 4)

        self.up = NearestUpsample(in_c, scale_factor=2)
        
        # ✓ ĐÃ CẬP NHẬT: Dùng Khối MicroAG Fusion siêu nhẹ
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
        
        # Gửi x_up và skip vào Cổng chú ý để cắt rác rồi trộn
        fused = self.fusion(x, skip)
        
        feat = self.act(self.bn_down(self.pw_down(fused)))
        feat = self.refine_spatial(feat)
        feat = self.attention(feat)
        
        out = self.bn_up(self.pw_up(feat))
        return self.act(out + fused)

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
# 5. MẠNG CHÍNH (PicoUNet EDGE)
# ==============================================================================
class PicoUNet_Edge(nn.Module):
    def __init__(self, num_classes=1, input_size=256):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"PicoUNet yêu cầu input_size chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32)
        self.e2 = EncoderBlock(32,  64)
        self.e3 = EncoderBlock(64,  128)
        self.e4 = EncoderBlock(128, 256)

        self.b4 = LinearBottleneck(256)

        # ✓ ĐÃ CẬP NHẬT: Gọi LightDecoderBlock_AG
        self.d4 = LightDecoderBlock_AG(256, 128)
        self.d3 = LightDecoderBlock_AG(128, 64)
        self.d2 = LightDecoderBlock_AG(64,  32)
        self.d1 = LightDecoderBlock_AG(32,  16)

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