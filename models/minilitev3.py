import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# LƯU Ý TƯƠNG THÍCH ONNX / TORCH.FX / MCU:
# - ✓ ĐÃ CẬP NHẬT 1: ECABlock hoàn toàn sạch bóng lệnh permute(), dùng Conv2d 1x1.
# - ✓ ĐÃ CẬP NHẬT 2: Thin Stem - Bắt đầu từ 8 kênh thay vì 16 kênh để ép FLOPs.
# - ✓ ĐÃ CẬP NHẬT 3: Factorized_PFCU_DG - Đổi 3 nhánh (3,5,7) thành 2 nhánh (3,5).
# - ✓ ĐÃ CẬP NHẬT 4: SequentialDilatedBottleneck - Xóa bỏ F.interpolate và SPP.
# - BOTTLE-NECK TĨNH & NEAREST UPSAMPLE: Giữ nguyên để tương thích.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES (ECA TỐI ƯU MCU)
# ==============================================================================
class ECABlock(nn.Module):
    """
    ✓ TỐI ƯU: Đã xóa sổ hoàn toàn permute().
    Dùng Conv2d 1x1 trên ma trận (B, C, 1, 1) để học Channel Attention.
    """
    def __init__(self, channels):
        super().__init__()
        mid_channels = max(8, channels // 4)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        )
        self.hardsigmoid = nn.Hardsigmoid() 

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)              
        y = self.hardsigmoid(self.conv(y))                     
        return x * y

# ==============================================================================
# 2. NEAREST UPSAMPLE & SIMPLE FUSION (GIỮ NGUYÊN BẢN ABLATION)
# ==============================================================================
class NearestUpsample(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.up     = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn     = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.refine(self.up(x)))

class SimpleConcatFusion(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x_up, x_skip):
        fused = torch.cat([x_up, x_skip], dim=1)
        return self.fuse_conv(fused)

# ==============================================================================
# 3. KHỐI ENCODER MỚI: FACTORIZED INCEPTION-LITE
# ==============================================================================
class SquareDW(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, 
                            dilation=dilation, groups=dim, bias=False)
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

class Factorized_PFCU_DG(nn.Module):
    """
    ✓ TỐI ƯU: Chỉ còn 2 nhánh song song (giảm Peak RAM).
    Nhánh 5x5 được tách thành 2 lớp DW 3x3 chạy nối tiếp để MCU tính nhanh hơn.
    """
    def __init__(self, dim):
        super().__init__()
        # Nhánh 1: Bắt chi tiết (3x3)
        self.branch_3 = SquareDW(dim, kernel_size=3)
        
        # Nhánh 2: Bắt ngữ cảnh (5x5 phân rã)
        self.branch_5_part1 = SquareDW(dim, kernel_size=3)
        self.branch_5_part2 = SquareDW(dim, kernel_size=3)
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim) 
        
        self.dg_shortcut = DetailGuidance(dim)
        self.eca = ECABlock(dim) 
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        b3 = self.branch_3(x)
        b5 = self.branch_5_part2(self.branch_5_part1(x))
        
        fused_context = self.bn_fuse(self.pw_fuse(b3 + b5))
        guided_details = self.dg_shortcut(x)
        return self.eca(self.act(fused_context + guided_details))

# ==============================================================================
# 4. ENCODER, DECODER & BOTTLE-NECK GIÃN NỞ TUẦN TỰ
# ==============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.pfcu_dg   = Factorized_PFCU_DG(in_c)
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

class LightDecoderBlock_NoUAFM(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        gc = max(out_c // 4, 4)

        self.up   = NearestUpsample(in_c, scale_factor=2)
        self.fusion = SimpleConcatFusion(in_c=in_c, skip_c=in_c, out_c=out_c)

        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        self.bn_down = nn.BatchNorm2d(gc)
        
        self.refine_spatial = SquareDW(gc, kernel_size=5)
        self.eca = ECABlock(gc) 
        
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        self.bn_up  = nn.BatchNorm2d(out_c)
        
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        fused = self.fusion(x, skip)
        
        feat = self.act(self.bn_down(self.pw_down(fused)))
        feat = self.refine_spatial(feat)
        feat = self.eca(feat)
        
        out = self.bn_up(self.pw_up(feat))
        return self.act(out + fused)

class SequentialDilatedBottleneck(nn.Module):
    """
    ✓ TỐI ƯU: Thay thế hoàn toàn SPP nặng nề.
    Chạy 3 lớp DW 3x3 tuần tự với Dilation tăng dần (1 -> 2 -> 4).
    Tạo ra Receptive Field tương đương 15x15 nhưng bộ nhớ chỉ nạp 1 đường thẳng.
    """
    def __init__(self, dim):
        super().__init__()
        self.dw1 = SquareDW(dim, kernel_size=3, dilation=1)
        self.dw2 = SquareDW(dim, kernel_size=3, dilation=2)
        self.dw3 = SquareDW(dim, kernel_size=3, dilation=4)
        
        self.pw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True)
        )
        self.eca = ECABlock(dim)

    def forward(self, x):
        out = self.dw1(x)
        out = self.dw2(out)
        out = self.dw3(out)
        out = self.pw(out)
        out = self.eca(out)
        return out + x

# ==============================================================================
# 5. MẠNG CHÍNH (THIN STEM: 128x128 INPUT TỐI ƯU)
# ==============================================================================
class PicoUNet_Ablation_NoUAFM(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"PicoUNet yêu cầu input_size chia hết cho 16.")

        # ✓ TỐI ƯU: Thin Stem. Bắt đầu với 8 kênh thay vì 16 kênh.
        self.conv_in = nn.Conv2d(3, 8, kernel_size=3, padding=1)

        # Số kênh nở dần: 8 -> 16 -> 32 -> 64 -> 128
        self.e1 = EncoderBlock(8,   16)
        self.e2 = EncoderBlock(16,  32)
        self.e3 = EncoderBlock(32,  64)
        self.e4 = EncoderBlock(64, 128)

        # Bottleneck mới: Dilation tuần tự
        self.b4 = SequentialDilatedBottleneck(128)

        # Decoder lùi dần số kênh: 128 -> 64 -> 32 -> 16 -> 8
        self.d4 = LightDecoderBlock_NoUAFM(128, 64)
        self.d3 = LightDecoderBlock_NoUAFM(64,  32)
        self.d2 = LightDecoderBlock_NoUAFM(32,  16)
        self.d1 = LightDecoderBlock_NoUAFM(16,   8)

        self.conv_out = nn.Conv2d(8, num_classes, kernel_size=1)

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

def build_model(num_classes=1, input_size=128):
    return PicoUNet_Ablation_NoUAFM(num_classes=num_classes, input_size=input_size)