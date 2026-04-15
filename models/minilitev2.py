import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# LƯU Ý TƯƠNG THÍCH ONNX / TORCH.FX / MCU:
# - ✓ ĐÃ CẬP NHẬT 1: ECABlock hoàn toàn sạch bóng lệnh permute(), dùng Conv2d 1x1.
# - ✓ ĐÃ CẬP NHẬT 2: Phục hồi Base Channel = 16 (Tối ưu cho 128x128 Input).
# - ✓ ĐÃ CẬP NHẬT 3: Factorized_PFCU_DG - Feature Reuse 3 lớp 3x3 để tạo 3x3, 5x5, 7x7.
# - ✓ ĐÃ CẬP NHẬT 4: Xóa sạch từ khóa `dilation` khỏi mọi class (Chống MCU lỗi).
# - ✓ ĐÃ CẬP NHẬT 5: PerfectedBottleneck - 2 lớp DW 3x3, 0 Pointwise thừa, 0 bóp kênh.
# - ✓ ĐÃ CẬP NHẬT 6: ECABlock_Lite - Tăng tỉ lệ nén (r=8) để giảm 50% tham số Attention.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES (ECA TỐI ƯU MCU)
# ==============================================================================
class ECABlock_Lite(nn.Module):
    """
    ✓ TỐI ƯU: Bản thay thế hoàn hảo cho khối Squeeze-and-Excitation (FC layers).
    Dùng Conv2d 1x1 để học Channel Attention không cần xáo trộn bộ nhớ.
    Tỉ lệ nén r=8 (channels // 8) giúp giảm 50% tham số so với r=4.
    """
    def __init__(self, channels):
        super().__init__()
        mid_channels = max(8, channels // 8) 
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
# 2. NEAREST UPSAMPLE & SIMPLE FUSION
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
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
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
    ✓ TỐI ƯU: Feature Reuse (Tái sử dụng đặc trưng).
    Thay vì 3 nhánh song song, ta chạy 3 lớp 3x3 nối tiếp và trích xuất ở từng chặng.
    Vừa đạt được 3x3, 5x5, 7x7, vừa ép Peak RAM xuống mức cực thấp.
    """
    def __init__(self, dim):
        super().__init__()
        self.dw_3x3 = SquareDW(dim)
        self.dw_5x5 = SquareDW(dim) # Chạy tiếp từ 3x3
        self.dw_7x7 = SquareDW(dim) # Chạy tiếp từ 5x5
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim) 
        
        self.dg_shortcut = DetailGuidance(dim)
        self.eca = ECABlock_Lite(dim) 
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        b3 = self.dw_3x3(x)         # Receptive Field: 3x3
        b5 = self.dw_5x5(b3)        # Receptive Field: 5x5
        b7 = self.dw_7x7(b5)        # Receptive Field: 7x7
        
        fused_context = self.bn_fuse(self.pw_fuse(b3 + b5 + b7))
        guided_details = self.dg_shortcut(x)
        return self.eca(self.act(fused_context + guided_details))

# ==============================================================================
# 4. ENCODER, DECODER & PERFECTED BOTTLE-NECK
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
        
        self.refine_spatial_1 = SquareDW(gc)
        self.refine_spatial_2 = SquareDW(gc)
        
        self.eca = ECABlock_Lite(gc) 
        
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        self.bn_up  = nn.BatchNorm2d(out_c)
        
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        fused = self.fusion(x, skip)
        
        feat = self.act(self.bn_down(self.pw_down(fused)))
        feat = self.refine_spatial_2(self.refine_spatial_1(feat))
        feat = self.eca(feat)
        
        out = self.bn_up(self.pw_up(feat))
        return self.act(out + fused)

class PerfectedBottleneck(nn.Module):
    """
    ✓ TỐI ƯU 1: Chỉ dùng 2 lớp DW 3x3 (Góc nhìn 5x5) để chống Nhiễu viền Padding ở ảnh 8x8.
    ✓ TỐI ƯU 2: Xóa sổ lớp Pointwise (Conv 1x1) dư thừa, nhường việc trộn kênh cho Decoder.
    ✓ TỐI ƯU 3: Dữ liệu 256 kênh đi qua mà không bị bóp nghẹt hay đánh mất một bit nào!
    """
    def __init__(self, dim):
        super().__init__()
        self.dw1 = SquareDW(dim) # Góc nhìn: 3x3
        self.dw2 = SquareDW(dim) # Góc nhìn: 5x5
        
        self.eca = ECABlock_Lite(dim)

    def forward(self, x):
        out = self.dw2(self.dw1(x))
        out = self.eca(out)
        return out + x

# ==============================================================================
# 5. MẠNG CHÍNH (ĐÃ PHỤC HỒI BASE CHANNELS = 16)
# ==============================================================================
class PicoUNet_Ablation_NoUAFM(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"PicoUNet yêu cầu input_size chia hết cho 16.")

        # ✓ Đã phục hồi 16 kênh
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32)
        self.e2 = EncoderBlock(32,  64)
        self.e3 = EncoderBlock(64,  128)
        self.e4 = EncoderBlock(128, 256)

        # ✓ Thay StackedBottleneck bằng PerfectedBottleneck
        self.b4 = PerfectedBottleneck(256)

        self.d4 = LightDecoderBlock_NoUAFM(256, 128)
        self.d3 = LightDecoderBlock_NoUAFM(128, 64)
        self.d2 = LightDecoderBlock_NoUAFM(64,  32)
        self.d1 = LightDecoderBlock_NoUAFM(32,  16)

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

def build_model(num_classes=1, input_size=128):
    return PicoUNet_Ablation_NoUAFM(num_classes=num_classes, input_size=input_size)