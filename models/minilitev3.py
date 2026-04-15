import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# LƯU Ý TƯƠNG THÍCH ONNX / TORCH.FX / MCU:
# - ✓ ĐÃ TỐI ƯU 1: Xóa permute trong ECABlock, dùng Conv2d 1x1.
# - ✓ ĐÃ TỐI ƯU 2: Gom 3 nhánh 3-5-7 thành 1 nhánh 7x7 trong PFCU để chống Cache Miss.
# - ✓ ĐÃ TỐI ƯU 3: Thay F.interpolate động bằng nn.Upsample tĩnh trong SPP Bottleneck.
# - BOTTLE-NECK TĨNH: Không dùng AdaptiveAvgPool2d, tính trước kernel trong __init__.
# - LƯỢNG HÓA INT8: Chuẩn hóa 100% dùng ReLU6 và Hardsigmoid.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES (ECA OPTIMIZED FOR MCU)
# ==============================================================================
class ECABlock(nn.Module):
    """
    ✓ ĐÃ TỐI ƯU 1: Xóa bỏ permute().
    Dùng Conv2d 1x1 để học sự chú ý của kênh trên không gian 1x1.
    Giữ nguyên định dạng NCHW giúp CPU không phải copy bộ nhớ.
    """
    def __init__(self, channels, k_size=3):
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
    """
    ✓ ĐÃ TỐI ƯU 2: Chống phân mảnh Cache (Cache Thrashing).
    Tên class giữ nguyên để tương thích code, nhưng ruột đã gộp 3 nhánh 
    song song thành 1 luồng DW 7x7 duy nhất.
    """
    def __init__(self, dim):
        super().__init__()
        # Thay 3 nhánh 3x3, 5x5, 7x7 bằng 1 nhánh 7x7 duy nhất
        self.large_kernel = SquareDW(dim, kernel_size=7)
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim) 
        
        self.dg_shortcut = DetailGuidance(dim)
        self.eca = ECABlock(dim) 
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        # Đọc RAM 1 lần, lướt 1 lần, giải phóng RAM ngay lập tức
        context = self.large_kernel(x)
        fused_context = self.bn_fuse(self.pw_fuse(context))
        
        guided_details = self.dg_shortcut(x)
        return self.eca(self.act(fused_context + guided_details))

# ==============================================================================
# 4. ENCODER, DECODER & BOTTLE-NECK TĨNH
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

class BottleNeckBlock_Static(nn.Module):
    """
    ✓ ĐÃ TỐI ƯU 3: Thay F.interpolate bằng nn.Upsample tĩnh.
    Loại bỏ vòng lặp tính toán tọa độ nội suy của CPU lúc runtime.
    """
    def __init__(self, dim, max_dim=128, input_size=256):
        super().__init__()
        hid = min(dim // 4, max_dim // 4)

        feat_size = input_size // 16  
        k1, k2, k3 = feat_size, feat_size // 2, feat_size // 4

        if input_size == 128:
            assert k3 >= 2, "input_size quá nhỏ"
        elif input_size == 256:
            pass
        else:
            raise ValueError(f"input_size={input_size} chưa được hỗ trợ.")

        self._sf1, self._sf2, self._sf3 = int(k1), int(k2), int(k3)

        # Khai báo sẵn các hàm Upsample tĩnh
        self.up1 = nn.Upsample(scale_factor=self._sf1, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=self._sf2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=self._sf3, mode='nearest')

        self.pool1 = nn.Sequential(nn.AvgPool2d(k1, k1), nn.Conv2d(dim, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(True))
        self.pool2 = nn.Sequential(nn.AvgPool2d(k2, k2), nn.Conv2d(dim, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(True))
        self.pool3 = nn.Sequential(nn.AvgPool2d(k3, k3), nn.Conv2d(dim, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(True))

        self.spp_fuse = nn.Sequential(
            nn.Conv2d(dim + hid * 3, dim, 1, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU6(inplace=True)
        )
        
        self.square_refine = SquareDW(dim, kernel_size=5)
        self.se = ECABlock(dim)

    def forward(self, x):
        # Gọi trực tiếp các hàm tĩnh đã được biên dịch sẵn
        x1 = self.up1(self.pool1(x))
        x2 = self.up2(self.pool2(x))
        x3 = self.up3(self.pool3(x))

        spp = self.spp_fuse(torch.cat([x, x1, x2, x3], dim=1))
        out = self.square_refine(spp)
        return self.se(out) + x

# ==============================================================================
# 5. MẠNG CHÍNH
# ==============================================================================
class PicoUNet_Ablation_NoUAFM(nn.Module):
    def __init__(self, num_classes=1, input_size=256):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"PicoUNet yêu cầu input_size chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32)
        self.e2 = EncoderBlock(32,  64)
        self.e3 = EncoderBlock(64,  128)
        self.e4 = EncoderBlock(128, 256)

        self.b4 = BottleNeckBlock_Static(256, max_dim=128, input_size=input_size)

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

def build_model(num_classes=1, input_size=256):
    return PicoUNet_Ablation_NoUAFM(num_classes=num_classes, input_size=input_size)