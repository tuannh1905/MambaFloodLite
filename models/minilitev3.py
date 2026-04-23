import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# PICO-UNET V2: BẢN PAPER - TỐI ƯU HÓA CHO ESP32-S3 (CHUẨN TEMPLATE)
# - ✓ TỐI ƯU 1: ECABlock hoàn toàn sạch bóng lệnh permute(), dùng Conv2d 1x1.
# - ✓ TỐI ƯU 2: MultiScale_PFCU_DG_v2 áp dụng Feature Reuse (tiết kiệm Peak RAM).
# - ✓ TỐI ƯU 3: Giới hạn kênh ở 128 (Cap at 128) để giảm GFLOPs triệt để.
# - ✓ TỐI ƯU 4: LightInvertedBottleneck thay thế Bottleneck cũ siêu nhẹ.
# ==============================================================================

# ==============================================================================
# 1. MODULES CƠ BẢN VÀ ATTENTION
# ==============================================================================
class ECABlock(nn.Module):
    """ Bản thay thế hoàn hảo cho Squeeze-and-Excitation, 0% Memory Copy. """
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

class ChannelShuffle(nn.Module):
    """ Trộn kênh (Channel Shuffle) giúp giao thoa thông tin sau khi chia nhánh. """
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        
        # Reshape -> Transpose -> Reshape (Đã được contiguous bảo vệ layout)
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x

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

class SimpleConcatFusion(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c), 
            nn.ReLU6(inplace=True)
        )

    def forward(self, x, skip):
        return self.fuse(torch.cat([x, skip], dim=1))

# ==============================================================================
# 3. KHỐI ENCODER & DECODER V2 (CÓ SHUFFLE VÀ FEATURE REUSE)
# ==============================================================================
class MultiScale_PFCU_DG_v2(nn.Module):
    """
    ✓ ĐÃ TỐI ƯU TEMPLATE (Feature Reuse):
    Thay vì chạy 3 nhánh độc lập rồi cộng lại, ta chạy 3 lớp 3x3 nối tiếp.
    Dữ liệu chảy thành 1 đường thẳng, tiết kiệm cực nhiều Peak RAM.
    """
    def __init__(self, dim):
        super().__init__()
        self.dw_3x3 = SquareDW(dim)
        self.dw_5x5 = SquareDW(dim) # Chạy tiếp từ 3x3
        self.dw_7x7 = SquareDW(dim) # Chạy tiếp từ 5x5
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        
        self.shuffle = ChannelShuffle(groups=4)
        self.eca = ECABlock(dim)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        # Đường ống nối tiếp
        b3 = self.dw_3x3(x)         # RF: 3x3
        b5 = self.dw_5x5(b3)        # RF: 5x5
        b7 = self.dw_7x7(b5)        # RF: 7x7
        
        fused = self.bn_fuse(self.pw_fuse(b3 + b5 + b7))
        fused = self.shuffle(fused)
        return self.eca(self.act(fused + x))

class EncoderBlock_v2(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pfcu_dg = MultiScale_PFCU_DG_v2(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))
        
        self.same_channels = (in_c == out_c)
        if not self.same_channels:
            self.pw = nn.Sequential(
                nn.Conv2d(in_c, out_c - in_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c - in_c)
            )
            self.down_pw = nn.MaxPool2d((2, 2))
            
        self.act = nn.ReLU6(inplace=True)

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

class LightDecoderBlock_NoUAFM(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        gc = max(out_c // 4, 4)
        
        self.up = NearestUpsample(in_c)
        self.fusion = SimpleConcatFusion(in_c, in_c, out_c)
        
        self.refine = nn.Sequential(
            nn.Conv2d(out_c, gc, kernel_size=1, bias=False), 
            nn.BatchNorm2d(gc), 
            nn.ReLU6(inplace=True),
            
            SquareDW(gc, kernel_size=5), 
            ECABlock(gc),
            
            nn.Conv2d(gc, out_c, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x, skip):
        up = self.up(x)
        fused = self.fusion(up, skip)
        return self.act(self.refine(fused) + fused)

# ==============================================================================
# 4. BOTTLE-NECK ĐÁY MẠNG (INVERTED RESIDUAL)
# ==============================================================================
class LightInvertedBottleneck(nn.Module):
    """ 
    Bottleneck siêu nhẹ lấy cảm hứng từ MobileNetV2.
    Expand -> Depthwise -> Project.
    """
    def __init__(self, dim, expand_ratio=2):
        super().__init__()
        hid = dim * expand_ratio
        self.conv = nn.Sequential(
            # 1. Expand
            nn.Conv2d(dim, hid, kernel_size=1, bias=False),
            nn.BatchNorm2d(hid), 
            nn.ReLU6(inplace=True),
            # 2. Depthwise
            nn.Conv2d(hid, hid, kernel_size=3, padding=1, groups=hid, bias=False),
            nn.BatchNorm2d(hid), 
            nn.ReLU6(inplace=True),
            # 3. Project
            nn.Conv2d(hid, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        
    def forward(self, x): 
        return x + self.conv(x)

# ==============================================================================
# 5. MẠNG CHÍNH PICO-UNET V2 PAPER
# ==============================================================================
class PicoUNet_v2_Paper(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"Input_size phải chia hết cho 16.")

        # Khởi tạo Base Channel = 16
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # Encoder: 16 -> 32 -> 64 -> 128 -> 128 (Đã Cap ở 128 theo Paper)
        self.e1 = EncoderBlock_v2(16, 32)
        self.e2 = EncoderBlock_v2(32, 64)
        self.e3 = EncoderBlock_v2(64, 128)
        self.e4 = EncoderBlock_v2(128, 128) 
        
        # Bottleneck: Inverted Residual cực nhẹ (128 kênh)
        self.bottleneck = LightInvertedBottleneck(128)
        
        # Decoder lùi dần: 128 -> 128 -> 64 -> 32 -> 16
        self.d4 = LightDecoderBlock_NoUAFM(128, 128)
        self.d3 = LightDecoderBlock_NoUAFM(128, 64)
        self.d2 = LightDecoderBlock_NoUAFM(64, 32)
        self.d1 = LightDecoderBlock_NoUAFM(32, 16)
        
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
    """
    Khởi tạo Pico-UNet v2 dành cho Paper nghiên cứu.
    Đã tối ưu hóa kernel 3x3, bỏ permute và giới hạn 128 channels.
    """
    return PicoUNet_v2_Paper(num_classes=num_classes, input_size=input_size)