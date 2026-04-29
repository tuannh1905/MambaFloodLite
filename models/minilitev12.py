import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 0. CUSTOM ACTIVATIONS
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
# PICO-UNET V21: THE GHOST-V4 EDITION
# - ✓ GHOST ENCODER: Thay thế toàn bộ cụm PFCU_DG nặng nề bằng GhostModule.
# - ✓ LINEAR GHOSTS: Nhánh sinh Ghost không dùng phi tuyến, giữ đặc tính làm mờ của V4.
# - ✓ LÕI V4: Giữ nguyên Attention, Bottleneck V4, và Additive Decoder V4.
# - ✓ DOWNSAMPLE: Trở lại dùng Learnable Downsample (Stride=2) bảo vệ viền.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES
# ==============================================================================
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
    # Dùng cho Bottleneck và Decoder
    def __init__(self, dim, kernel_size=3, act_type='relu6'):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))

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
# 3. KHỐI ENCODER GHOST (THAY MÁU HOÀN TOÀN)
# ==============================================================================
class LinearGhostModule(nn.Module):
    """
    Tạo ra một nửa số kênh bằng 1x1 Conv (Đắt).
    Sinh ra nửa còn lại bằng DW Conv (Siêu rẻ, KHÔNG dùng Activation để giữ hồn V4).
    """
    def __init__(self, in_c, out_c, dw_size=3, act_type='relu6'):
        super().__init__()
        self.out_c = out_c
        init_channels = out_c // 2
        new_channels = out_c - init_channels

        # Nhánh Primary (Intrinsic Features)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_c, init_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(init_channels),
            get_activation(act_type)
        )

        # Nhánh Ghost (Cheap Operation - Linear Manifold)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=dw_size, padding=dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels)
            # ✓ Bỏ qua Activation ở đây để tạo độ mượt cho nước (V4 Style)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # Nối lại là đủ 100% out_c
        return torch.cat([x1, x2], dim=1)

class GhostEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, is_deep=False, act_type='relu6'):
        super().__init__()
        # Tầng sâu dùng DW 5x5 để làm Ghost nhằm tăng Receptive Field
        dw_size = 5 if is_deep else 3
        
        # Thay thế hoàn toàn Dual/Multi Scale của V4 bằng GhostModule
        self.ghost_extract = LinearGhostModule(in_c, out_c, dw_size=dw_size, act_type=act_type)
            
        self.same_channels = (in_c == out_c)
        
        # Kỹ thuật Learnable Downsample Stride=2
        self.down_dw = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            get_activation(act_type)
        )

    def forward(self, x):
        # Trích xuất đặc trưng
        feat = self.ghost_extract(x)
        
        # Nhánh skip (Nếu in_c != out_c thì GhostModule đã lo việc tăng kênh rồi)
        skip = feat
            
        # Hạ mẫu đưa xuống tầng dưới
        out = self.down_dw(skip)
        return out, skip

# ==============================================================================
# 4. DECODER & BOTTLE-NECK CỦA V4
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
            
            SquareDW(gc, kernel_size=5, act_type=act_type), 
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

class SerialMultiScaleBottleneck(nn.Module):
    def __init__(self, dim, act_type='hswish'):
        super().__init__()
        # V4 Serial DW 
        self.dw_3x3 = SquareDW(dim, kernel_size=3, act_type=act_type)
        self.dw_5x5 = SquareDW(dim, kernel_size=3, act_type=act_type) 
        self.dw_7x7 = SquareDW(dim, kernel_size=3, act_type=act_type) 
        
        self.channel_attn = ECABlock(dim, act_type)
        self.spatial_attn = SpatialAttention_MCU(kernel_size=3)

    def forward(self, x):
        d1 = self.dw_3x3(x)        
        d2 = self.dw_5x5(d1)        
        d3 = self.dw_7x7(d2)        
        
        fused = d1 + d2 + d3
        out = self.channel_attn(fused)
        out = self.spatial_attn(out)
        return x + out

# ==============================================================================
# 5. MẠNG CHÍNH PICO-UNET V21 (GHOST EDITION)
# ==============================================================================
class PicoUNet_v21_Ghost(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError(f"Input_size phải chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # ✓ Thay toàn bộ bằng khối GhostEncoderBlock siêu rẻ MACs
        self.e1 = GhostEncoderBlock(32, 64,  is_deep=False, act_type='relu6')   
        self.e2 = GhostEncoderBlock(64, 128, is_deep=False, act_type='relu6')   
        self.e3 = GhostEncoderBlock(128, 192, is_deep=True, act_type='hswish') 
        self.e4 = GhostEncoderBlock(192, 192, is_deep=True, act_type='hswish') 
        
        self.bottleneck = SerialMultiScaleBottleneck(192, act_type='hswish')
        
        self.d4 = AdditiveDecoderBlock(in_c=192, skip_c=192, out_c=128, act_type='hswish') 
        self.d3 = AdditiveDecoderBlock(in_c=128, skip_c=192, out_c=64,  act_type='hswish')  
        self.d2 = AdditiveDecoderBlock(in_c=64,  skip_c=128, out_c=32,  act_type='hswish')   
        self.d1 = AdditiveDecoderBlock(in_c=32,  skip_c=64,  out_c=16,  act_type='hswish')   
        
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
    return PicoUNet_v21_Ghost(num_classes=num_classes, input_size=input_size)