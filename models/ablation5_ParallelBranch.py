import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 0. CUSTOM ACTIVATIONS CHO MCU
# ==============================================================================
def get_activation(act_type):
    return nn.ReLU6(inplace=True)

# ==============================================================================
# 1. KHỐI TÍCH CHẬP VÀ UPSAMPLE CƠ BẢN
# ==============================================================================
class SquareDW(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # Depthwise Convolution
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

# ==============================================================================
# 2. KHỐI MULTI-SCALE PARALLEL (ĐA TỶ LỆ SONG SONG)
# ==============================================================================
class MultiScaleBlock(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        # 3 nhánh song song bắt các đặc trưng từ vi mô đến vĩ mô
        self.branch_3x3 = SquareDW(dim, kernel_size=3)
        self.branch_5x5 = SquareDW(dim, kernel_size=5)
        self.branch_7x7 = SquareDW(dim, kernel_size=7)
        
        # Nén 3 nhánh (dim * 3) trở lại kích thước gốc (dim)
        self.pw_fuse = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        # Dữ liệu đi song song vào cả 3 nhánh
        out_3 = self.branch_3x3(x)
        out_5 = self.branch_5x5(x)
        out_7 = self.branch_7x7(x)
        
        # Ghép nối dọc theo trục channel
        concat_out = torch.cat([out_3, out_5, out_7], dim=1)
        
        # Trộn đặc trưng và giảm chiều
        fused = self.bn_fuse(self.pw_fuse(concat_out))
        
        # Kết nối thặng dư (Residual Connection)
        return self.act(fused + x)

class EncoderBlock_MultiScale(nn.Module):
    def __init__(self, in_c, out_c, act_type='relu6'):
        super().__init__()
        # Sử dụng khối Multi-scale mới
        self.block = MultiScaleBlock(in_c, act_type)
        self.down_pool = nn.MaxPool2d((2, 2))
        
        self.same_channels = (in_c == out_c)
        if not self.same_channels:
            self.pw = nn.Sequential(
                nn.Conv2d(in_c, out_c - in_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c - in_c)
            )
            self.down_pw = nn.MaxPool2d((2, 2))
            
        self.act = get_activation(act_type)

    def forward(self, x):
        feat = self.block(x)
        
        if self.same_channels:
            return self.act(self.down_pool(feat)), feat
        else:
            feat_pw = self.pw(feat)
            skip = torch.cat([feat, feat_pw], dim=1)
            
            pool_feat = self.down_pool(feat)
            pool_pw   = self.down_pw(feat_pw)
            
            out = self.act(torch.cat([pool_feat, pool_pw], dim=1))
            return out, skip

# ==============================================================================
# 3. DECODER & BOTTLE-NECK (MULTI-SCALE)
# ==============================================================================
class SerialBottleneck_MultiScale(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        # Bottleneck cũng được hưởng lợi rất lớn từ Multi-scale
        # vì đây là tầng chứa ngữ nghĩa sâu nhất
        self.branch_3x3 = SquareDW(dim, kernel_size=3)
        self.branch_5x5 = SquareDW(dim, kernel_size=5)
        self.branch_7x7 = SquareDW(dim, kernel_size=7)
        self.pw_fuse = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)

    def forward(self, x):
        out_3 = self.branch_3x3(x)
        out_5 = self.branch_5x5(x)
        out_7 = self.branch_7x7(x)
        
        concat_out = torch.cat([out_3, out_5, out_7], dim=1)
        fused = self.bn_fuse(self.pw_fuse(concat_out))
        return x + fused

class ConcatDecoderBlock_NoAttn(nn.Module):
    def __init__(self, in_c, skip_c, out_c, act_type='relu6'):
        super().__init__()
        self.up = NearestUpsample(in_c)
        concat_channels = in_c + skip_c
        gc = max(out_c // 4, 4)
        
        self.refine = nn.Sequential(
            nn.Conv2d(concat_channels, gc, kernel_size=1, bias=False), 
            nn.BatchNorm2d(gc), 
            get_activation(act_type),
            SquareDW(gc, kernel_size=5), 
            nn.Conv2d(gc, out_c, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(concat_channels, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act = get_activation(act_type)

    def forward(self, x, skip):
        up_feat = self.up(x)
        fused = torch.cat([up_feat, skip], dim=1)
        return self.act(self.refine(fused) + self.shortcut(fused))

# ==============================================================================
# 4. MẠNG ABLATION 5: MULTI-SCALE PARALLEL 3x3, 5x5, 7x7
# ==============================================================================
class Ablation5_MultiScale(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        # Phiên bản ép cân: 16 -> 32 -> 64 -> 128
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # Áp dụng Encoder Multi-scale
        self.e1 = EncoderBlock_MultiScale(16, 32,   act_type='relu6')   
        self.e2 = EncoderBlock_MultiScale(32, 64,   act_type='relu6')   
        self.e3 = EncoderBlock_MultiScale(64, 128,  act_type='relu6') 
        self.e4 = EncoderBlock_MultiScale(128, 128, act_type='relu6') 
        
        # Nhánh phụ (Aux Head) - Kích thước 32
        aux_dim = 32
        self.aux_head = nn.Sequential(
            nn.Conv2d(128, aux_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aux_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(aux_dim, num_classes, kernel_size=1)
        )
        
        # Áp dụng Bottleneck Multi-scale
        self.bottleneck = SerialBottleneck_MultiScale(128, act_type='relu6')
        
        self.d4 = ConcatDecoderBlock_NoAttn(in_c=128, skip_c=128, out_c=128, act_type='relu6') 
        self.d3 = ConcatDecoderBlock_NoAttn(in_c=128, skip_c=128, out_c=64,  act_type='relu6')  
        self.d2 = ConcatDecoderBlock_NoAttn(in_c=64,  skip_c=64,  out_c=32,  act_type='relu6')   
        self.d1 = ConcatDecoderBlock_NoAttn(in_c=32,  skip_c=32,  out_c=16,  act_type='relu6')   
        
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[2:] 

        x = self.conv_in(x)
        
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        
        aux_out = None
        if self.training:
            aux_out = self.aux_head(s4)
            aux_out = F.interpolate(aux_out, size=input_shape, mode='bilinear', align_corners=False)
        
        x = self.bottleneck(x)
        
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        
        main_out = self.conv_out(x)
        
        if self.training:
            return main_out, aux_out
        return main_out

def build_model(num_classes=1, input_size=128):
    return Ablation5_MultiScale(num_classes=num_classes, input_size=input_size)