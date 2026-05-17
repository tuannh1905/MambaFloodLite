import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 0. CUSTOM ACTIVATIONS CHO MCU
# ==============================================================================
def get_activation(act_type):
    return nn.ReLU6(inplace=True)

# ==============================================================================
# 1. KHỐI TÍCH CHẬP, UPSAMPLE VÀ ECA MODULE (EFFICIENT CHANNEL ATTENTION)
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

class ECAModule(nn.Module):
    """
    Efficient Channel Attention (ECA).
    Học sự phụ thuộc giữa các kênh kề nhau thông qua Fast 1D Convolution.
    Params gần như bằng 0.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Tính kích thước kernel k tự động theo công thức trong paper
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Conv1d thay thế cho các lớp Linear (FC) nặng nề
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        # y: (B, C, 1, 1) -> (B, 1, C) để đưa vào Conv1d
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        # Nhân attention map vào feature gốc
        return x * y.expand_as(x)

# ==============================================================================
# 2. ENCODER TỐI GIẢN (TÍCH HỢP ECA MODULE)
# ==============================================================================
class Straight3x3Block_ECA(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw1 = SquareDW(dim, kernel_size=3)
        self.dw2 = SquareDW(dim, kernel_size=3)
        self.dw3 = SquareDW(dim, kernel_size=3)
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        
        # Thêm ECA Module vào trước khi cộng residual
        self.eca = ECAModule(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        out = self.dw1(x)        
        out = self.dw2(out)        
        out = self.dw3(out)        
        
        fused = self.bn_fuse(self.pw_fuse(out))
        fused = self.eca(fused) # Kích hoạt ECA
        return self.act(fused + x)

class EncoderBlock_ECA(nn.Module):
    def __init__(self, in_c, out_c, act_type='relu6'):
        super().__init__()
        self.block = Straight3x3Block_ECA(in_c, act_type)
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
# 3. DECODER & BOTTLE-NECK (TÍCH HỢP ECA MODULE)
# ==============================================================================
class ConcatDecoderBlock_ECA(nn.Module):
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
        
        # Thêm ECA Module cho nhánh Decoder
        self.eca = ECAModule(out_c)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(concat_channels, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act = get_activation(act_type)

    def forward(self, x, skip):
        up_feat = self.up(x)
        fused = torch.cat([up_feat, skip], dim=1)
        
        refined = self.refine(fused)
        refined = self.eca(refined) # Kích hoạt ECA
        
        return self.act(refined + self.shortcut(fused))

class SerialBottleneck_ECA(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw1 = SquareDW(dim, kernel_size=3)
        self.dw2 = SquareDW(dim, kernel_size=3) 
        self.dw3 = SquareDW(dim, kernel_size=3) 
        self.eca = ECAModule(dim) # Kích hoạt ECA cho Bottleneck

    def forward(self, x):
        d1 = self.dw1(x)        
        d2 = self.dw2(d1)        
        d3 = self.dw3(d2)        
        fused = d1 + d2 + d3
        fused = self.eca(fused)
        return x + fused

# ==============================================================================
# 4. MẠNG ABLATION 2: MINILITEV11 + EFFICIENT CHANNEL ATTENTION (ECA)
# ==============================================================================
class Ablation2_ECA(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        # Phiên bản ép cân: 16 -> 32 -> 64 -> 128
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        self.e1 = EncoderBlock_ECA(16, 32,   act_type='relu6')   
        self.e2 = EncoderBlock_ECA(32, 64,   act_type='relu6')   
        self.e3 = EncoderBlock_ECA(64, 128,  act_type='relu6') 
        self.e4 = EncoderBlock_ECA(128, 128, act_type='relu6') 
        
        # Nhánh phụ (Aux Head) - Kích thước 32
        aux_dim = 32
        self.aux_head = nn.Sequential(
            nn.Conv2d(128, aux_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aux_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(aux_dim, num_classes, kernel_size=1)
        )
        
        self.bottleneck = SerialBottleneck_ECA(128, act_type='relu6')
        
        self.d4 = ConcatDecoderBlock_ECA(in_c=128, skip_c=128, out_c=128, act_type='relu6') 
        self.d3 = ConcatDecoderBlock_ECA(in_c=128, skip_c=128, out_c=64,  act_type='relu6')  
        self.d2 = ConcatDecoderBlock_ECA(in_c=64,  skip_c=64,  out_c=32,  act_type='relu6')   
        self.d1 = ConcatDecoderBlock_ECA(in_c=32,  skip_c=32,  out_c=16,  act_type='relu6')   
        
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

# Hàm build chuẩn cho file get_model của bạn
def build_model(num_classes=1, input_size=128):
    return Ablation2_ECA(num_classes=num_classes, input_size=input_size)