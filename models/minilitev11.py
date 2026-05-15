import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 0. CUSTOM ACTIVATIONS CHO ONNX OPSET 11 
# ==============================================================================
def get_activation(act_type):
    # Dùng ReLU6 cho toàn bộ để tối đa hóa FPS trên MCU
    return nn.ReLU6(inplace=True)

# ==============================================================================
# 1. KHỐI TÍCH CHẬP VÀ UPSAMPLE CƠ BẢN
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

# ==============================================================================
# 2. ENCODER TỐI GIẢN: CHỈ DÙNG 3 LỚP 3x3 NỐI TIẾP (KHÔNG MULTI-SCALE)
# ==============================================================================
class Straight3x3Block(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        # 3 lớp 3x3 nối tiếp nhau tạo ra trường nhìn (Receptive Field) 7x7 
        # nhưng tuyệt đối không có sự phân nhánh Multi-scale
        self.dw1 = SquareDW(dim, kernel_size=3)
        self.dw2 = SquareDW(dim, kernel_size=3)
        self.dw3 = SquareDW(dim, kernel_size=3)
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)

    def forward(self, x):
        out = self.dw1(x)        
        out = self.dw2(out)        
        out = self.dw3(out)        
        
        fused = self.bn_fuse(self.pw_fuse(out))
        return self.act(fused + x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, act_type='relu6'):
        super().__init__()
        self.block = Straight3x3Block(in_c, act_type)
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
# 3. DECODER & BOTTLE-NECK (KHÔNG ATTENTION)
# ==============================================================================
class ConcatDecoderBlock_NoAttn(nn.Module):
    def __init__(self, in_c, skip_c, out_c, act_type='relu6'):
        super().__init__()
        self.up = NearestUpsample(in_c)
        concat_channels = in_c + skip_c
        gc = max(out_c // 4, 4)
        
        # [LOẠI BỎ HOÀN TOÀN ECABlock]
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

class SerialBottleneck_NoAttn(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw1 = SquareDW(dim, kernel_size=3)
        self.dw2 = SquareDW(dim, kernel_size=3) 
        self.dw3 = SquareDW(dim, kernel_size=3) 

    def forward(self, x):
        d1 = self.dw1(x)        
        d2 = self.dw2(d1)        
        d3 = self.dw3(d2)        
        fused = d1 + d2 + d3
        return x + fused

# ==============================================================================
# 4. MẠNG CHÍNH PICO-UNET-MICRO (CÓ AUXILIARY HEAD)
# ==============================================================================
class PicoUNet_Micro_Aux(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        if input_size % 16 != 0:
            raise ValueError("Input_size phải chia hết cho 16.")

        self.conv_in = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # Encoder: Sử dụng Straight 3x3 Block
        self.e1 = EncoderBlock(32, 64,   act_type='relu6')   
        self.e2 = EncoderBlock(64, 128,  act_type='relu6')   
        self.e3 = EncoderBlock(128, 128, act_type='relu6') 
        self.e4 = EncoderBlock(128, 128, act_type='relu6') 
        
        # Nhánh phụ (Auxiliary Head) chĩa ra từ E4 (kích thước H/16, W/16)
        aux_dim = 64
        self.aux_head = nn.Sequential(
            nn.Conv2d(128, aux_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aux_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(aux_dim, num_classes, kernel_size=1)
        )
        self.aux_upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        
        # Bottleneck & Decoder (Hoàn toàn không Attention)
        self.bottleneck = SerialBottleneck_NoAttn(128, act_type='relu6')
        
        self.d4 = ConcatDecoderBlock_NoAttn(in_c=128, skip_c=128, out_c=128, act_type='relu6') 
        self.d3 = ConcatDecoderBlock_NoAttn(in_c=128, skip_c=128, out_c=64,  act_type='relu6')  
        self.d2 = ConcatDecoderBlock_NoAttn(in_c=64,  skip_c=128, out_c=32,  act_type='relu6')   
        self.d1 = ConcatDecoderBlock_NoAttn(in_c=32,  skip_c=64,  out_c=16,  act_type='relu6')   
        
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        
        # Tách nhánh Aux Head (Chỉ chạy khi đang Training)
        aux_out = None
        if self.training:
            aux_out = self.aux_head(s4)
            aux_out = self.aux_upsample(aux_out)
        
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
    return PicoUNet_Micro_Aux(num_classes=num_classes, input_size=input_size)