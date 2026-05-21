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
# 2. ENCODER TỐI GIẢN (GIỮ NGUYÊN)
# ==============================================================================
class Straight3x3Block(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
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
# 3. DECODER & BOTTLE-NECK [KHÔNG SKIP CONNECTION]
# ==============================================================================
class DecoderBlock_NoSkip(nn.Module):
    def __init__(self, in_c, out_c, act_type='relu6'):
        super().__init__()
        self.up = NearestUpsample(in_c)
        # Vì không còn nối skip_c nên in_c được giữ nguyên
        gc = max(out_c // 4, 4)
        
        self.refine = nn.Sequential(
            nn.Conv2d(in_c, gc, kernel_size=1, bias=False), 
            nn.BatchNorm2d(gc), 
            get_activation(act_type),
            SquareDW(gc, kernel_size=5), 
            nn.Conv2d(gc, out_c, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act = get_activation(act_type)

    # Hàm forward bây giờ chỉ nhận duy nhất x (từ tầng dưới truyền lên)
    def forward(self, x):
        up_feat = self.up(x)
        return self.act(self.refine(up_feat) + self.shortcut(up_feat))

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
# 4. MẠNG ABLATION 7: LƯỢC BỎ HOÀN TOÀN SKIP CONNECTIONS
# ==============================================================================
class Ablation7_NoSkip(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        self.e1 = EncoderBlock(16, 32,   act_type='relu6')   
        self.e2 = EncoderBlock(32, 64,   act_type='relu6')   
        self.e3 = EncoderBlock(64, 128,  act_type='relu6') 
        self.e4 = EncoderBlock(128, 128, act_type='relu6') 
        
        aux_dim = 32
        self.aux_head = nn.Sequential(
            nn.Conv2d(128, aux_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aux_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(aux_dim, num_classes, kernel_size=1)
        )
        
        self.bottleneck = SerialBottleneck_NoAttn(128, act_type='relu6')
        
        # Decoder sử dụng khối NoSkip
        self.d4 = DecoderBlock_NoSkip(in_c=128, out_c=128, act_type='relu6') 
        self.d3 = DecoderBlock_NoSkip(in_c=128, out_c=64,  act_type='relu6')  
        self.d2 = DecoderBlock_NoSkip(in_c=64,  out_c=32,  act_type='relu6')   
        self.d1 = DecoderBlock_NoSkip(in_c=32,  out_c=16,  act_type='relu6')   
        
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[2:] 

        x = self.conv_in(x)
        
        # Vẫn gọi Encoder bình thường nhưng...
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        
        aux_out = None
        if self.training:
            aux_out = self.aux_head(s4)
            aux_out = F.interpolate(aux_out, size=input_shape, mode='bilinear', align_corners=False)
        
        x = self.bottleneck(x)
        
        # ...Tuyệt đối không dùng các biến s4, s3, s2, s1 ở Decoder
        x = self.d4(x)
        x = self.d3(x)
        x = self.d2(x)
        x = self.d1(x)
        
        main_out = self.conv_out(x)
        
        if self.training:
            return main_out, aux_out
        return main_out

def build_model(num_classes=1, input_size=128):
    return Ablation7_NoSkip(num_classes=num_classes, input_size=input_size)