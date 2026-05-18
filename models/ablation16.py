import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 0. CUSTOM ACTIVATIONS CHO MCU
# ==============================================================================
def get_activation(act_type):
    return nn.ReLU6(inplace=True)

# ==============================================================================
# 1. KHỐI TÍCH CHẬP VÀ UPSAMPLE
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
# 2. ENCODER TỐI GIẢN
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
# 3. DECODER & BOTTLE-NECK
# ==============================================================================
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
# 4. MẠNG ABLATION 16: DEEP SUPERVISION (AUX HEADS Ở TẤT CẢ CÁC TẦNG)
# ==============================================================================
class Ablation16_DeepSupervision(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # Encoder 16 -> 32 -> 64 -> 128
        self.e1 = EncoderBlock(16, 32,   act_type='relu6')   # skip 1: 32 channels
        self.e2 = EncoderBlock(32, 64,   act_type='relu6')   # skip 2: 64 channels
        self.e3 = EncoderBlock(64, 128,  act_type='relu6')   # skip 3: 128 channels
        self.e4 = EncoderBlock(128, 128, act_type='relu6')   # skip 4: 128 channels
        
        # --- KHỞI TẠO 4 AUX HEADS CHO 4 TẦNG ---
        def create_aux_head(in_dim, aux_dim=32):
            return nn.Sequential(
                nn.Conv2d(in_dim, aux_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(aux_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(aux_dim, num_classes, kernel_size=1)
            )
            
        self.aux_head1 = create_aux_head(in_dim=32)
        self.aux_head2 = create_aux_head(in_dim=64)
        self.aux_head3 = create_aux_head(in_dim=128)
        self.aux_head4 = create_aux_head(in_dim=128)
        # ----------------------------------------
        
        self.bottleneck = SerialBottleneck_NoAttn(128, act_type='relu6')
        
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
        
        # --- TÍNH TOÁN 4 AUX OUTPUTS KHI TRAINING ---
        aux_outputs = []
        if self.training:
            a1 = self.aux_head1(s1)
            a2 = self.aux_head2(s2)
            a3 = self.aux_head3(s3)
            a4 = self.aux_head4(s4)
            
            # Phóng to tất cả về kích thước gốc
            a1 = F.interpolate(a1, size=input_shape, mode='bilinear', align_corners=False)
            a2 = F.interpolate(a2, size=input_shape, mode='bilinear', align_corners=False)
            a3 = F.interpolate(a3, size=input_shape, mode='bilinear', align_corners=False)
            a4 = F.interpolate(a4, size=input_shape, mode='bilinear', align_corners=False)
            
            # Đóng gói thành tuple
            aux_outputs = (a4, a3, a2, a1) # Sắp xếp từ tầng sâu nhất đến nông nhất
        # ---------------------------------------------
        
        x = self.bottleneck(x)
        
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        
        main_out = self.conv_out(x)
        
        if self.training:
            # Trả về main_out và cả 4 aux_out
            return main_out, aux_outputs[0], aux_outputs[1], aux_outputs[2], aux_outputs[3]
        return main_out

def build_model(num_classes=1, input_size=128):
    return Ablation16_DeepSupervision(num_classes=num_classes, input_size=input_size)