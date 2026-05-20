import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. CÁC KHỐI CƠ BẢN (BASE BLOCKS)
# ==============================================================================
def get_activation(act_type):
    return nn.ReLU6(inplace=True)

class SquareDW(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, bias=False)
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

class Straight3x3Block(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw1 = SquareDW(dim)
        self.dw2 = SquareDW(dim)
        self.dw3 = SquareDW(dim)
        self.pw_fuse = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        self.act = get_activation(act_type)
        
    def forward(self, x):
        fused = self.bn_fuse(self.pw_fuse(self.dw3(self.dw2(self.dw1(x)))))
        return self.act(fused + x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, act_type='relu6'):
        super().__init__()
        self.block = Straight3x3Block(in_c, act_type)
        self.down_pool = nn.MaxPool2d((2, 2))
        self.same_channels = (in_c == out_c)
        
        if not self.same_channels:
            self.pw = nn.Sequential(nn.Conv2d(in_c, out_c - in_c, 1, bias=False), nn.BatchNorm2d(out_c - in_c))
            self.down_pw = nn.MaxPool2d((2, 2))
            
        self.act = get_activation(act_type)
        
    def forward(self, x):
        feat = self.block(x)
        if self.same_channels: 
            return self.act(self.down_pool(feat)), feat
        else:
            feat_pw = self.pw(feat)
            skip = torch.cat([feat, feat_pw], dim=1)
            out = self.act(torch.cat([self.down_pool(feat), self.down_pw(feat_pw)], dim=1))
            return out, skip

class ConcatDecoderBlock_NoAttn(nn.Module):
    def __init__(self, in_c, skip_c, out_c, act_type='relu6'):
        super().__init__()
        self.up = NearestUpsample(in_c)
        gc = max(out_c // 4, 4)
        
        self.refine = nn.Sequential(
            nn.Conv2d(in_c + skip_c, gc, 1, bias=False), 
            nn.BatchNorm2d(gc), 
            get_activation(act_type), 
            SquareDW(gc, 5), 
            nn.Conv2d(gc, out_c, 1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        self.act = get_activation(act_type)
        
    def forward(self, x, skip):
        fused = torch.cat([self.up(x), skip], dim=1)
        return self.act(self.refine(fused) + self.shortcut(fused))

class SerialBottleneck_NoAttn(nn.Module):
    def __init__(self, dim, act_type='relu6'):
        super().__init__()
        self.dw1 = SquareDW(dim)
        self.dw2 = SquareDW(dim)
        self.dw3 = SquareDW(dim)
        
    def forward(self, x): 
        return x + self.dw1(x) + self.dw2(self.dw1(x)) + self.dw3(self.dw2(self.dw1(x)))


# ==============================================================================
# 2. ABLATION 18: SHALLOW BOUNDARY-ONLY SUPERVISION (CHỈ DÙNG AUX Ở E2)
# ==============================================================================
class Ablation18_ShallowAuxOnly(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        self.e1 = EncoderBlock(16, 32)   # out: 32 channels
        self.e2 = EncoderBlock(32, 64)   # out: 64 channels
        self.e3 = EncoderBlock(64, 128)  # out: 128 channels
        self.e4 = EncoderBlock(128, 128) # out: 128 channels
        
        # Aux 1: Đặt ở E2 (Tầng nông, độ phân giải cao). 
        # Cắt bỏ hoàn toàn Aux ở E4, dồn toàn lực bắt mạng học ranh giới ở E2.
        self.aux_head_detail = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        self.bottleneck = SerialBottleneck_NoAttn(128)
        
        self.d4 = ConcatDecoderBlock_NoAttn(128, 128, 128) 
        self.d3 = ConcatDecoderBlock_NoAttn(128, 128, 64)  
        self.d2 = ConcatDecoderBlock_NoAttn(64, 64, 32)   
        self.d1 = ConcatDecoderBlock_NoAttn(32, 32, 16)   
        
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[2:] 

        x = self.conv_in(x)
        x, s1 = self.e1(x)
        x, s2 = self.e2(x) # Lấy s2 cho Detail Aux
        x, s3 = self.e3(x)
        x, s4 = self.e4(x) 
        
        out_detail = None
        
        if self.training:
            # Chỉ tính toán 1 nhánh Aux duy nhất tại E2
            out_detail = self.aux_head_detail(s2)
            # Phóng to về kích thước gốc
            out_detail = F.interpolate(out_detail, size=input_shape, mode='bilinear', align_corners=False)
        
        x = self.bottleneck(x)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        
        main_out = self.conv_out(x)
        
        if self.training:
            # Tuple trả về: (Nhánh chính, Nhánh chi tiết viền E2)
            return main_out, out_detail
        return main_out


def build_model(num_classes=1, input_size=128):
    """
    Hàm gọi khởi tạo mô hình dùng cho file init của models/
    """
    return Ablation18_ShallowAuxOnly(num_classes=num_classes, input_size=input_size)