import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# ABLATION STUDY: BASELINE
# - THAY THẾ: Toàn bộ MultiScale_PFCU_DG bằng StandardConv3x3Block.
# - GIỮ NGUYÊN: TinyUAFM_v2, SPP Bottleneck, Hardsigmoid, Static Tensor.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES (ECA) - Vẫn giữ cho SPP Bottleneck dùng
# ==============================================================================
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), padding=(0, k_size//2), bias=False)
        self.hardsigmoid = nn.Hardsigmoid() 

    def forward(self, x):
        B, C, _, _ = x.shape 
        y = torch.mean(x, dim=[2, 3], keepdim=True).reshape(B, 1, 1, C)              
        y = self.hardsigmoid(self.conv(y))                     
        y = y.reshape(B, C, 1, 1) 
        return x * y

# ==============================================================================
# 2. NEAREST UPSAMPLE & TINY-UAFM_V2
# ==============================================================================
class NearestUpsample(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.up     = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn     = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.refine(self.up(x)))

class TinyUAFM_v2(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.reduce_up   = nn.Conv2d(in_c,   out_c, 1, bias=False) if in_c   != out_c else nn.Identity()
        self.reduce_skip = nn.Conv2d(skip_c, out_c, 1, bias=False) if skip_c != out_c else nn.Identity()
        
        self.alpha_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), 
            nn.Hardsigmoid() 
        )
        self.beta_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), 
            nn.Hardsigmoid() 
        )
        self.channel_att = ECABlock(out_c)

    def forward(self, x_up, x_skip):
        x_up   = self.reduce_up(x_up)
        x_skip = self.reduce_skip(x_skip)
        
        spatial_input = torch.cat([
            torch.mean(x_up, dim=1, keepdim=True),
            torch.max(x_skip, dim=1, keepdim=True)[0] 
        ], dim=1)
        
        alpha = self.alpha_conv(spatial_input)
        beta  = self.beta_conv(spatial_input)
        
        out = x_up * alpha + x_skip * beta
        return self.channel_att(out)

# ==============================================================================
# 3. STANDARD CONV BLOCK (THAY THẾ CHO PFCU-DG)
# ==============================================================================
class SquareDW(nn.Module):
    # Giữ lại class này vì Bottleneck ở đáy mạng vẫn cần dùng
    def __init__(self, dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim) 

    def forward(self, x):
        return self.bn(self.dw(x)) 

class StandardConv3x3Block(nn.Module):
    """
    ✓ ABLATION BASELINE: Khối Conv2d 3x3 tiêu chuẩn dùng để làm "bao cát" 
    so sánh với khối MultiScale_PFCU_DG tinh vi của chúng ta.
    """
    def __init__(self, dim):
        super().__init__()
        # Conv chuẩn (không phải Depthwise), có padding để giữ nguyên H, W
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ==============================================================================
# 4. ENCODER, DECODER & BOTTLE-NECK TĨNH
# ==============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        # ✓ ĐÃ THAY THẾ: Dùng StandardConv3x3Block thay vì MultiScale_PFCU_DG
        self.pfcu_dg   = StandardConv3x3Block(in_c)
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

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        gc = max(out_c // 4, 4)

        self.up   = NearestUpsample(in_c, scale_factor=2)
        self.uafm = TinyUAFM_v2(in_c=in_c, skip_c=in_c, out_c=out_c)

        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        # ✓ ĐÃ THAY THẾ: Dùng StandardConv3x3Block thay vì MultiScale_PFCU_DG
        self.pfcu_dg = StandardConv3x3Block(gc)
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)

        self.bn  = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.uafm(x, skip)
        out = self.bn(self.pw_up(self.pfcu_dg(self.pw_down(x))))
        return self.act(out + x)

class BottleNeckBlock_Static(nn.Module):
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
        x1 = F.interpolate(self.pool1(x), scale_factor=self._sf1, mode='nearest')
        x2 = F.interpolate(self.pool2(x), scale_factor=self._sf2, mode='nearest')
        x3 = F.interpolate(self.pool3(x), scale_factor=self._sf3, mode='nearest')

        spp = self.spp_fuse(torch.cat([x, x1, x2, x3], dim=1))
        out = self.square_refine(spp)
        return self.se(out) + x

# ==============================================================================
# 5. MẠNG CHÍNH
# ==============================================================================
class PicoUNet_Ablation_Baseline(nn.Module):
    def __init__(self, num_classes=1, input_size=256):
        super().__init__()
        
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32)
        self.e2 = EncoderBlock(32,  64)
        self.e3 = EncoderBlock(64,  128)
        self.e4 = EncoderBlock(128, 256)

        self.b4 = BottleNeckBlock_Static(256, max_dim=128, input_size=input_size)

        self.d4 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64,  32)
        self.d1 = DecoderBlock(32,  16)

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
    return PicoUNet_Ablation_Baseline(num_classes=num_classes, input_size=input_size)