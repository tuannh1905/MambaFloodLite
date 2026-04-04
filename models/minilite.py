import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# LƯU Ý TƯƠNG THÍCH ONNX / TORCH.FX / MCU:
# - Dùng explicit reshape chống lỗi Batch Size = 1.
# - Loại bỏ hoàn toàn nội suy 'bilinear' (thay bằng nearest + DW refinement).
# - TinyUAFM_v2 loại bỏ if fallback để đảm bảo đồ thị tĩnh (Static Graph) 100%.
# - ✓ BOTTLE-NECK TĨNH: Dùng toán học tính trước kernel cho AvgPool2d trong __init__. 
#   Không dùng AdaptiveAvgPool2d để TFLite Micro có thể cấp phát RAM tĩnh.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES (ECA)
# ==============================================================================
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Với size (1,1) thì an toàn cho static graph
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).reshape(B, 1, C)              
        y = self.sigmoid(self.conv(y))                     
        y = y.reshape(B, C, 1, 1).expand_as(x)             
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
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid()
        )
        self.beta_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid()
        )
        self.channel_att = ECABlock(out_c)

    def forward(self, x_up, x_skip):
        x_up   = self.reduce_up(x_up)
        x_skip = self.reduce_skip(x_skip)
        
        spatial_input = torch.cat([
            torch.mean(x_up,         dim=1, keepdim=True),
            torch.mean(x_skip.abs(), dim=1, keepdim=True)
        ], dim=1)
        
        alpha = self.alpha_conv(spatial_input)
        beta  = self.beta_conv(spatial_input)
        
        out = x_up * alpha + x_skip * beta
        return self.channel_att(out)

# ==============================================================================
# 3. AXIAL-PFCU-DG BLOCK
# ==============================================================================
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        pad_h, pad_w = (h - 1) * dilation // 2, (w - 1) * dilation // 2
        
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding=(pad_h, 0), groups=dim, dilation=dilation, bias=False)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding=(0, pad_w), groups=dim, dilation=dilation, bias=False)

    def forward(self, x):
        return x + self.dw_h(x) + self.dw_w(x)

class DetailGuidance(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dg_dw_h = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=False)
        self.dg_dw_w = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        return self.bn(x + self.dg_dw_h(x) + self.dg_dw_w(x))

class Axial_PFCU_DG(nn.Module):
    def __init__(self, dim, mixer_kernel=(5, 5)):
        super().__init__()
        self.branch_r1 = AxialDW(dim, mixer_kernel, dilation=1)
        self.branch_r2 = AxialDW(dim, mixer_kernel, dilation=2)
        self.branch_r5 = AxialDW(dim, mixer_kernel, dilation=5)
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        
        self.dg_shortcut = DetailGuidance(dim)
        self.eca = ECABlock(dim) 
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        b1, b2, b5 = self.branch_r1(x), self.branch_r2(x), self.branch_r5(x)
        fused_context = self.bn_fuse(self.pw_fuse(b1 + b2 + b5))
        guided_details = self.dg_shortcut(x)
        return self.eca(self.act(fused_context + guided_details))

# ==============================================================================
# 4. ENCODER, DECODER & BOTTLE-NECK TĨNH
# ==============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.pfcu_dg   = Axial_PFCU_DG(in_c, mixer_kernel=mixer_kernel)
        self.bn        = nn.BatchNorm2d(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw      = nn.Conv2d(in_c, conv_out, kernel_size=1, bias=False)
            self.down_pw = nn.MaxPool2d((2, 2))

        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        feat = self.bn(self.pfcu_dg(x))

        if self.same_channels:
            return self.act(self.bn2(self.down_pool(feat))), feat
        else:
            feat_pw = self.pw(feat)
            skip = torch.cat([feat, feat_pw], dim=1) 
            
            pool_feat, pool_pw = self.down_pool(feat), self.down_pw(feat_pw)
            x = self.act(self.bn2(torch.cat([pool_feat, pool_pw], dim=1)))
            return x, skip

class BottleNeckBlock_Static(nn.Module):
    """
    Tính toán sẵn tham số Pooling trong __init__ dựa trên input_size.
    Đảm bảo 100% C++ Static Allocation, không có luồng điều khiển động (If-else) ở forward.
    """
    def __init__(self, dim, max_dim=128, input_size=256):
        super().__init__()
        hid = min(dim // 4, max_dim // 4)
        
        # Vì ảnh đã qua 4 lớp MaxPool (chia 2^4 = 16)
        feat_size = input_size // 16 
        
        # Tính toán kernel động theo input size, NHƯNG chốt tĩnh thành hằng số (int)
        k1 = feat_size           # Gom từ feat_size -> 1x1
        k2 = feat_size // 2      # Gom từ feat_size -> 2x2
        k3 = feat_size // 4      # Gom từ feat_size -> 4x4
        
        self._sf1, self._sf2, self._sf3 = k1, k2, k3
        
        # Sử dụng AvgPool2d cứng truyền thống thay cho Adaptive
        self.pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=k1, stride=k1), 
            nn.Conv2d(dim, hid, 1, bias=False),
            nn.BatchNorm2d(hid), nn.ReLU6(inplace=True)
        )
        self.pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=k2, stride=k2), 
            nn.Conv2d(dim, hid, 1, bias=False),
            nn.BatchNorm2d(hid), nn.ReLU6(inplace=True)
        )
        self.pool3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=k3, stride=k3), 
            nn.Conv2d(dim, hid, 1, bias=False),
            nn.BatchNorm2d(hid), nn.ReLU6(inplace=True)
        )
        
        self.spp_fuse = nn.Sequential(
            nn.Conv2d(dim + hid * 3, dim, 1, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU6(inplace=True)
        )
        self.axial_refine = AxialDW(dim, mixer_kernel=(5, 5), dilation=1)
        self.bn_refine = nn.BatchNorm2d(dim)
        self.se = ECABlock(dim)

    def forward(self, x):
        # Dùng scale_factor tĩnh (số nguyên) -> ONNX Trace hoàn hảo
        x1 = F.interpolate(self.pool1(x), scale_factor=self._sf1, mode='nearest')
        x2 = F.interpolate(self.pool2(x), scale_factor=self._sf2, mode='nearest')
        x3 = F.interpolate(self.pool3(x), scale_factor=self._sf3, mode='nearest')
        
        spp = self.spp_fuse(torch.cat([x, x1, x2, x3], dim=1))
        out = self.bn_refine(self.axial_refine(spp))
        return self.se(out) + x

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        gc = max(out_c // 4, 4)
        
        self.up = NearestUpsample(in_c, scale_factor=2)
        self.uafm = TinyUAFM_v2(in_c=in_c, skip_c=in_c, out_c=out_c)
        
        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        self.pfcu_dg = Axial_PFCU_DG(gc, mixer_kernel=mixer_kernel)
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        
        self.bn  = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.uafm(x, skip)
        x = self.act(self.bn(self.pw_up(self.pfcu_dg(self.pw_down(x))) + x))
        return x

# ==============================================================================
# 5. MẠNG CHÍNH (FINAL MCU-READY)
# ==============================================================================
class ULiteModel_MCU(nn.Module):
    # ✓ Thêm input_size vào __init__
    def __init__(self, num_classes=1, input_size=256): 
        super().__init__()
        mk = (5, 5)

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32,  mixer_kernel=mk)
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=mk)
        self.e3 = EncoderBlock(64,  128, mixer_kernel=mk)
        self.e4 = EncoderBlock(128, 256, mixer_kernel=mk)

        # ✓ Chuyền input_size vào BottleNeck
        self.b4 = BottleNeckBlock_Static(256, max_dim=128, input_size=input_size)

        self.d4 = DecoderBlock(256, 128, mixer_kernel=mk)
        self.d3 = DecoderBlock(128, 64,  mixer_kernel=mk)
        self.d2 = DecoderBlock(64,  32,  mixer_kernel=mk)
        self.d1 = DecoderBlock(32,  16,  mixer_kernel=mk)

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

# ✓ Đón input_size từ file __init__.py mà bạn vừa sửa
def build_model(num_classes=1, input_size=256):
    return ULiteModel_MCU(num_classes=num_classes, input_size=input_size)