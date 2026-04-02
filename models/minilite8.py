import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# LƯU Ý TƯƠNG THÍCH ONNX / TORCH.FX / MCU:
# - Thay padding='same' bằng tính toán explicit padding tĩnh.
# - Tránh các hàm reshape, permute, split động.
# - Dùng Abs-Mean thay cho Max-Channel.
# - Dùng SEBlock thay cho CoordAtt chứa AdaptiveAvgPool2d((None, 1)).
# - ✓ ĐÃ FIX: BottleNeck dùng Python int thuần cho scale_factor để ONNX trace chuẩn.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES (Đã tối ưu cho NPU/MCU)
# ==============================================================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation cơ bản dùng Conv2d 1x1 thay cho Linear/Conv1d.
    Dễ dàng quantize, fully static shape, hỗ trợ native trên mọi loại NPU.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(8, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Đảm bảo shape tĩnh (1, 1)
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

# ==============================================================================
# 2. TINY-UAFM
# ==============================================================================
class TinyUAFM(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.reduce_up = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.reduce_skip = nn.Conv2d(skip_c, out_c, 1, bias=False) if skip_c != out_c else nn.Identity()
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.channel_att = SEBlock(out_c)

    def forward(self, x_up, x_skip):
        x_up = self.reduce_up(x_up)
        x_skip = self.reduce_skip(x_skip)

        spatial_input = torch.cat([
            torch.mean(x_up, dim=1, keepdim=True), 
            torch.mean(x_skip.abs(), dim=1, keepdim=True) # Abs-mean thay cho Max
        ], dim=1)
        
        alpha = self.spatial_att(spatial_input)
        out = x_up * alpha + x_skip * (1 - alpha)
        
        return self.channel_att(out)

# ==============================================================================
# 3. AXIAL-PFCU-DG BLOCK
# ==============================================================================
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        
        pad_h = (h - 1) * dilation // 2
        pad_w = (w - 1) * dilation // 2
        
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
        edges = self.dg_dw_h(x) + self.dg_dw_w(x)
        return self.bn(x + edges)

class Axial_PFCU_DG(nn.Module):
    def __init__(self, dim, mixer_kernel=(5, 5)):
        super().__init__()
        self.branch_r1 = AxialDW(dim, mixer_kernel, dilation=1)
        self.branch_r2 = AxialDW(dim, mixer_kernel, dilation=2)
        self.branch_r5 = AxialDW(dim, mixer_kernel, dilation=5)
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        
        self.dg_shortcut = DetailGuidance(dim)
        self.se = SEBlock(dim)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        b1 = self.branch_r1(x)
        b2 = self.branch_r2(x)
        b5 = self.branch_r5(x)
        
        fused_context = self.bn_fuse(self.pw_fuse(b1 + b2 + b5))
        guided_details = self.dg_shortcut(x)
        
        out = self.act(fused_context + guided_details)
        return self.se(out)

# ==============================================================================
# 4. ENCODER, DECODER & BOTTLENECK
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
            x = self.act(self.bn2(self.down_pool(feat)))
            return x, feat
        else:
            feat_pw = self.pw(feat)
            skip = torch.cat([feat, feat_pw], dim=1) 
            
            pool_feat = self.down_pool(feat)
            pool_pw   = self.down_pw(feat_pw)
            x = self.act(self.bn2(torch.cat([pool_feat, pool_pw], dim=1)))

            return x, skip

class BottleNeckBlock(nn.Module):
    def __init__(self, dim, max_dim=128, input_size=16):
        super().__init__()
        hid = min(dim // 4, max_dim // 4)
        
        # ✓ ĐÃ FIX: Lưu dưới dạng Python Integer thuần túy để ONNX coi là hằng số biên dịch
        self._sf1 = input_size
        self._sf2 = input_size // 2
        
        self.pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self._sf1), # (H,W) -> (1,1) static
            nn.Conv2d(dim, hid, 1, bias=False),
            nn.BatchNorm2d(hid), 
            nn.ReLU6(inplace=True)
        )
        self.pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self._sf2), # (H,W) -> (2,2) static
            nn.Conv2d(dim, hid, 1, bias=False),
            nn.BatchNorm2d(hid), 
            nn.ReLU6(inplace=True)
        )
        
        self.spp_fuse = nn.Sequential(
            nn.Conv2d(dim + hid * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True)
        )
        
        self.axial_refine = AxialDW(dim, mixer_kernel=(5, 5), dilation=1)
        self.bn_refine = nn.BatchNorm2d(dim)
        self.se = SEBlock(dim)

    def forward(self, x):
        # ✓ ĐÃ FIX: Truyền thẳng thuộc tính int Python vào scale_factor
        x1 = F.interpolate(self.pool1(x), scale_factor=self._sf1, mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.pool2(x), scale_factor=self._sf2, mode='bilinear', align_corners=False)
        
        spp_fused = self.spp_fuse(torch.cat([x, x1, x2], dim=1))
        out = self.bn_refine(self.axial_refine(spp_fused))
        
        return self.se(out) + x

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        gc = max(out_c // 4, 4)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.uafm = TinyUAFM(in_c=in_c, skip_c=in_c, out_c=out_c)
        
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
# 5. MẠNG CHÍNH (LITEV8 - ONNX/MCU READY)
# ==============================================================================
class ULiteModel_MCU(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (5, 5)

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32,  mixer_kernel=mk)
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=mk)
        self.e3 = EncoderBlock(64,  128, mixer_kernel=mk)
        self.e4 = EncoderBlock(128, 256, mixer_kernel=mk)

        # Cứng hóa input_size=16 vào Bottleneck (cho ảnh gốc 256x256)
        self.b4 = BottleNeckBlock(256, max_dim=128, input_size=16)

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

def build_model(num_classes=1):
    return ULiteModel_MCU(num_classes=num_classes)