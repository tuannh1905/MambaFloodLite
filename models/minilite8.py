import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# LƯU Ý TƯƠNG THÍCH ONNX / TORCH.FX:
# - Sử dụng nn.ReLU6 thay cho nn.PReLU (Tốt cho Quantization INT8, dễ fold vào BN).
# - Tránh các hàm reshape, permute, split động (gây lỗi khi export static graph).
# - Các lớp Conv2d đứng ngay trước BatchNorm2d đều được set bias=False để tối ưu.
# ==============================================================================

# ==============================================================================
# 1. ATTENTION MODULES (Đã tối ưu cho NPU/MCU)
# ==============================================================================
class SimplifiedCoordAtt(nn.Module):
    """
    CoordAtt tối ưu: Bỏ phép cat, split, permute phức tạp. 
    Xử lý nhánh H và nhánh W song song bằng Conv2d 1x1 thông thường.
    """
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        # Nhánh H
        self.conv_h1 = nn.Conv2d(inp, mip, kernel_size=1, bias=False)
        self.bn_h = nn.BatchNorm2d(mip)
        
        # Nhánh W
        self.conv_w1 = nn.Conv2d(inp, mip, kernel_size=1, bias=False)
        self.bn_w = nn.BatchNorm2d(mip)
        
        self.act = nn.ReLU6(inplace=True) # Thay PReLU bằng ReLU6
        
        self.conv_h2 = nn.Conv2d(mip, oup, kernel_size=1, bias=False)
        self.conv_w2 = nn.Conv2d(mip, oup, kernel_size=1, bias=False)
        
    def forward(self, x):
        identity = x
        
        # pooling
        x_h = self.pool_h(x) # (B, C, H, 1)
        x_w = self.pool_w(x) # (B, C, 1, W)
        
        # Giảm chiều
        y_h = self.act(self.bn_h(self.conv_h1(x_h)))
        y_w = self.act(self.bn_w(self.conv_w1(x_w)))
        
        # Khôi phục và tạo attention map
        a_h = self.conv_h2(y_h).sigmoid()
        a_w = self.conv_w2(y_w).sigmoid()
        
        return identity * a_h * a_w

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation cơ bản dùng Conv2d 1x1 thay cho Linear/Conv1d.
    Dễ dàng quantize và được hỗ trợ native trên mọi loại NPU.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(8, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
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
        
        # Thay thế ECA phức tạp bằng SE Block tĩnh
        self.channel_att = SEBlock(out_c)

    def forward(self, x_up, x_skip):
        x_up = self.reduce_up(x_up)
        x_skip = self.reduce_skip(x_skip)

        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)

        spatial_input = torch.cat([
            torch.mean(x_up, dim=1, keepdim=True), 
            torch.max(x_skip, dim=1, keepdim=True)[0]
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
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation, bias=False)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation, bias=False)

    def forward(self, x):
        return x + self.dw_h(x) + self.dw_w(x)

class DetailGuidance(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dg_dw_h = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding='same', groups=dim, bias=False)
        self.dg_dw_w = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding='same', groups=dim, bias=False)
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
        self.coord_att = SimplifiedCoordAtt(dim, dim)
        self.act = nn.ReLU6(inplace=True) # Thay PReLU bằng ReLU6

    def forward(self, x):
        b1 = self.branch_r1(x)
        b2 = self.branch_r2(x)
        b5 = self.branch_r5(x)
        
        fused_context = self.bn_fuse(self.pw_fuse(b1 + b2 + b5))
        guided_details = self.dg_shortcut(x)
        
        out = self.act(fused_context + guided_details)
        return self.coord_att(out)

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
        self.act = nn.ReLU6(inplace=True) # ReLU6

    def forward(self, x):
        skip = self.bn(self.pfcu_dg(x))
        pool = self.down_pool(skip)

        if self.same_channels:
            x = self.act(self.bn2(pool))
        else:
            conv = self.down_pw(self.pw(skip))
            x    = self.act(self.bn2(torch.cat([pool, conv], dim=1)))

        return x, skip

class BottleNeckBlock(nn.Module):
    def __init__(self, dim, max_dim=128):
        super().__init__()
        hid = min(dim // 4, max_dim // 4)
        
        # Giảm còn 2 mức SPP (1, 2) để bỏ redundant feature map ở kích thước nhỏ
        self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True))
        self.pool2 = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Conv2d(dim, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True))
        
        # Nhận x + 2 nhánh pooling
        self.spp_fuse = nn.Sequential(
            nn.Conv2d(dim + hid * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True)
        )
        
        self.axial_refine = AxialDW(dim, mixer_kernel=(5, 5), dilation=1)
        self.bn_refine = nn.BatchNorm2d(dim)
        self.coord_att = SimplifiedCoordAtt(dim, dim)

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.pool1(x), size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.pool2(x), size, mode='bilinear', align_corners=False)
        
        spp_fused = self.spp_fuse(torch.cat([x, x1, x2], dim=1))
        
        out = self.bn_refine(self.axial_refine(spp_fused))
        return self.coord_att(out + spp_fused)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        gc = max(out_c // 4, 4)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.uafm = TinyUAFM(in_c=in_c, skip_c=out_c, out_c=out_c)
        
        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        self.pfcu_dg = Axial_PFCU_DG(gc, mixer_kernel=mixer_kernel)
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        
        self.bn  = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True) # ReLU6

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

        self.b4 = BottleNeckBlock(256, max_dim=128)

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