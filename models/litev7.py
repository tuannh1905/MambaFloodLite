import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. ATTENTION MODULES (Chỉ giữ lại CoordAtt cho không gian)
# ==============================================================================
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.PReLU(mip) 
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return identity * a_w * a_h

# ĐÃ XÓA BỎ ECAMODULE ĐỂ TRÁNH ĐỤNG ĐỘ ATTENTION (ATTENTION COLLISION)

# ==============================================================================
# 2. SUB-PIXEL UPSAMPLE & TINY-UAFM (Đã tối giản)
# ==============================================================================
class SubPixelUpsample(nn.Module):
    """
    Học cách phóng to ảnh (x2) thay vì dùng Bilinear.
    Tăng kênh lên 4 lần rồi dùng PixelShuffle(2) để xếp thành pixel không gian.
    """
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * 4, kernel_size=1, bias=False)
        self.ps = nn.PixelShuffle(2)
        self.bn = nn.BatchNorm2d(in_c)
        self.act = nn.PReLU(in_c)
        
    def forward(self, x):
        return self.act(self.bn(self.ps(self.conv(x))))

class TinyUAFM(nn.Module):
    """
    Chỉ tập trung làm Spatial Gated Fusion (Tạo mặt nạ chọn vùng Pixel).
    Đã loại bỏ ECA để gradient chạy mượt hơn.
    """
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.reduce_up = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.reduce_skip = nn.Conv2d(skip_c, out_c, 1, bias=False) if skip_c != out_c else nn.Identity()
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_up, x_skip):
        x_up = self.reduce_up(x_up)
        x_skip = self.reduce_skip(x_skip)

        # Bilinear ở đây chỉ dùng để an toàn hóa (nếu có lỗi làm tròn chênh lệch 1 pixel)
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)

        spatial_input = torch.cat([
            torch.mean(x_up, dim=1, keepdim=True), 
            torch.max(x_skip, dim=1, keepdim=True)[0]
        ], dim=1)
        
        alpha = self.spatial_att(spatial_input)
        
        # Chỉ hòa trộn Spatial, trả thẳng ra ngoài
        return x_up * alpha + x_skip * (1 - alpha)

# ==============================================================================
# 3. LÕI AXIAL PFCU-DG BLOCK
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
        self.coord_att = CoordAtt(dim, dim)
        self.act = nn.PReLU(dim) 

    def forward(self, x):
        b1 = self.branch_r1(x)
        b2 = self.branch_r2(x)
        b5 = self.branch_r5(x)
        
        fused_context = self.bn_fuse(self.pw_fuse(b1 + b2 + b5))
        guided_details = self.dg_shortcut(x)
        
        out = self.act(fused_context + guided_details)
        return self.coord_att(out)

# ==============================================================================
# 4. MODULES KIẾN TRÚC MẠNG (STEM, ENCODER, BOTTLENECK, DECODER)
# ==============================================================================
class StemBlock(nn.Module):
    """Lọc nhiễu và Ép khung hình: Hạ mẫu xuống 1/2 ngay từ cổng vào"""
    def __init__(self, in_c=3, out_c=16):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

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
        self.act = nn.PReLU(out_c)

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
        
        self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.PReLU(hid))
        self.pool2 = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Conv2d(dim, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.PReLU(hid))
        self.pool4 = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Conv2d(dim, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.PReLU(hid))
        
        self.spp_fuse = nn.Sequential(
            nn.Conv2d(dim + hid * 3, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(dim)
        )
        
        self.axial_refine = AxialDW(dim, mixer_kernel=(5, 5), dilation=1)
        self.bn_refine = nn.BatchNorm2d(dim)
        self.coord_att = CoordAtt(dim, dim)

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.pool1(x), size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.pool2(x), size, mode='bilinear', align_corners=False)
        x4 = F.interpolate(self.pool4(x), size, mode='bilinear', align_corners=False)
        
        spp_fused = self.spp_fuse(torch.cat([x, x1, x2, x4], dim=1))
        
        out = self.bn_refine(self.axial_refine(spp_fused))
        return self.coord_att(out + spp_fused)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        gc = max(out_c // 4, 4)
        
        # SỬ DỤNG PIXEL-SHUFFLE THAY VÌ BILINEAR MÙ
        self.up = SubPixelUpsample(in_c)
        
        self.uafm = TinyUAFM(in_c=in_c, skip_c=out_c, out_c=out_c)
        
        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        self.pfcu_dg = Axial_PFCU_DG(gc, mixer_kernel=mixer_kernel)
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        
        self.bn  = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.uafm(x, skip)
        x = self.act(self.bn(self.pw_up(self.pfcu_dg(self.pw_down(x))) + x))
        return x

# ==============================================================================
# 5. MẠNG CHÍNH (LITE V8 + STEM + PIXELSHUFFLE)
# ==============================================================================
class ULiteModel_PFCU_UAFM_V8_StemPS(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (5, 5)

        # 1. Khởi tạo bằng StemBlock (1/1 -> 1/2)
        self.stem = StemBlock(in_c=3, out_c=16)

        # 2. Encoder chạy trên các độ phân giải đã được nén
        self.e1 = EncoderBlock(16,  32,  mixer_kernel=mk) # Input 1/2 -> Pool 1/4
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=mk) # Input 1/4 -> Pool 1/8
        self.e3 = EncoderBlock(64,  128, mixer_kernel=mk) # Input 1/8 -> Pool 1/16
        self.e4 = EncoderBlock(128, 256, mixer_kernel=mk) # Input 1/16 -> Pool 1/32

        # 3. Bottleneck ở tận đáy 1/32 (FOV cực khủng)
        self.b4 = BottleNeckBlock(256, max_dim=128)

        # 4. Decoder với PixelShuffle
        self.d4 = DecoderBlock(256, 128, mixer_kernel=mk) # Lên 1/16
        self.d3 = DecoderBlock(128, 64,  mixer_kernel=mk) # Lên 1/8
        self.d2 = DecoderBlock(64,  32,  mixer_kernel=mk) # Lên 1/4
        self.d1 = DecoderBlock(32,  16,  mixer_kernel=mk) # Lên 1/2

        # 5. Khôi phục lại ảnh 1/1 bằng PixelShuffle lần cuối
        self.final_up = SubPixelUpsample(16)
        
        # 6. Đầu ra
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)         # Hạ mẫu 1/2

        x, skip1 = self.e1(x)    # skip1: 1/2
        x, skip2 = self.e2(x)    # skip2: 1/4
        x, skip3 = self.e3(x)    # skip3: 1/8
        x, skip4 = self.e4(x)    # skip4: 1/16

        x = self.b4(x)           # Bottleneck ở 1/32

        x = self.d4(x, skip4)    # Giải mã về 1/16
        x = self.d3(x, skip3)    # Giải mã về 1/8
        x = self.d2(x, skip2)    # Giải mã về 1/4
        x = self.d1(x, skip1)    # Giải mã về 1/2

        x = self.final_up(x)     # Phóng to trả lại 1/1 sắc nét
        return self.conv_out(x)

def build_model(num_classes=1):
    return ULiteModel_PFCU_UAFM_V8_StemPS(num_classes=num_classes)