import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. ATTENTION MODULES & DUAL-UAFM (NÂNG CẤP)
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

class DualUAFM(nn.Module):
    """
    NÂNG CẤP 2: Dual-UAFM (~5K params)
    Tích hợp Spatial Attention (Tìm vị trí) + Channel Attention (Nhận diện bản chất)
    """
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.reduce_up = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.reduce_skip = nn.Conv2d(skip_c, out_c, 1, bias=False) if skip_c != out_c else nn.Identity()
        
        # 1. Spatial Attention (Ranh giới ở đâu?)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
        # 2. Channel Attention (Đây là Nước hay Mái nhà?) - Ép cân kịch kim
        mip = max(8, out_c // 4)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_c, mip, kernel_size=1, bias=False),
            nn.PReLU(mip),
            nn.Conv2d(mip, out_c, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_up, x_skip):
        x_up = self.reduce_up(x_up)
        x_skip = self.reduce_skip(x_skip)

        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)

        # Trích xuất gợi ý không gian
        spatial_input = torch.cat([
            torch.mean(x_up, dim=1, keepdim=True), 
            torch.max(x_skip, dim=1, keepdim=True)[0]
        ], dim=1)
        
        # Dung hợp có cổng theo Không gian
        alpha_s = self.spatial_att(spatial_input)
        out = x_up * alpha_s + x_skip * (1 - alpha_s)
        
        # Chốt chặn lọc nhiễu theo Kênh
        alpha_c = self.channel_att(out)
        return out * alpha_c

# ==============================================================================
# 2. LÕI AXIAL-GE-DG BLOCK (DS-GATHER BẢO TOÀN TỪ VÒNG TRƯỚC)
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

class Axial_GE_DG(nn.Module):
    def __init__(self, dim, mixer_kernel=(5, 5), dilations=(1, 2, 5), exp_ratio=1):
        super().__init__()
        hid_dim = int(dim * exp_ratio)
        
        # DS-Gather giữ từ bản Recovery (Lọc thông thấp + Trộn kênh)
        self.gather = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(dim),
            nn.Conv2d(dim, hid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hid_dim),
            nn.PReLU(hid_dim)
        )
        
        self.branch_r1 = AxialDW(hid_dim, mixer_kernel, dilation=dilations[0])
        self.branch_r2 = AxialDW(hid_dim, mixer_kernel, dilation=dilations[1])
        self.branch_r5 = AxialDW(hid_dim, mixer_kernel, dilation=dilations[2])
        
        self.project = nn.Sequential(
            nn.Conv2d(hid_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        
        self.dg_shortcut = DetailGuidance(dim)
        self.coord_att = CoordAtt(dim, dim)
        self.act = nn.PReLU(dim)

    def forward(self, x):
        x_gather = self.gather(x)
        
        b1 = self.branch_r1(x_gather)
        b2 = self.branch_r2(x_gather)
        b5 = self.branch_r5(x_gather)
        
        fused_context = self.project(b1 + b2 + b5)
        
        guided_details = self.dg_shortcut(x)
        out = self.act(fused_context + guided_details)
        
        return self.coord_att(out)

# ==============================================================================
# 3. ENCODER, DECODER & MICRO-ASPP BOTTLENECK (NÂNG CẤP)
# ==============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5), dilations=(1, 2, 5)):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.ge_dg = Axial_GE_DG(in_c, mixer_kernel=mixer_kernel, dilations=dilations)
        self.bn = nn.BatchNorm2d(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw = nn.Conv2d(in_c, conv_out, kernel_size=1, bias=False)
            self.down_pw = nn.MaxPool2d((2, 2))

        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        skip = self.bn(self.ge_dg(x))
        pool = self.down_pool(skip)

        if self.same_channels:
            x = self.act(self.bn2(pool))
        else:
            conv = self.down_pw(self.pw(skip))
            x = self.act(self.bn2(torch.cat([pool, conv], dim=1)))

        return x, skip

class MicroASPPBlock(nn.Module):
    """
    NÂNG CẤP 1: Micro-ASPP (~40K Params)
    Dùng Depthwise Conv đa quy mô + Phép CỘNG (Add) để triệt tiêu tham số nén.
    """
    def __init__(self, dim):
        super().__init__()
        
        # Nhánh 1: Dilation = 1 (Chi tiết cục bộ)
        self.b1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(dim)
        )
        
        # Nhánh 2: Dilation = 6 (Bối cảnh vừa)
        self.b2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=6, dilation=6, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(dim)
        )
        
        # Nhánh 3: Dilation = 12 (Bối cảnh lớn)
        self.b3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=12, dilation=12, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(dim)
        )
        
        # Nhánh 4: Global Context (Bối cảnh vô cực)
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(dim)
        )
        
        # Nén sau khi CỘNG (Mixer)
        self.project = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(dim)
        )
        
        self.coord_att = CoordAtt(dim, dim)

    def forward(self, x):
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = F.interpolate(self.b4(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # CHIẾN THUẬT SIÊU NHẸ: CỘNG thay vì CAT
        out = feat1 + feat2 + feat3 + feat4
        
        out = self.project(out)
        return self.coord_att(out)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5), dilations=(1, 2, 5)):
        super().__init__()
        gc = max(out_c // 4, 4)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # THAY THẾ BẰNG DUAL-UAFM
        self.uafm = DualUAFM(in_c=in_c, skip_c=out_c, out_c=out_c)
        
        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        self.ge_dg = Axial_GE_DG(gc, mixer_kernel=mixer_kernel, dilations=dilations)
        self.pw_up = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.uafm(x, skip)
        x = self.act(self.bn(self.pw_up(self.ge_dg(self.pw_down(x))) + x))
        return x

# ==============================================================================
# 4. MẠNG CHÍNH (ULTIMATE: MICRO-ASPP + DUAL-UAFM)
# ==============================================================================
class ULiteModel_Ultimate(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (5, 5)

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        # Encoder: Giữ lại Dilation cực rộng ở e4
        self.e1 = EncoderBlock(16,  32,  mixer_kernel=mk, dilations=(1, 2, 4))
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=mk, dilations=(1, 4, 8))
        self.e3 = EncoderBlock(64,  128, mixer_kernel=mk, dilations=(1, 6, 12))
        self.e4 = EncoderBlock(128, 192, mixer_kernel=mk, dilations=(1, 12, 24))

        # Bottleneck: Nâng cấp lên Micro-ASPP đa quy mô
        self.b4 = MicroASPPBlock(192)

        # Decoder: Đối xứng, dùng Dual-UAFM
        self.d4 = DecoderBlock(192, 128, mixer_kernel=mk, dilations=(1, 12, 24))
        self.d3 = DecoderBlock(128, 64,  mixer_kernel=mk, dilations=(1, 6, 12))
        self.d2 = DecoderBlock(64,  32,  mixer_kernel=mk, dilations=(1, 4, 8))
        self.d1 = DecoderBlock(32,  16,  mixer_kernel=mk, dilations=(1, 2, 4))

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
    return ULiteModel_Ultimate(num_classes=num_classes)