import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. CHANNEL ATTENTION & UAFM (Giữ lại UAFM cho khâu dung hợp theo đúng chuẩn full model)
# ==============================================================================

class ECAModule(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)

class TinyUAFM(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.reduce_up = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.reduce_skip = nn.Conv2d(skip_c, out_c, 1, bias=False) if skip_c != out_c else nn.Identity()
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.eca = ECAModule(out_c)

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
        return self.eca(out)

# ==============================================================================
# 2. AXIAL-PFCU-DG (TRƯỜNG HỢP 3.1: KHÔNG CÓ COORD-ATT)
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

class Axial_PFCU_DG_NoAttn(nn.Module):
    """Lõi đặc trưng cấu trúc đầy đủ NHƯNG LOẠI BỎ COORD-ATT"""
    def __init__(self, dim, mixer_kernel=(5, 5)):
        super().__init__()
        self.branch_r1 = AxialDW(dim, mixer_kernel, dilation=1)
        self.branch_r2 = AxialDW(dim, mixer_kernel, dilation=2)
        self.branch_r5 = AxialDW(dim, mixer_kernel, dilation=5)
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        
        self.dg_shortcut = DetailGuidance(dim)
        self.act = nn.PReLU(dim) 

    def forward(self, x):
        b1 = self.branch_r1(x)
        b2 = self.branch_r2(x)
        b5 = self.branch_r5(x)
        
        fused_context = self.bn_fuse(self.pw_fuse(b1 + b2 + b5))
        guided_details = self.dg_shortcut(x)
        
        # Trả thẳng ra sau hàm kích hoạt, không qua CoordAtt
        return self.act(fused_context + guided_details)

# ==============================================================================
# 3. ENCODER, BOTTLENECK (SPP) & DECODER (ĐỀU KHÔNG COORD-ATT)
# ==============================================================================

class EncoderBlock_3_1(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.dw = Axial_PFCU_DG_NoAttn(in_c, mixer_kernel=mixer_kernel)
        self.bn = nn.BatchNorm2d(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw = nn.Conv2d(in_c, conv_out, kernel_size=1, bias=False)
            self.down_pw = nn.MaxPool2d((2, 2))

        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        skip = self.bn(self.dw(x))
        pool = self.down_pool(skip)

        if self.same_channels:
            x = self.act(self.bn2(pool))
        else:
            conv = self.down_pw(self.pw(skip))
            x = self.act(self.bn2(torch.cat([pool, conv], dim=1)))

        return x, skip


class BottleNeckBlock_SPP_NoAttn(nn.Module):
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
        self.act_out = nn.PReLU(dim)

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.pool1(x), size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.pool2(x), size, mode='bilinear', align_corners=False)
        x4 = F.interpolate(self.pool4(x), size, mode='bilinear', align_corners=False)
        
        spp_fused = self.spp_fuse(torch.cat([x, x1, x2, x4], dim=1))
        out = self.bn_refine(self.axial_refine(spp_fused))
        
        # Trả thẳng ra sau hàm kích hoạt, không qua CoordAtt
        return self.act_out(out + spp_fused)


class DecoderBlock_3_1(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        gc = max(out_c // 4, 4)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.uafm = TinyUAFM(in_c=in_c, skip_c=out_c, out_c=out_c)
        
        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        self.pfcu_dg = Axial_PFCU_DG_NoAttn(gc, mixer_kernel=mixer_kernel)
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        
        self.bn  = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.uafm(x, skip)
        
        # Vẫn dùng PFCU_DG_NoAttn để refine ranh giới, không có CoordAtt
        x = self.act(self.bn(self.pw_up(self.pfcu_dg(self.pw_down(x))) + x))
        return x


# ==============================================================================
# 4. MẠNG CHÍNH (ABLATION 3.1: FULL MODEL - NO COORD-ATT)
# ==============================================================================

class ULiteModel_Ablation_3_1(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (7, 7) # Giữ chuẩn 7x7 theo yêu cầu gốc

        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')

        self.e1 = EncoderBlock_3_1(16, 32, mixer_kernel=mk)
        self.e2 = EncoderBlock_3_1(32, 64, mixer_kernel=mk)
        self.e3 = EncoderBlock_3_1(64, 128, mixer_kernel=mk)
        self.e4 = EncoderBlock_3_1(128, 256, mixer_kernel=mk)
        self.e5 = EncoderBlock_3_1(256, 512, mixer_kernel=mk)

        # Bottleneck đầy đủ SPP, không CoordAtt
        self.b5 = BottleNeckBlock_SPP_NoAttn(512, max_dim=128)

        # Decoder đầy đủ UAFM, không CoordAtt
        self.d5 = DecoderBlock_3_1(512, 256, mixer_kernel=mk)
        self.d4 = DecoderBlock_3_1(256, 128, mixer_kernel=mk)
        self.d3 = DecoderBlock_3_1(128, 64, mixer_kernel=mk)
        self.d2 = DecoderBlock_3_1(64, 32, mixer_kernel=mk)
        self.d1 = DecoderBlock_3_1(32, 16, mixer_kernel=mk)

        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)

        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        x = self.b5(x)

        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)

        return self.conv_out(x)

def build_model(num_classes=1):
    return ULiteModel_Ablation_3_1(num_classes=num_classes)