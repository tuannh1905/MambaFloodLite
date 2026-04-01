import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. ATTENTION MODULES (CoordAtt & ECA)
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

# ==============================================================================
# 2. TINY-UAFM & AUX HEAD
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

class AuxHead(nn.Module):
    """
    Nhánh giám sát phụ (Deep Supervision) siêu nhẹ.
    Dùng Depthwise Separable Conv để tiết kiệm tham số.
    """
    def __init__(self, in_chan, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            # DW lọc không gian
            nn.Conv2d(in_chan, in_chan, kernel_size=3, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.PReLU(in_chan),
            # PW hạ kênh về 64 cho nhẹ
            nn.Conv2d(in_chan, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            # Dropout chống Overfit cực mạnh
            nn.Dropout2d(0.1),
            # Phân loại
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x, target_size):
        x = self.head(x)
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

# ==============================================================================
# 3. AXIAL-PFCU-DG BLOCK (Khối đặc trưng lõi)
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
    def __init__(self, dim, mixer_kernel=(5, 5), dilations=(1, 2, 5)):
        super().__init__()
        self.branch_r1 = AxialDW(dim, mixer_kernel, dilation=dilations[0])
        self.branch_r2 = AxialDW(dim, mixer_kernel, dilation=dilations[1])
        self.branch_r5 = AxialDW(dim, mixer_kernel, dilation=dilations[2])
        
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
# 4. ENCODER, DECODER & BOTTLENECK (SPP-Axial)
# ==============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5), dilations=(1, 2, 5)):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.pfcu_dg   = Axial_PFCU_DG(in_c, mixer_kernel=mixer_kernel, dilations=dilations)
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
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5), dilations=(1, 2, 5)):
        super().__init__()
        gc = max(out_c // 4, 4)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.uafm = TinyUAFM(in_c=in_c, skip_c=out_c, out_c=out_c)
        
        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1, bias=False)
        self.pfcu_dg = Axial_PFCU_DG(gc, mixer_kernel=mixer_kernel, dilations=dilations)
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        
        self.bn  = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.uafm(x, skip)
        x = self.act(self.bn(self.pw_up(self.pfcu_dg(self.pw_down(x))) + x))
        return x

# ==============================================================================
# 5. MẠNG CHÍNH (ULITE-PFCU-UAFM VỚI AUX HEAD)
# ==============================================================================
class ULiteModel_PFCU_UAFM(nn.Module):
    def __init__(self, num_classes=1, aux_mode='train'):
        super().__init__()
        self.aux_mode = aux_mode
        mk = (5, 5)

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32,  mixer_kernel=mk, dilations=(1, 2, 4))
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=mk, dilations=(1, 4, 8))
        self.e3 = EncoderBlock(64,  128, mixer_kernel=mk, dilations=(1, 6, 12))
        self.e4 = EncoderBlock(128, 256, mixer_kernel=mk, dilations=(1, 8, 16))

        # Cắm Aux Head vào e3 (nhận 128 kênh) và e4 (nhận 256 kênh)
        if self.aux_mode == 'train':
            self.aux3 = AuxHead(128, num_classes)
            self.aux4 = AuxHead(256, num_classes)

        self.b4 = BottleNeckBlock(256, max_dim=128)

        self.d4 = DecoderBlock(256, 128, mixer_kernel=mk, dilations=(1, 8, 16))
        self.d3 = DecoderBlock(128, 64,  mixer_kernel=mk, dilations=(1, 6, 12))
        self.d2 = DecoderBlock(64,  32,  mixer_kernel=mk, dilations=(1, 4, 8))
        self.d1 = DecoderBlock(32,  16,  mixer_kernel=mk, dilations=(1, 2, 4))

        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        target_size = x.shape[2:] # Lấy kích thước ảnh gốc cho Aux Head
        
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

        logits_main = self.conv_out(x)
        
        # Nếu đang train và bật aux_mode -> Trả về 3 nhánh Loss
        if self.training and self.aux_mode == 'train':
            logits_aux3 = self.aux3(skip3, target_size)
            logits_aux4 = self.aux4(skip4, target_size)
            return logits_main, logits_aux3, logits_aux4
            
        # Nếu đang Validate/Inference -> Chỉ trả 1 nhánh chính
        return logits_main

def build_model(num_classes=1):
    # Khởi tạo kèm cờ aux_mode='train' để Trainer nhận được Tuple 3 Loss
    return ULiteModel_PFCU_UAFM(num_classes=num_classes, aux_mode='train')