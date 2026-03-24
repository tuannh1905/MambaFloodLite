import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. TỌA ĐỘ CHÚ Ý (COORDINATE ATTENTION)
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

# ==============================================================================
# 2. KHỐI CẢI TIẾN: AXIAL-D-BLOCK VỚI LEARNABLE SKIP
# ==============================================================================
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation, bias=False)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation, bias=False)

    def forward(self, x):
        return x + self.dw_h(x) + self.dw_w(x)

class Axial_D_Block(nn.Module):
    def __init__(self, dim, mixer_kernel=(5, 5), dilation_context=4):
        super().__init__()
        self.half_c = dim // 2
        
        # --- NHÁNH MAIN (BỐI CẢNH): Chia đôi kênh, 2 mức Dilation ---
        self.local_dw = AxialDW(self.half_c, mixer_kernel, dilation=1)
        self.context_dw = AxialDW(dim - self.half_c, mixer_kernel, dilation=dilation_context)
        
        # --- NHÁNH DETAIL (CHI TIẾT): Learnable Shortcut mỏng ---
        # Dùng Axial 3x3 dilation=1 để chuyên bắt ranh giới
        self.detail_skip = AxialDW(dim, (3, 3), dilation=1)
        
        self.bn = nn.BatchNorm2d(dim)
        self.coord_att = CoordAtt(dim, dim)

    def forward(self, x):
        # 1. Trích xuất ranh giới (Detail Path)
        detail = self.detail_skip(x)
        
        # 2. Trích xuất bối cảnh (Main Path)
        x1, x2 = torch.split(x, [self.half_c, x.shape[1] - self.half_c], dim=1)
        x1 = self.local_dw(x1)
        x2 = self.context_dw(x2)
        main = torch.cat([x1, x2], dim=1)
        
        # 3. Gộp, Normalize và định vị tọa độ
        fused = self.bn(main + detail)
        
        # Trả về cả output chính lẫn bản đồ detail để giám sát
        return self.coord_att(fused), detail

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5), dilation_context=4):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        # Áp dụng khối Axial_D_Block thay cho Axial thông thường
        self.axial_d = Axial_D_Block(in_c, mixer_kernel, dilation_context)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw = nn.Conv2d(in_c, conv_out, kernel_size=1, bias=False)
            self.down_pw = nn.MaxPool2d((2, 2))

        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        fused, detail = self.axial_d(x)
        pool = self.down_pool(fused)

        if self.same_channels:
            out = self.act(self.bn2(pool))
        else:
            conv = self.down_pw(self.pw(fused))
            out = self.act(self.bn2(torch.cat([pool, conv], dim=1)))

        return out, detail

# ==============================================================================
# 3. CÁC MODULES PHỤ TRỢ (Decoder & Detail Head)
# ==============================================================================
class TinyUAFM(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x_up, x_skip):
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)
        s_input = torch.cat([torch.mean(x_up, 1, keepdim=True), torch.max(x_skip, 1, keepdim=True)[0]], 1)
        alpha = self.spatial_att(s_input)
        return x_up * alpha + x_skip * (1 - alpha)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        gc = max(out_c // 4, 4)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.uafm = TinyUAFM(out_c)
        
        self.pw_down = nn.Conv2d(out_c, gc, 1, bias=False)
        self.dw = AxialDW(gc, mixer_kernel, dilation=1)
        self.pw_up = nn.Conv2d(gc, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x, skip):
        x = self.reduce(self.up(x))
        x = self.uafm(x, skip)
        x = self.act(self.bn(self.pw_up(self.dw(self.pw_down(x))) + x))
        return x

class DetailHead(nn.Module):
    """Đầu ra phụ để áp dụng Loss_detail (Ép mạng học ranh giới)"""
    def __init__(self, in_c1, in_c2, num_classes=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c1 + in_c2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            nn.Conv2d(16, num_classes, kernel_size=1) # Trả về Raw Logits
        )
    def forward(self, detail1, detail2, target_size):
        d2_up = self.up(detail2)
        if d2_up.shape[2:] != detail1.shape[2:]:
            d2_up = F.interpolate(d2_up, size=detail1.shape[2:], mode='bilinear', align_corners=False)
        fuse = torch.cat([detail1, d2_up], dim=1)
        out = self.conv(fuse)
        return F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)

# ==============================================================================
# 4. MẠNG CHÍNH (ULITE DUAL-LOSS AXIAL-D)
# ==============================================================================
class ULite_Dual_AxialD(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (5, 5)

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        # Encoder sử dụng Dilation tăng dần ở nhánh bối cảnh
        self.e1 = EncoderBlock(16, 32, mixer_kernel=mk, dilation_context=2)
        self.e2 = EncoderBlock(32, 64, mixer_kernel=mk, dilation_context=4)
        self.e3 = EncoderBlock(64, 128, mixer_kernel=mk, dilation_context=8)
        self.e4 = EncoderBlock(128, 256, mixer_kernel=mk, dilation_context=16)

        # Đầu ra phụ chuyên dự đoán Ranh giới (Boundary/Edge Map)
        self.detail_head = DetailHead(in_c1=16, in_c2=32, num_classes=num_classes)

        self.d4 = DecoderBlock(256, 128, mixer_kernel=mk)
        self.d3 = DecoderBlock(128, 64, mixer_kernel=mk)
        self.d2 = DecoderBlock(64, 32, mixer_kernel=mk)
        self.d1 = DecoderBlock(32, 16, mixer_kernel=mk)

        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x = self.conv_in(x)

        x, detail1 = self.e1(x)
        x, detail2 = self.e2(x)
        x, detail3 = self.e3(x)
        x, detail4 = self.e4(x)

        # Giải mã dựa trên Detail Guidance từ Encoder
        d = self.d4(x, detail4)
        d = self.d3(d, detail3)
        d = self.d2(d, detail2)
        d = self.d1(d, detail1)

        seg_out = self.conv_out(d)

        # CHẾ ĐỘ TRAINING: Xuất ra cả 2 output để tính Dual-Loss
        if self.training:
            # Gộp detail từ 2 tầng nông nhất (nơi chứa ranh giới sắc nét nhất)
            detail_out = self.detail_head(detail1, detail2, size)
            return seg_out, detail_out
        
        # CHẾ ĐỘ INFERENCE: Chỉ xuất mask cuối cùng để tối đa FPS
        return seg_out

def build_model(num_classes=1):
    return ULite_Dual_AxialD(num_classes=num_classes)