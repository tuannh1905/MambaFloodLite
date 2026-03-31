import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. ATTENTION MODULES (Chỉ giữ lại CoordAtt)
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
# 2. TINY-UAFM & SUB-PIXEL UPSAMPLE (CẢI TIẾN MỚI)
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
        return out

class SubPixelUpsample(nn.Module):
    """
    CẢI TIẾN 1: Học Phóng To siêu nhẹ bằng Depthwise Expansion + PixelShuffle.
    Mỗi kênh học cách tự bung ra thành 4 điểm ảnh con, khôi phục mép nước cực sắc.
    """
    def __init__(self, in_c, scale_factor=2):
        super().__init__()
        out_channels = in_c * (scale_factor ** 2)
        # Depthwise Conv: groups=in_c giúp tiết kiệm 99% tham số so với Standard Conv
        self.expand_dw = nn.Conv2d(in_c, out_channels, kernel_size=3, padding=1, groups=in_c, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.bn = nn.BatchNorm2d(in_c)
        self.act = nn.PReLU(in_c)

    def forward(self, x):
        x = self.expand_dw(x)
        x = self.pixel_shuffle(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DeepStem(nn.Module):
    """
    CẢI TIẾN 2: Deep Stem (ResNet-D style).
    Trộn 3 kênh RGB bằng 3 lớp Conv liên tiếp để khai thác tối đa màu sắc.
    """
    def __init__(self, in_c=3, out_c=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(out_c)
        )

    def forward(self, x):
        return self.stem(x)

# ==============================================================================
# 3. LÕI AXIAL-GE-DG BLOCK
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
# 4. ENCODER, DECODER & DW CONTEXT EMBEDDING BOTTLENECK
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

class ContextEmbeddingBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_dw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(dim)
        )
        self.coord_att = CoordAtt(dim, dim)

    def forward(self, x):
        global_context = self.gap(x)
        global_context = self.conv1x1(global_context)
        out = x + global_context
        out = self.conv3x3_dw(out)
        return self.coord_att(out)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5), dilations=(1, 2, 5), inject_c=0):
        super().__init__()
        gc = max(out_c // 4, 4)
        
        # Đã thay đổi: Dùng PixelShuffle (SubPixelUpsample) thay vì Bilinear
        self.up = SubPixelUpsample(in_c, scale_factor=2)
        
        self.uafm = TinyUAFM(in_c=in_c, skip_c=out_c, out_c=out_c)
        
        self.pw_down = nn.Conv2d(out_c + inject_c, gc, kernel_size=1, bias=False)
        self.ge_dg = Axial_GE_DG(gc, mixer_kernel=mixer_kernel, dilations=dilations)
        self.pw_up = nn.Conv2d(gc, out_c, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x, skip, injected_img=None):
        x = self.up(x)
        x = self.uafm(x, skip)
        
        if injected_img is not None:
            x = torch.cat([x, injected_img], dim=1)
            
        x_residual = x[:, :self.pw_up.out_channels, :, :] if injected_img is not None else x
        x = self.act(self.bn(self.pw_up(self.ge_dg(self.pw_down(x))) + x_residual))
        return x

# ==============================================================================
# 5. MẠNG CHÍNH (VISIONARY: SUB-PIXEL UPSAMPLE + DEEP STEM + INJECTION)
# ==============================================================================
class ULiteModel_Visionary(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (5, 5)

        # Đã thay đổi: Sử dụng Deep Stem
        self.conv_in = DeepStem(in_c=3, out_c=16)

        # Encoder
        self.e1 = EncoderBlock(16,  32,  mixer_kernel=mk, dilations=(1, 2, 4))
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=mk, dilations=(1, 4, 8))
        self.e3 = EncoderBlock(64,  128, mixer_kernel=mk, dilations=(1, 6, 12))
        self.e4 = EncoderBlock(128, 192, mixer_kernel=mk, dilations=(1, 12, 24))

        # Bottleneck
        self.b4 = ContextEmbeddingBlock(192)

        # Trạm thu phát ảnh gốc
        self.img_proj_8 = nn.Conv2d(3, 8, kernel_size=1, bias=False)  
        self.img_proj_4 = nn.Conv2d(3, 8, kernel_size=1, bias=False)  
        self.img_proj_2 = nn.Conv2d(3, 8, kernel_size=1, bias=False)  
        
        self.avg_pool_8 = nn.AvgPool2d(8, stride=8)
        self.avg_pool_4 = nn.AvgPool2d(4, stride=4)
        self.avg_pool_2 = nn.AvgPool2d(2, stride=2)

        # Decoder
        self.d4 = DecoderBlock(192, 128, mixer_kernel=mk, dilations=(1, 12, 24), inject_c=8)
        self.d3 = DecoderBlock(128, 64,  mixer_kernel=mk, dilations=(1, 6, 12), inject_c=8)
        self.d2 = DecoderBlock(64,  32,  mixer_kernel=mk, dilations=(1, 4, 8), inject_c=8)
        self.d1 = DecoderBlock(32,  16,  mixer_kernel=mk, dilations=(1, 2, 4), inject_c=0)

        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        raw_img = x 
        
        x = self.conv_in(x)

        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)

        x = self.b4(x)
        
        # Bơm ảnh gốc vào Decoder
        img_8 = self.img_proj_8(self.avg_pool_8(raw_img)) 
        img_4 = self.img_proj_4(self.avg_pool_4(raw_img)) 
        img_2 = self.img_proj_2(self.avg_pool_2(raw_img)) 
        
        x = self.d4(x, skip4, injected_img=img_8)
        x = self.d3(x, skip3, injected_img=img_4)
        x = self.d2(x, skip2, injected_img=img_2)
        x = self.d1(x, skip1)
        
        return self.conv_out(x)

def build_model(num_classes=1):
    return ULiteModel_Visionary(num_classes=num_classes)