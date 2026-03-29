import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. ATTENTION MODULES
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
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
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

class TinyUAFM(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.reduce_up = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.reduce_skip = nn.Conv2d(skip_c, out_c, 1, bias=False) if skip_c != out_c else nn.Identity()
        self.spatial_att = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid())
        self.eca = ECAModule(out_c)

    def forward(self, x_up, x_skip):
        x_up = self.reduce_up(x_up)
        x_skip = self.reduce_skip(x_skip)
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)
        spatial_input = torch.cat([torch.mean(x_up, dim=1, keepdim=True), torch.max(x_skip, dim=1, keepdim=True)[0]], dim=1)
        alpha = self.spatial_att(spatial_input)
        out = x_up * alpha + x_skip * (1 - alpha)
        return self.eca(out)

# ==============================================================================
# 2. CONTINUOUS DNAS (LÕI TỰ HỌC DILATION BẰNG GRID SAMPLE)
# ==============================================================================
class ContinuousAxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, initial_r=1.0):
        super().__init__()
        self.kh, self.kw = mixer_kernel
        self.r = nn.Parameter(torch.tensor(float(initial_r)))
        
        self.weight_h = nn.Parameter(torch.Tensor(dim, 1, self.kh, 1))
        self.weight_w = nn.Parameter(torch.Tensor(dim, 1, 1, self.kw))
        nn.init.kaiming_normal_(self.weight_h, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_w, mode='fan_out', nonlinearity='relu')

    def _shift_and_multiply(self, x, weight, r, axis):
        b, c, h, w = x.shape
        device = x.device
        k = weight.shape[2] if axis == 'h' else weight.shape[3]
        pad = k // 2

        gy, gx = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                                torch.linspace(-1, 1, w, device=device), indexing='ij')
        base_grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        
        dy = 2.0 / (h - 1) if h > 1 else 0
        dx = 2.0 / (w - 1) if w > 1 else 0

        out = torch.zeros_like(x)
        for i, offset in enumerate(range(-pad, pad + 1)):
            if axis == 'h':
                w_slice = weight[:, 0, i, 0].view(1, c, 1, 1)
                shift_y, shift_x = offset * r * dy, 0.0
            else:
                w_slice = weight[:, 0, 0, i].view(1, c, 1, 1)
                shift_y, shift_x = 0.0, offset * r * dx

            grid = base_grid.clone()
            grid[..., 0] = grid[..., 0] + shift_x
            grid[..., 1] = grid[..., 1] + shift_y
            
            x_sampled = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            out = out + x_sampled * w_slice
            
        return out

    def forward(self, x):
        r_val = torch.clamp(self.r, min=1.0)
        out_h = self._shift_and_multiply(x, self.weight_h, r_val, 'h')
        out_w = self._shift_and_multiply(x, self.weight_w, r_val, 'w')
        return x + out_h + out_w

class DetailGuidance(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dg_dw_h = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding='same', groups=dim, bias=False)
        self.dg_dw_w = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding='same', groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
    def forward(self, x):
        edges = self.dg_dw_h(x) + self.dg_dw_w(x)
        return self.bn(x + edges)

# ==============================================================================
# 3. ENCODER BLOCK (DNAS)
# ==============================================================================
class Axial_PFCU_Continuous(nn.Module):
    def __init__(self, dim, mixer_kernel=(5, 5), init_dilations=(4.0, 8.0)):
        super().__init__()
        self.branch_m = ContinuousAxialDW(dim, mixer_kernel, initial_r=init_dilations[0])
        self.branch_l = ContinuousAxialDW(dim, mixer_kernel, initial_r=init_dilations[1])
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        
        self.dg_anchor = DetailGuidance(dim)
        self.coord_att = CoordAtt(dim, dim)
        self.act = nn.PReLU(dim)

    def forward(self, x):
        m = self.branch_m(x)
        l = self.branch_l(x)
        fused_context = self.bn_fuse(self.pw_fuse(m + l))
        anchor = self.dg_anchor(x)
        out = self.act(fused_context + anchor)
        return self.coord_att(out)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5), init_dilations=(4.0, 8.0)):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.pfcu_dnas = Axial_PFCU_Continuous(in_c, mixer_kernel=mixer_kernel, init_dilations=init_dilations)
        self.bn = nn.BatchNorm2d(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw = nn.Conv2d(in_c, conv_out, kernel_size=1, bias=False)
            self.down_pw = nn.MaxPool2d((2, 2))

        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        skip = self.bn(self.pfcu_dnas(x))
        pool = self.down_pool(skip)

        if self.same_channels:
            x = self.act(self.bn2(pool))
        else:
            conv = self.down_pw(self.pw(skip))
            x = self.act(self.bn2(torch.cat([pool, conv], dim=1)))
        return x, skip

# ==============================================================================
# 4. BOTTLENECK & REGSEG DECODER (128-128-8)
# ==============================================================================
class BottleNeckBlock(nn.Module):
    def __init__(self, dim, max_dim=128):
        super().__init__()
        hid = min(dim // 4, max_dim // 4)
        self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.PReLU(hid))
        self.pool2 = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Conv2d(dim, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.PReLU(hid))
        self.pool4 = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Conv2d(dim, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.PReLU(hid))
        self.spp_fuse = nn.Sequential(nn.Conv2d(dim + hid * 3, dim, kernel_size=1, bias=False), nn.BatchNorm2d(dim), nn.PReLU(dim))
        
        self.axial_refine = nn.Conv2d(dim, dim, kernel_size=(5, 5), padding='same', groups=dim, bias=False)
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

class RegSegDecoder(nn.Module):
    def __init__(self, c_16, c_8, c_4, num_classes):
        super().__init__()
        # Xử lý ngữ nghĩa sâu
        self.proj_16 = nn.Sequential(nn.Conv2d(c_16, 128, 1, bias=False), nn.BatchNorm2d(128), nn.PReLU(128))
        self.proj_8  = nn.Sequential(nn.Conv2d(c_8, 128, 1, bias=False), nn.BatchNorm2d(128), nn.PReLU(128))
        self.uafm_8  = TinyUAFM(in_c=128, skip_c=128, out_c=128)

        # Xử lý ranh giới nông
        self.proj_8_to_4 = nn.Sequential(nn.Conv2d(128, 8, 1, bias=False), nn.BatchNorm2d(8), nn.PReLU(8))
        self.proj_4      = nn.Sequential(nn.Conv2d(c_4, 8, 1, bias=False), nn.BatchNorm2d(8), nn.PReLU(8))
        self.uafm_4      = TinyUAFM(in_c=8, skip_c=8, out_c=8)

        # Scale up 4x về ảnh gốc
        self.classifier = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, f_16, f_8, f_4):
        # 1. Nâng 1/16 lên 1/8 và trộn
        x_16 = self.proj_16(f_16)
        x_16_up = F.interpolate(x_16, size=f_8.shape[2:], mode='bilinear', align_corners=False)
        x_8 = self.proj_8(f_8)
        fuse_8 = self.uafm_8(x_16_up, x_8)

        # 2. Nén 1/8 xuống 8 kênh, nâng lên 1/4 và trộn
        x_8_downproj = self.proj_8_to_4(fuse_8)
        x_8_up = F.interpolate(x_8_downproj, size=f_4.shape[2:], mode='bilinear', align_corners=False)
        x_4 = self.proj_4(f_4)
        fuse_4 = self.uafm_4(x_8_up, x_4)

        # 3. Scale về ảnh gốc
        out = F.interpolate(fuse_4, scale_factor=4, mode='bilinear', align_corners=False)
        return self.classifier(out)

# ==============================================================================
# 5. MẠNG CHÍNH (CONTINUOUS DNAS + 128-128-8) - ĐÃ FIX KÍCH THƯỚC SCALE
# ==============================================================================
class ULiteModel_Continuous_RegSeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32,  mixer_kernel=(5, 5), init_dilations=(2.0, 3.0))   
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=(5, 5), init_dilations=(3.0, 5.0))   
        self.e3 = EncoderBlock(64,  128, mixer_kernel=(7, 7), init_dilations=(4.0, 8.0))   
        self.e4 = EncoderBlock(128, 256, mixer_kernel=(7, 7), init_dilations=(10.0, 18.0)) 

        self.b5 = BottleNeckBlock(256, max_dim=128) 

        # ĐÃ SỬA: Khai báo đúng số lượng kênh của skip4 (128) và skip3 (64)
        self.decoder = RegSegDecoder(c_16=256, c_8=128, c_4=64, num_classes=num_classes)

    def forward(self, x):
        x = self.conv_in(x)

        x, skip1 = self.e1(x) # Tỉ lệ 1/1
        x, skip2 = self.e2(x) # Tỉ lệ 1/2
        x, skip3 = self.e3(x) # Tỉ lệ 1/4 (f_4, 64 channels)
        x, skip4 = self.e4(x) # Tỉ lệ 1/8 (f_8, 128 channels)

        f_16 = self.b5(x)     # Tỉ lệ 1/16 (256 channels)

        # ĐÃ SỬA: Truyền đúng skip4 (cho f_8) và skip3 (cho f_4) vào Decoder
        out = self.decoder(f_16, skip4, skip3)
        
        return out

def build_model(num_classes=1):
    return ULiteModel_Continuous_RegSeg(num_classes=num_classes)