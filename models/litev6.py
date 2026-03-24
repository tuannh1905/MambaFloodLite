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
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU6(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
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
# 2. TINY-UAFM & SPPM
# ==============================================================================
class TinyUAFM(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_up, x_skip):
        # Đảm bảo an toàn kích thước trước khi nối đặc trưng
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)
            
        # Tính toán alpha để lọc ranh giới
        s_input = torch.cat([torch.mean(x_up, 1, keepdim=True), torch.max(x_skip, 1, keepdim=True)[0]], dim=1)
        alpha = self.spatial_att(s_input)
        
        return x_up * alpha + x_skip * (1 - alpha)

class SPPM(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        hid = in_c // 4
        self.p1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_c, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True))
        self.p2 = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Conv2d(in_c, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True))
        self.p3 = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Conv2d(in_c, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True))
        self.conv = nn.Sequential(nn.Conv2d(in_c + hid * 3, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU6(inplace=True))

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.p1(x), size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.p2(x), size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(self.p3(x), size, mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, x1, x2, x3], dim=1))

# ==============================================================================
# 3. CORE MODULES (AXIAL DW & BLOCKS)
# ==============================================================================
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)
        
    def forward(self, x):
        return x + self.dw_h(x) + self.dw_w(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.same = (in_c == out_c)
        self.dw = AxialDW(in_c, mixer_kernel)
        self.bn = nn.BatchNorm2d(in_c)
        self.pool = nn.MaxPool2d(2)
        
        if not self.same:
            self.pw = nn.Conv2d(in_c, out_c - in_c, 1)
            self.p_pool = nn.MaxPool2d(2)
            
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)
        
    def forward(self, x):
        skip = self.bn(self.dw(x))
        p = self.pool(skip)
        
        if self.same:
            x = self.act(self.bn2(p))
        else:
            conv = self.p_pool(self.pw(skip))
            x = self.act(self.bn2(torch.cat([p, conv], dim=1)))
        return x, skip

class BottleNeckBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, 1)
        self.dw1 = AxialDW(gc, (3, 3), 1)
        self.dw2 = AxialDW(gc, (3, 3), 2)
        # Dilation 5 để phủ rộng, chống gridding effect
        self.dw3 = AxialDW(gc, (3, 3), 5) 
        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, 1)
        self.act = nn.ReLU6(inplace=True)
        
    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], dim=1)
        return self.act(self.pw2(self.bn(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.uafm = TinyUAFM(out_c)
        
        self.pw_d = nn.Conv2d(out_c, out_c // 4, 1)
        self.dw = AxialDW(out_c // 4, mixer_kernel)
        self.pw_u = nn.Conv2d(out_c // 4, out_c, 1)
        
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)
        
    def forward(self, x, skip):
        x = self.uafm(self.reduce(self.up(x)), skip)
        x = self.act(self.bn(self.pw_u(self.dw(self.pw_d(x))) + x))
        return x

# ==============================================================================
# 4. MẠNG CHÍNH (ULITE PLUS)
# ==============================================================================
class ULiteModelPlus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 16, 3, padding=1)
        
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        # Ép nhẹ kênh e4 để dành tham số cho module khác
        self.e4 = EncoderBlock(128, 160) 

        self.b4 = BottleNeckBlock(160)
        # Gom ngữ cảnh bối cảnh
        self.sppm = SPPM(160, 160) 
        # Định vị tọa độ
        self.coord = CoordAtt(160, 160)

        self.d4 = DecoderBlock(160, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)
        self.d1 = DecoderBlock(32, 16)
        
        # Raw Logits Output
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x = self.conv_in(x)
        
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        
        x = self.b4(x)
        x = self.sppm(x)
        x = self.coord(x)
        
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        
        return self.head(x)

# ==============================================================================
# 5. HÀM TỰ ĐỘNG BUILD MODEL CHO INIT.PY
# ==============================================================================
def build_model(num_classes=1):
    return ULiteModelPlus(num_classes=num_classes)