import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. TỌA ĐỘ CHÚ Ý (COORDINATE ATTENTION - Khắc phục lỗi nhầm địa hình)
# ==============================================================================
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # Pooling tách biệt theo 2 trục X (Width) và Y (Height)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True) if hasattr(nn, 'Hardswish') else nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # Ép thông tin không gian vào 2 vector 1D
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Trộn thông tin tọa độ
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 

        # Tách lại ra 2 trục và tính Attention Map
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        # Áp dụng tọa độ chú ý lên ảnh gốc
        out = identity * a_w * a_h
        return out

class CoordUAFM(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.reduce_up = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.reduce_skip = nn.Conv2d(skip_c, out_c, 1, bias=False) if skip_c != out_c else nn.Identity()
        
        # Dùng CoordAtt để đánh giá tầm quan trọng của các tọa độ khi gộp đặc trưng
        self.coord_att = CoordAtt(out_c, out_c)

    def forward(self, x_up, x_skip):
        x_up = self.reduce_up(x_up)
        x_skip = self.reduce_skip(x_skip)

        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)

        # Cộng dồn đặc trưng rồi dùng CoordAtt làm màng lọc không gian/kênh
        out = x_up + x_skip
        return self.coord_att(out)

# ==============================================================================
# 2. SPPM (MẮT THẦN - Thu thập bối cảnh toàn cục)
# ==============================================================================
class SPPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hid_channels = max(in_channels // 4, 16) 
        
        # 3 tầm nhìn: Rộng (1x1), Vừa (2x2), Gần (4x4)
        self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, hid_channels, 1, bias=False), nn.BatchNorm2d(hid_channels), nn.ReLU(inplace=True))
        self.pool2 = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Conv2d(in_channels, hid_channels, 1, bias=False), nn.BatchNorm2d(hid_channels), nn.ReLU(inplace=True))
        self.pool3 = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Conv2d(in_channels, hid_channels, 1, bias=False), nn.BatchNorm2d(hid_channels), nn.ReLU(inplace=True))

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + hid_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.pool1(x), size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.pool2(x), size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(self.pool3(x), size, mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, x1, x2, x3], dim=1))

# ==============================================================================
# 3. CORE MODULES (AXIAL DW & BLOCKS)
# ==============================================================================
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        pad_h = (h + (h - 1) * (dilation - 1)) // 2
        pad_w = (w + (w - 1) * (dilation - 1)) // 2
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding=(pad_h, 0), groups=dim, dilation=dilation, bias=False)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding=(0, pad_w), groups=dim, dilation=dilation, bias=False)

    def forward(self, x):
        return x + self.dw_h(x) + self.dw_w(x)

class Axial_NonBottleneck_1D(nn.Module):
    def __init__(self, chann, dropprob, dilated, mixer_kernel=(5, 5)):
        super().__init__()
        self.pw1 = nn.Conv2d(chann, chann, kernel_size=1, bias=False)
        self.axial1 = AxialDW(chann, mixer_kernel=mixer_kernel, dilation=1)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.pw2 = nn.Conv2d(chann, chann, kernel_size=1, bias=False)
        self.axial2 = AxialDW(chann, mixer_kernel=mixer_kernel, dilation=dilated)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        out = F.relu(self.bn1(self.axial1(self.pw1(input))))
        out = self.bn2(self.axial2(self.pw2(out)))
        if self.dropout.p != 0:
            out = self.dropout(out)
        return F.relu(out + input)

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        return F.relu(self.bn(output))

class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        return F.relu(self.bn(self.conv(input)))

# ==============================================================================
# 4. MẠNG CHÍNH (ERF-AXIAL-COORD-SPPM)
# ==============================================================================
class ERF_AxialNet_CoordSPPM(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (5, 5) 

        # --- ENCODER ---
        self.initial_block = DownsamplerBlock(3, 16) 
        
        self.down_64 = DownsamplerBlock(16, 64)
        self.enc_64 = nn.Sequential(
            *[Axial_NonBottleneck_1D(64, 0.03, dilated=1, mixer_kernel=mk) for _ in range(5)]
        )

        self.down_128 = DownsamplerBlock(64, 128)
        
        # Dense Dilation triệt tiêu Gridding Effect
        dilations = [2, 4, 8, 2, 4, 8, 2, 4]
        self.enc_128 = nn.Sequential(
            *[Axial_NonBottleneck_1D(128, 0.3, dilated=d, mixer_kernel=mk) for d in dilations]
        )

        # --- SPPM (MẮT THẦN) ---
        # Đặt ngay sau khối đặc trưng sâu nhất để nhìn toàn cảnh trước khi giải mã
        self.sppm = SPPM(128, 128)

        # --- DECODER ---
        self.up_64 = UpsamplerBlock(128, 64)
        # Sử dụng CoordUAFM để lọc nhiễu tọa độ
        self.uafm_64 = CoordUAFM(in_c=64, skip_c=64, out_c=64)
        self.dec_64 = Axial_NonBottleneck_1D(64, 0, dilated=1, mixer_kernel=mk)

        self.up_16 = UpsamplerBlock(64, 16)
        self.uafm_16 = CoordUAFM(in_c=16, skip_c=16, out_c=16)
        self.dec_16 = Axial_NonBottleneck_1D(16, 0, dilated=1, mixer_kernel=mk)

        # Raw Logits
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        # 1. ENCODER
        skip1 = self.initial_block(input)           
        
        x = self.down_64(skip1)
        skip2 = self.enc_64(x)                      
        
        x = self.down_128(skip2)
        x = self.enc_128(x)                         

        # 2. SPPM (Global Context)
        x = self.sppm(x)

        # 3. DECODER
        x = self.up_64(x)                           
        x = self.uafm_64(x, skip2)                  
        x = self.dec_64(x)                          

        x = self.up_16(x)                           
        x = self.uafm_16(x, skip1)                  
        x = self.dec_16(x)                          

        # 4. OUTPUT
        return self.output_conv(x)                  

# ==============================================================================
# 5. HÀM TỰ ĐỘNG BUILD MODEL CHO INIT.PY
# ==============================================================================
def build_model(num_classes=1):
    return ERF_AxialNet_CoordSPPM(num_classes=num_classes)