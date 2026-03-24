import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. MODULE TỐI ƯU KÊNH (ECA) & ATTENTION (TinyUAFM)
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
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class TinyUAFM(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        # Nhánh Spatial Attention siêu nhẹ (Chỉ 2 kênh đầu vào: mean_up và max_skip)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        # Nhánh Channel Attention bằng ECA
        self.eca = ECAModule(out_c)

    def forward(self, x_up, x_skip):
        # Tính Spatial weight từ cả 2 nguồn (Global view từ Up, Local Boundary từ Skip)
        spatial_input = torch.cat([
            torch.mean(x_up, dim=1, keepdim=True), 
            torch.max(x_skip, dim=1, keepdim=True)[0]
        ], dim=1)
        
        att_v = self.spatial_att(spatial_input)
        
        # Hòa trộn có trọng số (Loại bỏ torch.cat thô sơ)
        out = x_up * att_v + x_skip * (1 - att_v)
        
        # Tinh chỉnh lần cuối bằng ECA
        return self.eca(out)

# ==============================================================================
# 2. SIMPLE PYRAMID POOLING MODULE (SPPM) - Bắt Global Context
# ==============================================================================
class SPPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hid_channels = max(in_channels // 4, 16)
        
        # 3 mức Pooling để gom thông tin bối cảnh diện rộng
        self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, hid_channels, 1, bias=False), nn.BatchNorm2d(hid_channels), nn.GELU())
        self.pool2 = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Conv2d(in_channels, hid_channels, 1, bias=False), nn.BatchNorm2d(hid_channels), nn.GELU())
        self.pool3 = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Conv2d(in_channels, hid_channels, 1, bias=False), nn.BatchNorm2d(hid_channels), nn.GELU())

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + hid_channels * 3, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.pool1(x), size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.pool2(x), size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(self.pool3(x), size, mode='bilinear', align_corners=False)
        
        out = torch.cat([x, x1, x2, x3], dim=1)
        return self.conv(out)

# ==============================================================================
# 3. CORE MODULES (Axial DW, Encoder, Decoder, Bottleneck)
# ==============================================================================
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        pad_h = (h + (h - 1) * (dilation - 1)) // 2
        pad_w = (w + (w - 1) * (dilation - 1)) // 2
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding=(pad_h, 0), groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding=(0, pad_w), groups=dim, dilation=dilation)

    def forward(self, x):
        return x + self.dw_h(x) + self.dw_w(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.dw        = AxialDW(in_c, mixer_kernel=mixer_kernel)
        self.bn        = nn.BatchNorm2d(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw      = nn.Conv2d(in_c, conv_out, kernel_size=1)
            self.down_pw = nn.MaxPool2d((2, 2))

        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        pool = self.down_pool(skip)

        if self.same_channels:
            x = self.act(self.bn2(pool))
        else:
            conv = self.down_pw(self.pw(skip))
            x    = self.act(self.bn2(torch.cat([pool, conv], dim=1)))

        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        gc = max(out_c // 4, 4)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Căn chỉnh số kênh từ tầng dưới (in_c) về bằng tầng skip (out_c) trước khi trộn
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        ) if in_c != out_c else nn.Identity()

        # Áp dụng TinyUAFM thay thế cho torch.cat thô sơ
        self.uafm = TinyUAFM(out_c) 
        
        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1)
        self.dw      = AxialDW(gc, mixer_kernel=mixer_kernel)
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1)
        self.act     = nn.GELU()

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        x = self.uafm(x, skip) # Hòa trộn thông minh
        x = self.act(self.pw_up(self.dw(self.pw_down(x))) + x)
        return x

class BottleNeckBlock(nn.Module):
    def __init__(self, dim, max_dim=128):
        super().__init__()
        gc = min(dim // 4, max_dim // 4)
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)
        self.bn  = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], dim=1)
        x = self.act(self.pw2(self.bn(x)))
        return x

# ==============================================================================
# 4. RÁP THÀNH MẠNG HOÀN CHỈNH (Tích hợp SwiftNet Interleaved Fusion)
# ==============================================================================
class ULite_Swift(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32,  mixer_kernel=(7, 7))
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=(7, 7))
        self.e3 = EncoderBlock(64,  128, mixer_kernel=(7, 7))
        self.e4 = EncoderBlock(128, 256, mixer_kernel=(7, 7))

        self.b4 = BottleNeckBlock(256, max_dim=128)

        # SPPM được gắn ngay sau khối BottleNeck lõi
        self.sppm = SPPM(256, 256)

        self.d4 = DecoderBlock(256, 128, mixer_kernel=(7, 7))
        self.d3 = DecoderBlock(128, 64,  mixer_kernel=(7, 7))
        self.d2 = DecoderBlock(64,  32,  mixer_kernel=(7, 7))
        self.d1 = DecoderBlock(32,  16,  mixer_kernel=(7, 7))

        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def _shared_encoder_forward(self, x):
        """Khối dùng chung trọng số cho 2 quy mô ảnh"""
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x = self.b4(x)
        x = self.sppm(x)
        return x, skip1, skip2, skip3, skip4

    def forward(self, x):
        # 1. Scale 0.5x (Để ép mô hình học đặc trưng bất biến quy mô - Scale Invariant)
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        out_half, _, _, _, _ = self._shared_encoder_forward(x_half) # Bỏ qua skip connections của scale 0.5

        # 2. Scale 1.0x (Ảnh gốc)
        out_full, skip1, skip2, skip3, skip4 = self._shared_encoder_forward(x)

        # 3. Interleaved Pyramid Fusion: Phục hồi kích thước và hòa trộn Global Context ở tầng sâu nhất
        out_half_up = F.interpolate(out_half, size=out_full.shape[2:], mode='bilinear', align_corners=False)
        fused_context = out_full + out_half_up

        # 4. Giải mã (Decoding) với bối cảnh đã hòa trộn và TinyUAFM lọc nhiễu
        d = self.d4(fused_context, skip4)
        d = self.d3(d, skip3)
        d = self.d2(d, skip2)
        d = self.d1(d, skip1)

        return self.conv_out(d)

# ==============================================================================
# 5. HÀM TỰ ĐỘNG BUILD MODEL CHO INIT.PY
# ==============================================================================
def build_model(num_classes=1):
    return ULite_Swift(num_classes=num_classes)