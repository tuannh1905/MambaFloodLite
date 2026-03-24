import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. MODULE TINY-UAFM & ECA (LỌC NHIỄU BẰNG ATTENTION)
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
        # Căn chỉnh số kênh nếu tầng Up và tầng Skip lệch nhau
        self.reduce_up = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.reduce_skip = nn.Conv2d(skip_c, out_c, 1, bias=False) if skip_c != out_c else nn.Identity()
        
        # Nhánh Spatial Attention siêu nhẹ
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.eca = ECAModule(out_c)

    def forward(self, x_up, x_skip):
        x_up = self.reduce_up(x_up)
        x_skip = self.reduce_skip(x_skip)

        # Căn chỉnh kích thước không gian bằng nội suy (An toàn cho shape)
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)

        # Tính Spatial weight từ đặc trưng toàn cục (Up) và chi tiết (Skip)
        spatial_input = torch.cat([
            torch.mean(x_up, dim=1, keepdim=True), 
            torch.max(x_skip, dim=1, keepdim=True)[0]
        ], dim=1)
        att_v = self.spatial_att(spatial_input)
        
        # Hòa trộn thông minh (Loại bỏ torch.cat thô sơ)
        out = x_up * att_v + x_skip * (1 - att_v)
        return self.eca(out)

# ==============================================================================
# 2. MODULES CỐT LÕI (AXIAL DW & BLOCKS THU PHÓNG)
# ==============================================================================
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        # Padding tự động theo công thức để giữ nguyên shape với mọi dilation
        pad_h = (h + (h - 1) * (dilation - 1)) // 2
        pad_w = (w + (w - 1) * (dilation - 1)) // 2
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding=(pad_h, 0), groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding=(0, pad_w), groups=dim, dilation=dilation)

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
# 3. MẠNG CHÍNH ĐÃ CẢI TIẾN (ERF-AXIAL-UAFM)
# ==============================================================================
class ERF_AxialNet_UAFM(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (5, 5) 

        # --- ENCODER ---
        # Tầng nông (1/2 độ phân giải) -> Trích xuất Skip 1
        self.initial_block = DownsamplerBlock(3, 16) 
        
        # Tầng trung (1/4 độ phân giải) -> Trích xuất Skip 2
        self.down_64 = DownsamplerBlock(16, 64)
        self.enc_64 = nn.Sequential(
            *[Axial_NonBottleneck_1D(64, 0.03, dilated=1, mixer_kernel=mk) for _ in range(5)]
        )

        # Tầng sâu (1/8 độ phân giải)
        self.down_128 = DownsamplerBlock(64, 128)
        
        # CẢI TIẾN 2: Dense Dilation [2, 4, 8, 2, 4, 8, 2, 4] triệt tiêu Gridding Effect
        dilations = [2, 4, 8, 2, 4, 8, 2, 4]
        self.enc_128 = nn.Sequential(
            *[Axial_NonBottleneck_1D(128, 0.3, dilated=d, mixer_kernel=mk) for d in dilations]
        )

        # --- DECODER (CHIẾN LƯỢC FLD) ---
        # Bước lên 1/4 độ phân giải
        self.up_64 = UpsamplerBlock(128, 64)
        self.uafm_64 = TinyUAFM(in_c=64, skip_c=64, out_c=64)
        # Chỉ dùng ĐÚNG 1 khối Axial làm Refinement (nhẹ hóa Decoder)
        self.dec_64 = Axial_NonBottleneck_1D(64, 0, dilated=1, mixer_kernel=mk)

        # Bước lên 1/2 độ phân giải
        self.up_16 = UpsamplerBlock(64, 16)
        self.uafm_16 = TinyUAFM(in_c=16, skip_c=16, out_c=16)
        # Chỉ dùng ĐÚNG 1 khối Axial làm Refinement
        self.dec_16 = Axial_NonBottleneck_1D(16, 0, dilated=1, mixer_kernel=mk)

        # Trả về 1/1 độ phân giải với Logits gốc
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        # 1. ENCODER (Lưu lại Skip Connections)
        skip1 = self.initial_block(input)           # [B, 16, H/2, W/2]
        
        x = self.down_64(skip1)
        skip2 = self.enc_64(x)                      # [B, 64, H/4, W/4]
        
        x = self.down_128(skip2)
        x = self.enc_128(x)                         # [B, 128, H/8, W/8]

        # 2. DECODER (Hòa trộn UAFM + FLD)
        x = self.up_64(x)                           # Lên [B, 64, H/4, W/4]
        x = self.uafm_64(x, skip2)                  # Attention Fusion với Skip 2
        x = self.dec_64(x)                          # Refine ranh giới

        x = self.up_16(x)                           # Lên [B, 16, H/2, W/2]
        x = self.uafm_16(x, skip1)                  # Attention Fusion với Skip 1
        x = self.dec_16(x)                          # Refine ranh giới

        # 3. OUTPUT
        return self.output_conv(x)                  # [B, 1, H, W]

# ==============================================================================
# 4. HÀM TỰ ĐỘNG BUILD MODEL CHO INIT.PY
# ==============================================================================
def build_model(num_classes=1):
    return ERF_AxialNet_UAFM(num_classes=num_classes)