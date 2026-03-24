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
        self.act = nn.GELU()
        
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
# 2. AXIAL-PFCU-DG BLOCK (Khối lai tạo đề xuất)
# ==============================================================================
class AxialDW(nn.Module):
    """Lõi cuộn 1D siêu nhẹ của U-Lite"""
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        # Dùng padding='same' để tự động tính toán viền với mọi dilation
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation, bias=False)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation, bias=False)

    def forward(self, x):
        return x + self.dw_h(x) + self.dw_w(x)

class DetailGuidance(nn.Module):
    """Nhánh Lọc chi tiết (Detail Guidance) thay thế Shortcut truyền thống"""
    def __init__(self, dim):
        super().__init__()
        # Sử dụng kernel 3x3 nhỏ gọn, dilation=1 để chỉ bắt lấy mép sắc nét
        self.dg_dw_h = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding='same', groups=dim, bias=False)
        self.dg_dw_w = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding='same', groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        # Tách lọc biên dọc và ngang
        edges = self.dg_dw_h(x) + self.dg_dw_w(x)
        return self.bn(x + edges) # Cộng lại gốc để không làm mất hoàn toàn hình dạng

class Axial_PFCU_DG(nn.Module):
    """Khối Parallel Factorized Convolutional Unit với Detail Guidance"""
    def __init__(self, dim, mixer_kernel=(5, 5)):
        super().__init__()
        
        # --- NHÁNH CHÍNH (PFCU - 3 Dilation song song: 1, 2, 5) ---
        self.branch_r1 = AxialDW(dim, mixer_kernel, dilation=1)
        self.branch_r2 = AxialDW(dim, mixer_kernel, dilation=2)
        self.branch_r5 = AxialDW(dim, mixer_kernel, dilation=5)
        
        # Pointwise trộn 3 nhánh
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        
        # --- NHÁNH PHỤ (Detail Guidance - Thay thế Identity Shortcut) ---
        self.dg_shortcut = DetailGuidance(dim)
        
        # --- MẮT THẦN (CoordAtt) ---
        self.coord_att = CoordAtt(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        # 1. Trích xuất ngữ cảnh qua 3 nhánh song song
        b1 = self.branch_r1(x)
        b2 = self.branch_r2(x)
        b5 = self.branch_r5(x)
        
        # Trộn ngữ cảnh đa quy mô
        fused_context = self.bn_fuse(self.pw_fuse(b1 + b2 + b5))
        
        # 2. Trích xuất chi tiết qua nhánh DG
        guided_details = self.dg_shortcut(x)
        
        # 3. Cộng gộp và chốt hạ bằng Coordinate Attention
        out = self.act(fused_context + guided_details)
        return self.coord_att(out)

# ==============================================================================
# 3. CÁC BLOCKS CƠ BẢN CỦA MẠNG
# ==============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        # SỬ DỤNG KHỐI ĐỀ XUẤT MỚI: Axial_PFCU_DG
        self.pfcu_dg   = Axial_PFCU_DG(in_c, mixer_kernel=mixer_kernel)
        self.bn        = nn.BatchNorm2d(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))

        if not self.same_channels:
            self.pw      = nn.Conv2d(in_c, conv_out, kernel_size=1)
            self.down_pw = nn.MaxPool2d((2, 2))

        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        # Đi qua khối PFCU_DG
        skip = self.bn(self.pfcu_dg(x))
        pool = self.down_pool(skip)

        if self.same_channels:
            x = self.act(self.bn2(pool))
        else:
            conv = self.down_pw(self.pw(skip))
            x    = self.act(self.bn2(torch.cat([pool, conv], dim=1)))

        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        gc = max(out_c // 4, 4)
        self.up      = nn.Upsample(scale_factor=2, mode='nearest')
        self.pw      = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn      = nn.BatchNorm2d(out_c)
        
        self.pw_down = nn.Conv2d(out_c, gc, kernel_size=1)
        # Sử dụng Axial_PFCU_DG ở Decoder để làm mượt ranh giới
        self.pfcu_dg = Axial_PFCU_DG(gc, mixer_kernel=mixer_kernel)
        self.pw_up   = nn.Conv2d(gc, out_c, kernel_size=1)
        self.act     = nn.GELU()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.bn(self.pw(x))
        x = self.act(self.pw_up(self.pfcu_dg(self.pw_down(x))) + x)
        return x

class BottleNeckBlock(nn.Module):
    """Giữ nguyên Bottleneck nhưng tích hợp Detail Guidance và CoordAtt"""
    def __init__(self, dim, max_dim=128):
        super().__init__()
        # Về bản chất, khối Bottleneck cũ đã có bóng dáng của PFCU
        # Ở đây ta sử dụng nguyên khối Axial_PFCU_DG để đồng bộ triết lý
        self.pfcu_dg = Axial_PFCU_DG(dim, mixer_kernel=(5, 5))

    def forward(self, x):
        return self.pfcu_dg(x)

# ==============================================================================
# 4. MẠNG CHÍNH (ULITE-PFCU-DG)
# ==============================================================================
class ULiteModel_PFCU_DG(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Kernel 5x5 là điểm ngọt giữa Receptive Field và FPS
        mk = (5, 5)

        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.e1 = EncoderBlock(16,  32,  mixer_kernel=mk)
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=mk)
        self.e3 = EncoderBlock(64,  128, mixer_kernel=mk)
        self.e4 = EncoderBlock(128, 256, mixer_kernel=mk)

        self.b4 = BottleNeckBlock(256, max_dim=128)

        self.d4 = DecoderBlock(256, 128, mixer_kernel=mk)
        self.d3 = DecoderBlock(128, 64,  mixer_kernel=mk)
        self.d2 = DecoderBlock(64,  32,  mixer_kernel=mk)
        self.d1 = DecoderBlock(32,  16,  mixer_kernel=mk)

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

        x = self.conv_out(x)
        return x

# ==============================================================================
# 5. HÀM TỰ ĐỘNG BUILD MODEL
# ==============================================================================
def build_model(num_classes=1):
    return ULiteModel_PFCU_DG(num_classes=num_classes)