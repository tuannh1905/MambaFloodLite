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
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return identity * a_w * a_h

# ==============================================================================
# 2. LÕI AXIAL-PFCU-DG BLOCK (Cho nhánh Semantic)
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
    def __init__(self, dim, mixer_kernel=(5, 5)):
        super().__init__()
        self.branch_r1 = AxialDW(dim, mixer_kernel, dilation=1)
        self.branch_r2 = AxialDW(dim, mixer_kernel, dilation=2)
        self.branch_r5 = AxialDW(dim, mixer_kernel, dilation=5)
        
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

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c

        self.pfcu_dg   = Axial_PFCU_DG(in_c, mixer_kernel=mixer_kernel)
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
        return x

class AdditiveBottleNeck(nn.Module):
    """Giữ nguyên Bottleneck nhưng dùng cơ chế Cộng (Additive) siêu nhẹ"""
    def __init__(self, dim):
        super().__init__()
        self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.PReLU(dim))
        self.pool2 = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.PReLU(dim))
        self.pool4 = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.PReLU(dim))
        
        self.spp_fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
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
        
        # Additive SPP
        spp_fused = self.spp_fuse(x + x1 + x2 + x4)
        
        out = self.bn_refine(self.axial_refine(spp_fused))
        return self.coord_att(out + spp_fused)

# ==============================================================================
# 3. NHÁNH CHI TIẾT (DETAIL PATH) VÀ BỘ TRỘN (BGA)
# ==============================================================================
class DetailBranch(nn.Module):
    """
    Cặp mắt cận cảnh: Trích xuất góc nhìn nông, dừng lại ở 1/8.
    Giữ các mép nước cực kỳ sắc nét.
    """
    def __init__(self):
        super().__init__()
        self.S1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), # 1/2
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(32)
        )
        self.S2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 1/4
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.S3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 1/8
            nn.BatchNorm2d(128),
            nn.PReLU(128)
        )

    def forward(self, x):
        return self.S3(self.S2(self.S1(x)))

class BilateralGuidedAggregation(nn.Module):
    """
    Trạm kiểm soát BGA: Dùng Detail Mask để hướng dẫn Semantic Path.
    """
    def __init__(self, detail_c=128, semantic_c=256, out_c=128):
        super().__init__()
        # Căn chỉnh kênh và phóng to nhánh Semantic từ 1/16 lên 1/8
        self.sem_project = nn.Sequential(
            nn.Conv2d(semantic_c, detail_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_c),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Nhánh Detail tự tạo mặt nạ ranh giới (Mask)
        self.detail_gate = nn.Sequential(
            nn.Conv2d(detail_c, detail_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(detail_c),
            nn.Sigmoid()
        )
        
        # Dung hợp
        self.fuse = nn.Sequential(
            nn.Conv2d(detail_c * 2, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(out_c),
            CoordAtt(out_c, out_c) # Chốt chặn Attention cuối cùng
        )

    def forward(self, detail_feat, semantic_feat):
        # 1. Chuẩn bị ngữ nghĩa
        sem_up = self.sem_project(semantic_feat)
        
        # 2. Chi tiết tạo cổng (Gate)
        gate = self.detail_gate(detail_feat)
        
        # 3. Ngữ nghĩa đi qua cổng (Chỉ đổ màu vào vùng có ranh giới)
        sem_guided = sem_up * gate
        
        # 4. Gộp chung lại
        out = torch.cat([detail_feat, sem_guided], dim=1)
        return self.fuse(out)

# ==============================================================================
# 4. MẠNG CHÍNH (BILATERAL-PFCU)
# ==============================================================================
class Bilateral_PFCU(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (5, 5)

        # ---------------------------------------------------
        # LUỒNG 1: NHÁNH CHI TIẾT (Dừng ở 1/8)
        # ---------------------------------------------------
        self.detail_path = DetailBranch() # Output: 128 channels, 1/8 resolution

        # ---------------------------------------------------
        # LUỒNG 2: NHÁNH NGỮ NGHĨA (Xuống tận 1/16)
        # ---------------------------------------------------
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # Encoder hạ mẫu từ từ
        self.e1 = EncoderBlock(16,  32,  mixer_kernel=mk) # 1/2
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=mk) # 1/4
        self.e3 = EncoderBlock(64,  128, mixer_kernel=mk) # 1/8
        self.e4 = EncoderBlock(128, 256, mixer_kernel=mk) # 1/16

        # Bottleneck ở đáy mạng
        self.b4 = AdditiveBottleNeck(256) # Output: 256 channels, 1/16 resolution

        # ---------------------------------------------------
        # BỘ TRỘN VÀ PHÂN LOẠI
        # ---------------------------------------------------
        self.bga = BilateralGuidedAggregation(detail_c=128, semantic_c=256, out_c=128)
        
        # Đầu ra (Nhận 128 channels từ BGA)
        self.conv_out = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()[2:]

        # 1. Nhánh Detail
        feat_detail = self.detail_path(x)

        # 2. Nhánh Semantic
        feat_sem = self.conv_in(x)
        feat_sem = self.e1(feat_sem)
        feat_sem = self.e2(feat_sem)
        feat_sem = self.e3(feat_sem)
        feat_sem = self.e4(feat_sem)
        feat_sem = self.b4(feat_sem)

        # 3. Dung hợp tại BGA
        fused = self.bga(detail_feat=feat_detail, semantic_feat=feat_sem)

        # 4. Phân loại và Phóng to về kích thước gốc (từ 1/8 lên 1/1)
        out = self.conv_out(fused)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        return out

def build_model(num_classes=1):
    return Bilateral_PFCU(num_classes=num_classes)