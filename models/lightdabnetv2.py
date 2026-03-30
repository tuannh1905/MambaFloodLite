import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. CÁC LỚP BỔ TRỢ (ĐÃ TỐI ƯU HÓA)
# =========================================================================
class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        return self.acti(self.bn(input))

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output

class DW_DownSamplingBlock(nn.Module):
    """
    Downsample tuân thủ tuyệt đối quy tắc: DW cho Không gian, 1x1 cho Kênh.
    """
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        nConv = nOut - nIn if nIn < nOut else nOut

        # Tích chập chiều sâu (Spatial) + 1x1 (Channel)
        self.dw_conv = Conv(nIn, nIn, kSize=3, stride=2, padding=1, groups=nIn, bn_acti=False)
        self.pw_conv = Conv(nIn, nConv, kSize=1, stride=1, padding=0, bn_acti=False)
        
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        feat = self.pw_conv(self.dw_conv(input))
        if self.nIn < self.nOut:
            pool = self.max_pool(input)
            feat = torch.cat([feat, pool], dim=1)
        return self.bn_prelu(feat)

# =========================================================================
# 2. MODULE CHÚ Ý VÀ RANH GIỚI
# =========================================================================
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
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
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()
        return identity * a_w * a_h

class DetailGuidance(nn.Module):
    """ Mỏ neo giữ ranh giới tại nhánh Detail 1/4 """
    def __init__(self, dim):
        super().__init__()
        self.dg_dw_h = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding='same', groups=dim, bias=False)
        self.dg_dw_w = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding='same', groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        return self.bn(x + self.dg_dw_h(x) + self.dg_dw_w(x))

# =========================================================================
# 3. MODULE CỐT LÕI MỚI (DAB DW MODULE)
# =========================================================================
class DAB_DW_Module(nn.Module):
    """
    Bản nâng cấp tuân thủ quy tắc: 
    - 1x1 Conv để trộn kênh.
    - Dùng 100% DW Conv (groups=channels) cho Không gian.
    """
    def __init__(self, nIn, d=1, dkSize=3):
        super().__init__()
        nMid = nIn // 2
        
        self.bn_relu_1 = BNPReLU(nIn)
        
        # 1. TRỘN KÊNH TRƯỚC (Bỏ Conv 3x3)
        self.conv1x1_in = Conv(nIn, nMid, 1, 1, padding=0, bn_acti=True)

        # 2. XỬ LÝ KHÔNG GIAN BẰNG DEPTHWISE (groups = nMid)
        # Nhánh 1: Không đối xứng cơ bản
        self.dconv3x1 = Conv(nMid, nMid, (dkSize, 1), 1, padding=(1, 0), groups=nMid, bn_acti=True)
        self.dconv1x3 = Conv(nMid, nMid, (1, dkSize), 1, padding=(0, 1), groups=nMid, bn_acti=True)
        
        # Nhánh 2: Không đối xứng mở rộng (Dilated)
        self.ddconv3x1 = Conv(nMid, nMid, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nMid, bn_acti=True)
        self.ddconv1x3 = Conv(nMid, nMid, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nMid, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nMid)
        
        # 3. TRỘN KÊNH VÀ PHỤC HỒI
        self.conv1x1_out = Conv(nMid, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        x = self.bn_relu_1(input)
        x = self.conv1x1_in(x)

        br1 = self.dconv1x3(self.dconv3x1(x))
        br2 = self.ddconv1x3(self.ddconv3x1(x))

        out = self.bn_relu_2(br1 + br2)
        out = self.conv1x1_out(out)

        return out + input

# =========================================================================
# 4. MẠNG CHÍNH (DABNET V2 - DUAL BRANCH)
# =========================================================================
class DABNet_V2(nn.Module):
    def __init__(self, classes=1):
        super().__init__()
        
        # STEM (Hạ mẫu xuống 1/2)
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        # ---------------------------------------------------------
        # NHÁNH 1: SPATIAL DETAIL BRANCH (Độ phân giải 1/4)
        # ---------------------------------------------------------
        self.downsample_detail = DW_DownSamplingBlock(32, 64)
        self.detail_block = nn.Sequential(
            DAB_DW_Module(64, d=1),
            DAB_DW_Module(64, d=1),
            DAB_DW_Module(64, d=1)
        )
        self.detail_guidance = DetailGuidance(64)

        # ---------------------------------------------------------
        # NHÁNH 2: SEMANTIC CONTEXT BRANCH (Đào sâu xuống 1/16)
        # ---------------------------------------------------------
        self.downsample_sem_1 = DW_DownSamplingBlock(64, 128)  # Xuống 1/8
        self.downsample_sem_2 = DW_DownSamplingBlock(128, 256) # Xuống 1/16
        
        # Áp dụng Hybrid DeepLab Style Dilation (Ziczac)
        dilations = [2, 4, 8, 16, 8, 4]
        self.semantic_block = nn.Sequential()
        for i, d in enumerate(dilations):
            self.semantic_block.add_module(f"DAB_Sem_{i}", DAB_DW_Module(256, d=d))

        # ---------------------------------------------------------
        # FUSION MODULE
        # ---------------------------------------------------------
        self.up_sem = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.fuse_conv = nn.Sequential(
            Conv(256 + 64, 128, 1, 1, padding=0, bn_acti=True),
            Conv(128, 128, 3, 1, padding=1, groups=128, bn_acti=True) # DW Smooth
        )
        
        # Cú chốt hạ quyết định ranh giới
        self.coord_att = CoordAtt(128, 128)
        self.classifier = nn.Conv2d(128, classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        
        x_init = self.init_conv(x) # 1/2
        
        # Xử lý nhánh Detail (1/4)
        feat_detail = self.downsample_detail(x_init)
        feat_detail = self.detail_block(feat_detail)
        feat_detail = self.detail_guidance(feat_detail) # Mỏ neo ranh giới
        
        # Xử lý nhánh Semantic (1/16)
        feat_sem = self.downsample_sem_1(feat_detail)
        feat_sem = self.downsample_sem_2(feat_sem)
        feat_sem = self.semantic_block(feat_sem) # Ziczac Dilation
        
        # Dung hợp (Phóng 1/16 lên 1/4 để khớp với Detail)
        feat_sem_up = self.up_sem(feat_sem)
        fused = torch.cat([feat_detail, feat_sem_up], dim=1)
        
        fused = self.fuse_conv(fused)
        fused = self.coord_att(fused)
        
        out = self.classifier(fused)
        
        # Phóng to trả về kích thước gốc
        return F.interpolate(out, size, mode='bilinear', align_corners=False)

def build_model(num_classes=1):
    return DABNet_V2(classes=num_classes)