import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# NÂNG CẤP 1: COORDINATE ATTENTION (Thay thế ECA)
# =========================================================================
class CoordAtt(nn.Module):
    """
    Coordinate Attention: Bắt ngữ cảnh không gian theo 2 chiều X và Y.
    Giúp mạng hiểu quy luật "nước ở dưới thấp, không ở trên trời".
    """
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        return identity * a_w * a_h

# =========================================================================
# NÂNG CẤP 2: LITE-PYRAMID POOLING MODULE (Lite-PPM)
# =========================================================================
class LitePPM(nn.Module):
    """
    Thu thập bối cảnh từ vùng rộng đến hẹp (1x1, 3x3) để tránh mạng bị 
    "loạn thị" khi nhìn thấy vùng lũ quá lớn.
    """
    def __init__(self, in_dim, reduction_dim=16, bins=(1, 3)):
        super(LitePPM, self).__init__()
        self.features = nn.ModuleList()
        for bin_size in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.PReLU()
            ))

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), size=x_size[2:], mode='bilinear', align_corners=False))
        return torch.cat(out, 1)

# =========================================================================
# CÁC LỚP BỔ TRỢ & DSCONV
# =========================================================================
class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# =========================================================================
# MODULE CỐT LÕI CỦA DABNET
# =========================================================================
class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        
        # Dùng CoordAtt thay cho ECA
        self.coord_att = CoordAtt(nIn)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.dconv1x3(self.dconv3x1(output))
        br2 = self.ddconv1x3(self.ddconv3x1(output))

        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        output = self.coord_att(output)
        return output + input

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        nConv = nOut - nIn if self.nIn < self.nOut else nOut

        self.dsconv = DSConv(nIn, nConv, kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.dsconv(input)
        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)
        return self.bn_prelu(output)

class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList([nn.AvgPool2d(3, stride=2, padding=1) for _ in range(ratio)])

    def forward(self, input):
        for pool in self.pool: input = pool(input)
        return input

# =========================================================================
# MẠNG CHÍNH (PRO DABNET V5)
# =========================================================================
class ProDABNet(nn.Module):
    def __init__(self, classes=1, block_1=3, block_2=6, ch_b1=64, ch_b2=96):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.down_1 = InputInjection(1)  
        self.down_2 = InputInjection(2)  
        self.down_3 = InputInjection(3)  

        self.bn_prelu_1 = BNPReLU(32 + 3)

        # Stage 1
        self.downsample_1 = DownSamplingBlock(32 + 3, ch_b1)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(ch_b1, d=2))
        self.bn_prelu_2 = BNPReLU((ch_b1 * 2) + 3)

        # Stage 2 (Hybrid Dilation)
        dilation_block_2 = [1, 2, 4, 8, 1, 2] 
        in_ch_b2 = (ch_b1 * 2) + 3
        self.downsample_2 = DownSamplingBlock(in_ch_b2, ch_b2)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i), DABModule(ch_b2, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU((ch_b2 * 2) + 3)

        # NÂNG CẤP 2 & 3: LITE-PPM & SPATIAL SKIP-FUSION
        # =========================================================
        # 1. Lite-PPM cho luồng sâu (Độ phân giải 1/8)
        deep_channels = (ch_b2 * 2) + 3 # 195 channels
        ppm_reductions = 16
        self.lite_ppm = LitePPM(in_dim=deep_channels, reduction_dim=ppm_reductions, bins=(1, 3))
        ppm_out_channels = deep_channels + (ppm_reductions * 2) # 195 + 32 = 227 channels
        
        # 2. Nhánh Shallow lấy chi tiết từ init_conv (Độ phân giải 1/2)
        self.skip_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU()
        )
        
        # 3. Nút thắt phân loại (Bottleneck Classifier)
        # Nhận: Nhánh Sâu đã upsample (227) + Nhánh Nông (16) = 243 channels
        fusion_channels = ppm_out_channels + 16
        
        self.classifier = nn.Sequential(
            nn.Conv2d(fusion_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True), 
            nn.Dropout2d(0.1),
            nn.Conv2d(64, classes, kernel_size=1, bias=True)
        )

    def forward(self, input):
        # 1. Nhánh Nông (Độ phân giải 1/2)
        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))

        # 2. Stage 1 (Độ phân giải 1/4)
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))

        # 3. Stage 2 (Độ phân giải 1/8)
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))

        # --- CÁC BƯỚC FUSION PRO ---
        # Đi qua Lite-PPM ở 1/8
        deep_feat = self.lite_ppm(output2_cat) 
        
        # Lấy mép nước từ init_conv ở 1/2
        shallow_feat = self.skip_conv(output0) 
        
        # Upsample nhánh sâu từ 1/8 lên 1/2 để ghép với nhánh nông
        deep_upsampled = F.interpolate(deep_feat, size=shallow_feat.size()[2:], mode='bilinear', align_corners=False)
        
        # Nối tại 1/2 độ phân giải
        fusion_feat = torch.cat([deep_upsampled, shallow_feat], dim=1)

        # Phân loại tại 1/2 -> Upsample lần cuối ra 1/1
        out = self.classifier(fusion_feat)
        out = F.interpolate(out, size=input.size()[2:], mode='bilinear', align_corners=False)

        return out

# =========================================================================
# HÀM BUILD MODEL CHUẨN TEMPLATE
# =========================================================================
def build_model(num_classes=1):
    """
    Khởi tạo mạng ProDABNet V5.
    (DSConv + CoordAtt + Lite-PPM + Spatial Skip-Fusion + Hybrid Dilation)
    """
    return ProDABNet(classes=num_classes, block_1=3, block_2=6, ch_b1=64, ch_b2=96)