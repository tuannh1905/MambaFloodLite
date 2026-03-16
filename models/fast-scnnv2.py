import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. STRIP POOLING MODULE (THAY THẾ PYRAMID POOLING)
# =========================================================================
# [ĐÃ SỬA]: Thêm module StripPooling mới để bắt các dải nước ngập ngang/dọc
class StripPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # Dải ngang
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # Dải dọc
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        # Thêm 1 lớp conv 1x1 để fuse (tương đương với out của Pyramid Pooling cũ)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = F.interpolate(self.conv1(x1), (h, w), mode='bilinear', align_corners=True)
        
        x2 = self.pool2(x)
        x2 = F.interpolate(self.conv2(x2), (h, w), mode='bilinear', align_corners=True)
        
        # Cộng dải ngang + dọc, sau đó concat với input gốc
        sp_feat = torch.sigmoid(x1 + x2) * x
        out = self.out_conv(torch.cat([x, sp_feat], dim=1))
        return out


# =========================================================================
# 2. CÁC LỚP CƠ BẢN (GIỮ NGUYÊN TỪ BẢN GỐC)
# =========================================================================
class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x): return self.conv(x)


class _DSConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x): return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x): return self.conv(x)


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            _ConvBNReLU(in_channels, in_channels * t, 1),
            _DWConv(in_channels * t, in_channels * t, stride),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


# =========================================================================
# 3. CÁC MODULE CHÍNH TRONG MẠNG
# =========================================================================
class LearningToDownsample(nn.Module):
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2, 1) # [ĐÃ SỬA]: Thêm padding=1 cho Conv đầu tiên
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        
        # [ĐÃ SỬA]: Thay PyramidPooling gốc bằng StripPooling
        # self.ppm = PyramidPooling(block_channels[2], out_channels)
        self.ppm = StripPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        # [ĐÃ SỬA]: align_corners=True có thể gây warning ở bản PyTorch mới, đổi thành False cho an toàn nếu có
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )
    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


# =========================================================================
# 4. MAIN NETWORK (CẤU HÌNH THEO CHIẾN LƯỢC WIDTH MULTIPLIER)
# =========================================================================
class FastSCNN_Slim(nn.Module):
    def __init__(self, num_classes=1, **kwargs):
        super(FastSCNN_Slim, self).__init__()
        
        # [ĐÃ SỬA]: Áp dụng Width Multiplier
        # LDS Module: alpha_lds = 0.5 (Giữ thông tin nông)
        # GFE Module: alpha_gfe = 0.35 (Ép cân sâu)
        alpha_lds = 0.5
        alpha_gfe = 0.35
        
        # Original LDS: [32, 48, 64] -> Slim: [16, 24, 32]
        lds_ch = [int(ch * alpha_lds) for ch in [32, 48, 64]]
        
        # Original GFE: block_channels=[64, 96, 128], out=128 
        # -> Slim: [22, 33, 44], out=44
        gfe_block_ch = [int(ch * alpha_gfe) for ch in [64, 96, 128]]
        gfe_out_ch = int(128 * alpha_gfe)
        
        # Original FFM: high=64, low=128, out=128
        # -> Slim: high=32, low=44, out=128 (FFM out giữ 128 để classifier dễ học, hoặc ép luôn xuống 44)
        ffm_out_ch = gfe_out_ch # Ép luôn xuống 44 cho đồng bộ
        
        self.learning_to_downsample = LearningToDownsample(
            dw_channels1=lds_ch[0], 
            dw_channels2=lds_ch[1], 
            out_channels=lds_ch[2]
        )
        
        self.global_feature_extractor = GlobalFeatureExtractor(
            in_channels=lds_ch[2], 
            block_channels=gfe_block_ch, 
            out_channels=gfe_out_ch, 
            t=6, 
            num_blocks=(3, 3, 3)
        )
        
        self.feature_fusion = FeatureFusionModule(
            highter_in_channels=lds_ch[2], 
            lower_in_channels=gfe_out_ch, 
            out_channels=ffm_out_ch
        )
        
        self.classifier = Classifer(dw_channels=ffm_out_ch, num_classes=num_classes)

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x_gfe = self.global_feature_extractor(higher_res_features)
        x_ffm = self.feature_fusion(higher_res_features, x_gfe)
        out = self.classifier(x_ffm)
        
        # Trả về tensor duy nhất tương thích trainer.py
        return F.interpolate(out, size, mode='bilinear', align_corners=True)


# =========================================================================
# 5. HÀM BUILD MODEL CHUẨN TEMPLATE
# =========================================================================
def build_model(num_classes=1):
    return FastSCNN_Slim(num_classes=num_classes)