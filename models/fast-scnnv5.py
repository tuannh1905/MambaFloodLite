import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# [CẢI TIẾN 2]: KHỐI SQUEEZE-AND-EXCITATION (SE)
# =========================================================================
class SEBlock(nn.Module):
    """Khối SE giúp mạng tự động đánh trọng số cho các kênh đặc trưng quan trọng"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # max(1, ...) để tránh lỗi khi channel < 16
        mid_channel = max(1, channel // reduction) 
        self.fc = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# =========================================================================
# [CẢI TIẾN 3]: KHỐI STRIP POOLING (THAY THẾ PYRAMID POOLING)
# =========================================================================
class StripPooling(nn.Module):
    """Bắt các dải nước kéo dài theo chiều ngang/dọc thay vì khối vuông"""
    def __init__(self, in_channels, out_channels):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # Dải ngang
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # Dải dọc
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
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
        
        sp_feat = torch.sigmoid(x1 + x2) * x
        out = self.out_conv(torch.cat([x, sp_feat], dim=1))
        return out


# =========================================================================
# [CẢI TIẾN 4]: SPATIAL ATTENTION MODULE
# =========================================================================
class SpatialAttention(nn.Module):
    """Tập trung không gian: Dạy nhánh chi tiết biết đâu là vùng cần lấy nét"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)


# =========================================================================
# CÁC LỚP CƠ BẢN (CHỨA CẢI TIẾN SE)
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
        # [CẢI TIẾN 2]: Gắn SE Block vào cuối mỗi LinearBottleneck
        self.se = SEBlock(out_channels)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        if self.use_shortcut:
            out = x + out
        return out


# =========================================================================
# MODULES CHÍNH
# =========================================================================
class LearningToDownsample(nn.Module):
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2, 1) # Thêm padding 1
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, block_channels=(32, 48, 64),
                 out_channels=64, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        
        # [CẢI TIẾN 3]: StripPooling thay cho PyramidPooling
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
        
        # [CẢI TIẾN 4]: Spatial Attention Module
        self.spatial_attn = SpatialAttention()

    def forward(self, higher_res_feature, lower_res_feature):
        # 1. Xử lý nhánh Ngữ cảnh (Lower / Global)
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        # 2. Xử lý nhánh Chi tiết (Higher / Shallow)
        higher_res_feature = self.conv_higher_res(higher_res_feature)

        # [CẢI TIẾN 4]: Dùng nhánh Ngữ cảnh tạo map Spatial Attention để LỌC nhánh Chi tiết
        attn_map = self.spatial_attn(lower_res_feature)
        higher_res_feature = higher_res_feature * attn_map

        # 3. Cộng gộp
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
# GÓI GỌN VÀO MODEL CHÍNH
# =========================================================================
class FastSCNN(nn.Module):
    def __init__(self, num_classes=1, **kwargs):
        super(FastSCNN, self).__init__()
        
        # [CẢI TIẾN 1]: Đầu vào giữ nguyên để tránh mất tín hiệu, ép kênh mạnh ở GFE
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        
        # block_channels gốc [64, 96, 128] -> Ép xuống [32, 48, 64]
        # Đầu ra của GFE được ép về 64 (thay vì 128)
        self.global_feature_extractor = GlobalFeatureExtractor(
            in_channels=64, 
            block_channels=[32, 48, 64], 
            out_channels=64, 
            t=6, 
            num_blocks=(3, 3, 3)
        )
        
        # FFM nhận input: higher=64, lower=64, output_channel=64
        self.feature_fusion = FeatureFusionModule(64, 64, 64)
        
        # Classifier nhận 64 thay vì 128
        self.classifier = Classifer(64, num_classes)

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x_gfe = self.global_feature_extractor(higher_res_features)
        x_ffm = self.feature_fusion(higher_res_features, x_gfe)
        out = self.classifier(x_ffm)
        
        return F.interpolate(out, size, mode='bilinear', align_corners=True)


# =========================================================================
# TEMPLATE CỦA PROJECT
# =========================================================================
def build_model(num_classes=1):
    return FastSCNN(num_classes=num_classes)