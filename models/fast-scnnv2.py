import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. CÁC LỚP BỔ TRỢ CƠ BẢN VÀ ATTENTION
# ==============================================================================
class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.conv(x)

class _DSConv(nn.Module):
    """Depthwise Separable Conv: Tối ưu tham số cực độ (DW + PW)"""
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
    def forward(self, x):
        return self.conv(x)

class _DWConv(nn.Module):
    """Chỉ gom không gian, KHÔNG trộn kênh"""
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.conv(x)

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
    def __init__(self, dim):
        super().__init__()
        self.dg_dw_h = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=False)
        self.dg_dw_w = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        return self.bn(x + self.dg_dw_h(x) + self.dg_dw_w(x))

# ==============================================================================
# 2. KHỐI DAB BOTTLENECK MỚI (CHUẨN DEPTHWISE & ASYMMETRIC)
# ==============================================================================
class DAB_Bottleneck(nn.Module):
    """
    Kết hợp LinearBottleneck + DAB Module.
    Đã hạ Expansion Ratio (t) xuống để ép cân cực hạn.
    """
    def __init__(self, in_channels, out_channels, t=2, stride=1, d=2, **kwargs):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        mid_channels = int(in_channels * t) # T=2 giúp giảm tham số 1x1 rất mạnh

        # 1. Trộn và mở rộng kênh (Pointwise 1x1)
        self.conv1x1_in = _ConvBNReLU(in_channels, mid_channels, 1)

        # 2. Asymmetric Depthwise (Không gian)
        self.dconv3x1_1 = nn.Conv2d(mid_channels, mid_channels, (3, 1), stride, padding=(1, 0), groups=mid_channels, bias=False)
        self.dconv1x3_1 = nn.Conv2d(mid_channels, mid_channels, (1, 3), 1, padding=(0, 1), groups=mid_channels, bias=False)

        self.dconv3x1_2 = nn.Conv2d(mid_channels, mid_channels, (3, 1), stride, padding=(d, 0), dilation=(d, 1), groups=mid_channels, bias=False)
        self.dconv1x3_2 = nn.Conv2d(mid_channels, mid_channels, (1, 3), 1, padding=(0, d), dilation=(1, d), groups=mid_channels, bias=False)

        self.bn_relu_mid = nn.Sequential(nn.BatchNorm2d(mid_channels), nn.ReLU(True))

        # 3. Nén kênh lại (Pointwise Linear 1x1)
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv1x1_in(x)

        br1 = self.dconv1x3_1(self.dconv3x1_1(out))
        br2 = self.dconv1x3_2(self.dconv3x1_2(out))

        out = self.bn_relu_mid(br1 + br2)
        out = self.conv1x1_out(out)

        if self.use_shortcut:
            out = x + out
        return out

# ==============================================================================
# 3. MACRO TWO-BRANCH ARCHITECTURE
# ==============================================================================
class LearningToDownsample(nn.Module):
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super().__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2, padding=1)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)
        self.detail_guidance = DetailGuidance(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.detail_guidance(x)
        return x

class ContextEmbeddingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = _ConvBNReLU(in_channels, out_channels, 1)
        self.conv3x3_dw = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.align = nn.Identity() if in_channels == out_channels else _ConvBNReLU(in_channels, out_channels, 1)

    def forward(self, x):
        global_context = self.conv1x1(self.gap(x))
        out = self.align(x) + global_context
        return self.conv3x3_dw(out)

class GlobalFeatureExtractor(nn.Module):
    """
    Nhánh Semantic đã được "Ép cân": 
    - Channel Capping: Nở tối đa [64, 64, 96]
    - Expansion Ratio: t=2
    - Block config: [2, 2, 3] với Dilation Ziczac
    """
    def __init__(self, in_channels=64, block_channels=(64, 64, 96), out_channels=96, t=2, num_blocks=(2, 2, 3), **kwargs):
        super().__init__()
        
        # Ziczac Dilation điều chỉnh cho cấu hình [2, 2, 3]
        self.bottleneck1 = self._make_layer(DAB_Bottleneck, in_channels, block_channels[0], num_blocks[0], t, 2, [2, 4])
        self.bottleneck2 = self._make_layer(DAB_Bottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2, [8, 16])
        self.bottleneck3 = self._make_layer(DAB_Bottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1, [8, 4, 2])
        
        self.ce_block = ContextEmbeddingBlock(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t, stride, dils):
        layers = []
        layers.append(block(inplanes, planes, t, stride, d=dils[0]))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1, d=dils[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ce_block(x)
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super().__init__()
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)
        self.coord_att = CoordAtt(out_channels, out_channels)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, size=higher_res_feature.shape[2:], mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        
        out = higher_res_feature + lower_res_feature
        out = self.relu(out)
        
        return self.coord_att(out)

class Classifer(nn.Module):
    """
    Sử dụng hoàn toàn Depthwise Separable (_DSConv) 
    Giữ tham số lớp này ở mức ~17K (nếu channels = 96)
    """
    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super().__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1) # Raw logits 1x1 mapping
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return self.conv(x)

# ==============================================================================
# 4. MẠNG CHÍNH (FAST-SCNN MICRO - < 200K PARAMS)
# ==============================================================================
class FastSCNN_Micro(nn.Module):
    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        # Nhánh Detail (High-Res 1/4) - Output 64 channels
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        
        # Nhánh Semantic (Low-Res 1/16) - Ép cân Input=64, Output=96
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 64, 96], 96, t=2, num_blocks=(2, 2, 3))
        
        # Dung hợp hai nhánh: Đưa tất cả về mức Capped 96 Channels
        self.feature_fusion = FeatureFusionModule(highter_in_channels=64, lower_in_channels=96, out_channels=96)
        
        # Phân loại (Nhận 96 channels từ FFM)
        self.classifier = Classifer(96, num_classes)

    def forward(self, x):
        size = x.size()[2:]
        
        higher_res_features = self.learning_to_downsample(x)
        x_sem = self.global_feature_extractor(higher_res_features)
        
        x_fuse = self.feature_fusion(higher_res_features, x_sem)
        out = self.classifier(x_fuse)
        
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)
        return out 

def build_model(num_classes=1):
    return FastSCNN_Micro(num_classes=num_classes)