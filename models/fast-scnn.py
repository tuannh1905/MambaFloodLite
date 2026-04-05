import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# FAST-SCNN (MCU-NATIVE EDITION)
# - Hỗ trợ input_size: 128 và 256
# - Loại bỏ Bilinear (thay bằng Nearest)
# - Loại bỏ AdaptiveAvgPool2d (thay bằng StaticPyramidPooling)
# - Static scale_factor=8 ở output thay vì size động.
# - Chuẩn hóa 100% BN Folding.
# ==============================================================================

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU (Tương thích 100% BN Folding)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""
    def __init__(self, dw_channels, out_channels, stride=1):
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
    def __init__(self, dw_channels, out_channels, stride=1):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class LinearBottleneck(nn.Module):
    """LinearBottleneck (Residual cộng sau BN -> An toàn cho Folding)"""
    def __init__(self, in_channels, out_channels, t=6, stride=2):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = (stride == 1 and in_channels == out_channels)
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

class StaticPyramidPooling(nn.Module):
    """
    ✓ ĐÃ FIX: Pyramid Pooling TĨNH (Không dùng AdaptiveAvgPool)
    """
    def __init__(self, in_channels, out_channels, input_size=256):
        super(StaticPyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        
        # Fast-SCNN hạ sample 32 lần (2^5). 
        # Nếu input=256 -> feat=8. Nếu input=128 -> feat=4.
        feat_size = input_size // 32
        
        k1, k2, k3 = feat_size, feat_size // 2, feat_size // 4
        
        if input_size == 128:
            assert k3 >= 1, "Input size quá nhỏ"
        elif input_size == 256:
            pass
        else:
            raise ValueError(f"input_size={input_size} chưa hỗ trợ. Dùng 128 hoặc 256.")

        self._sf1, self._sf2, self._sf3 = int(k1), int(k2), int(k3)

        self.pool1 = nn.Sequential(nn.AvgPool2d(k1, k1), _ConvBNReLU(in_channels, inter_channels, 1))
        self.pool2 = nn.Sequential(nn.AvgPool2d(k2, k2), _ConvBNReLU(in_channels, inter_channels, 1))
        self.pool3 = nn.Sequential(nn.AvgPool2d(k3, k3), _ConvBNReLU(in_channels, inter_channels, 1))
        # Bỏ nhánh pool thứ 4 (k=6) của bản gốc vì nó vô lý về mặt toán học với ảnh 128x128.

        self.out = _ConvBNReLU(in_channels + inter_channels * 3, out_channels, 1)

    def forward(self, x):
        # ✓ ĐÃ FIX: Dùng Nearest Upsample tĩnh thay vì Bilinear
        feat1 = F.interpolate(self.pool1(x), scale_factor=self._sf1, mode='nearest')
        feat2 = F.interpolate(self.pool2(x), scale_factor=self._sf2, mode='nearest')
        feat3 = F.interpolate(self.pool3(x), scale_factor=self._sf3, mode='nearest')
        
        x = torch.cat([x, feat1, feat2, feat3], dim=1)
        return self.out(x)

class LearningToDownsample(nn.Module):
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2, padding=1) # Thêm padding=1 để bù cho kernel 3x3
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x

class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), input_size=256):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        
        # Truyền input_size vào Pyramid Pooling
        self.ppm = StaticPyramidPooling(block_channels[2], out_channels, input_size=input_size)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, highter_in_channels, lower_in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
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
        # ✓ ĐÃ FIX: mode='nearest', scale_factor=4 (vì lower_res đang nhỏ hơn higher_res 4 lần: 1/32 vs 1/8)
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='nearest')
        
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)

class Classifer(nn.Module):
    def __init__(self, dw_channels, num_classes, stride=1):
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

class FastSCNN_MCU(nn.Module):
    def __init__(self, num_classes=1, aux=False, input_size=256):
        super(FastSCNN_MCU, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        
        # ✓ ĐÃ FIX: Truyền input_size để tính toán tĩnh
        self.global_feature_extractor = GlobalFeatureExtractor(
            64, [64, 96, 128], 128, 6, [3, 3, 3], input_size=input_size
        )
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        higher_res_features = self.learning_to_downsample(x) # scale 1/8
        x = self.global_feature_extractor(higher_res_features) # scale 1/32
        x = self.feature_fusion(higher_res_features, x) # scale 1/8
        x = self.classifier(x) # scale 1/8
        
        # ✓ ĐÃ FIX: Dùng scale_factor=8 (hằng số tĩnh) thay cho size = x.shape[2:] động.
        x = F.interpolate(x, scale_factor=8, mode='nearest')
        
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, scale_factor=8, mode='nearest')
            return x, auxout
        
        return x 

def build_model(num_classes=1, input_size=256):
    """
    Hàm gọi chuẩn theo format của project (Có nhận tham số input_size)
    """
    return FastSCNN_MCU(num_classes=num_classes, aux=False, input_size=input_size)