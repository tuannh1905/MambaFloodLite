import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# [CẢI TIẾN 4]: THAY RELU BẰNG HARDSWISH
# =========================================================================
# (Thay thế toàn bộ ReLU bằng Hardswish trong các khối cơ bản)
class _ConvBNHSwish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(_ConvBNHSwish, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish()  # [CẢI TIẾN 4]
        )
    def forward(self, x): return self.conv(x)

class _DSConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.Hardswish(), # [CẢI TIẾN 4]
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish()  # [CẢI TIẾN 4]
        )
    def forward(self, x): return self.conv(x)

class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish()  # [CẢI TIẾN 4]
        )
    def forward(self, x): return self.conv(x)


# =========================================================================
# [CẢI TIẾN 1]: COORDINATE ATTENTION (CA) THAY CHO SE BLOCK
# =========================================================================
class CoordAtt(nn.Module):
    """Nhạy cảm với vị trí không gian ngang/dọc của dải nước lũ"""
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish() # [CẢI TIẾN 4]
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

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

        out = identity * a_w * a_h
        return out


# =========================================================================
# [CẢI TIẾN 2]: GHOST MODULE CHO BOTTLENECK
# =========================================================================
class GhostModule(nn.Module):
    """Tạo ra gấp đôi lượng Feature Maps nhưng chỉ tốn một nửa tham số"""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, use_act=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.Hardswish() if use_act else nn.Identity(), # [CẢI TIẾN 4]
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.Hardswish() if use_act else nn.Identity(), # [CẢI TIẾN 4]
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class GhostBottleneck_CA(nn.Module):
    """Linear Bottleneck được độ lại bằng Ghost Conv + Coordinate Attention"""
    def __init__(self, in_channels, out_channels, t=6, stride=2):
        super(GhostBottleneck_CA, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * t
        
        self.block = nn.Sequential(
            GhostModule(in_channels, hidden_dim, kernel_size=1, use_act=True), # [CẢI TIẾN 2]
            _DWConv(hidden_dim, hidden_dim, stride),                           # Depthwise Conv
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),                # Pointwise Linear
            nn.BatchNorm2d(out_channels)
        )
        self.ca = CoordAtt(out_channels, out_channels) # [CẢI TIẾN 1]

    def forward(self, x):
        out = self.block(x)
        out = self.ca(out)
        if self.use_shortcut:
            out = x + out
        return out


# =========================================================================
# MODULES PHỤ TRỢ TỪ PHIÊN BẢN TRƯỚC (STRIP POOLING & SPATIAL ATTENTION)
# =========================================================================
class StripPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish() # [CẢI TIẾN 4]
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = F.interpolate(self.conv1(x1), (h, w), mode='bilinear', align_corners=True)
        x2 = self.pool2(x)
        x2 = F.interpolate(self.conv2(x2), (h, w), mode='bilinear', align_corners=True)
        sp_feat = torch.sigmoid(x1 + x2) * x
        return self.out_conv(torch.cat([x, sp_feat], dim=1))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(y))


# =========================================================================
# MODULES CHÍNH
# =========================================================================
class LearningToDownsample(nn.Module):
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNHSwish(3, dw_channels1, 3, 2, 1) 
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x

class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, block_channels=(32, 48, 64),
                 out_channels=64, t=6, num_blocks=(3, 3, 3)):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(GhostBottleneck_CA, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(GhostBottleneck_CA, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(GhostBottleneck_CA, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = StripPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t=t, stride=stride))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, t=t, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.conv_higher_res = nn.Sequential(nn.Conv2d(highter_in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.act = nn.Hardswish() # [CẢI TIẾN 4]
        self.spatial_attn = SpatialAttention()

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        attn_map = self.spatial_attn(lower_res_feature)
        higher_res_feature = higher_res_feature * attn_map

        out = higher_res_feature + lower_res_feature
        return self.act(out)

class Classifer(nn.Module):
    def __init__(self, dw_channels, num_classes, stride=1):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(dw_channels, num_classes, 1))

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


# =========================================================================
# MẠNG CHÍNH
# =========================================================================
class FastSCNN_Ultimate(nn.Module):
    def __init__(self, num_classes=1, aux=False):
        super(FastSCNN_Ultimate, self).__init__()
        self.aux = aux
        
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        
        self.global_feature_extractor = GlobalFeatureExtractor(
            in_channels=64, 
            block_channels=[32, 48, 64], 
            out_channels=64, 
            t=6, 
            num_blocks=(3, 3, 3)
        )
        
        self.feature_fusion = FeatureFusionModule(64, 64, 64)
        self.classifier = Classifer(64, num_classes)
        
        # [CẢI TIẾN 3]: Auxiliary Loss ở nhánh Downsample
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.Hardswish(),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x_gfe = self.global_feature_extractor(higher_res_features)
        x_ffm = self.feature_fusion(higher_res_features, x_gfe)
        out = self.classifier(x_ffm)
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)
        
        # [CẢI TIẾN 3]: Chỉ kích hoạt nhánh Aux khi đang Training
        if self.aux and self.training:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            return out, auxout
            
        return out


# =========================================================================
# TEMPLATE CỦA PROJECT
# =========================================================================
def build_model(num_classes=1):
    # Đặt aux=False để tương thích 100% với trainer.py của bạn lúc Inference.
    # Nếu bạn muốn test Aux Loss lúc train, bạn có thể chỉnh trainer.py
    # để xử lý Tuple đầu ra, sau đó đổi aux=True ở đây.
    return FastSCNN_Ultimate(num_classes=num_classes, aux=False)