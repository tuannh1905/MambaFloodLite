import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. RESNET BACKBONE (Với Dilated Convolution cho Output Stride = 16)
# ==============================================================================
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetDilated(nn.Module):
    def __init__(self, layers, output_stride=16):
        super(ResNetDilated, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise ValueError("output_stride must be 8 or 16")

        self.layer1 = self._make_layer(Bottleneck, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=strides[3], dilation=dilations[3])

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        # Xử lý downsample cho nhánh shortcut
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        # Lớp đầu tiên của block có thể thay đổi stride và apply 1/2 dilation
        layers.append(block(self.inplanes, planes, stride, dilation=1 if dilation == 1 else dilation//2, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# ==============================================================================
# 2. KHỐI ASPP (Atrous Spatial Pyramid Pooling)
# ==============================================================================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1), # Sức mạnh động của DeepLab, giữ nguyên không tĩnh hóa
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # Resize động lại bằng kích thước feature map hiện tại
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPPModule, self).__init__()
        modules = []
        
        # 1x1 Conv branch
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # Atrous Conv branches
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Image Pooling branch
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Concatenate & Project
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

# ==============================================================================
# 3. DEEPLAB V3 CHÍNH
# ==============================================================================
class DeepLabV3Model(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet50', output_stride=16, input_size=256):
        super(DeepLabV3Model, self).__init__()

        # ✓ KIỂM TRA TOÁN HỌC: Output stride = 16 yêu cầu input chia hết cho 16
        if input_size % 16 != 0:
            raise ValueError(f"DeepLabV3 yêu cầu input_size chia hết cho 16. Kích thước {input_size} không hợp lệ.")

        if backbone == 'resnet50':
            self.backbone = ResNetDilated([3, 4, 6, 3], output_stride=output_stride)
        elif backbone == 'resnet101':
            self.backbone = ResNetDilated([3, 4, 23, 3], output_stride=output_stride)
        else:
            raise NotImplementedError("Backbone chưa được hỗ trợ.")

        in_channels = 2048
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]

        self.aspp = ASPPModule(in_channels, atrous_rates)

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        
        # 1. Extract Features
        x = self.backbone(x)
        
        # 2. Atrous Spatial Pyramid Pooling
        x = self.aspp(x)
        
        # 3. Logits (1x1 Conv)
        x = self.classifier(x)
        
        # 4. Bilinear Upsample thẳng lên kích thước ảnh gốc (DeepLabV3 nguyên thủy)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x

# ==============================================================================
# 4. TEMPLATE BUILD MODEL
# ==============================================================================
def build_model(num_classes=1, input_size=256):
    """
    Khởi tạo DeepLabV3 chuẩn với backbone ResNet-50.
    Cực kỳ thích hợp làm model đối chứng (heavy baseline).
    """
    return DeepLabV3Model(num_classes=num_classes, backbone='resnet50', output_stride=16, input_size=input_size)