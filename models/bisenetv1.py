import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SpatialPath(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, 64, stride=2)
        self.conv2 = ConvBNReLU(64, 128, stride=2)
        self.conv3 = ConvBNReLU(128, out_channels, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionRefinementModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv(attn)
        attn = self.bn(attn)
        attn = self.sigmoid(attn)
        return x * attn


class ContextPath(nn.Module):
    def __init__(self, out_channels=128):
        super(ContextPath, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer2 = resnet.layer1
        self.layer3 = resnet.layer2
        self.layer4 = resnet.layer3
        self.layer5 = resnet.layer4

        self.arm16 = AttentionRefinementModule(256)
        self.arm32 = AttentionRefinementModule(512)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv16 = ConvBNReLU(256, out_channels, kernel_size=1, padding=0)
        self.conv32 = ConvBNReLU(512, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x8 = self.layer3(x)
        x16 = self.layer4(x8)
        x32 = self.layer5(x16)

        global_ctx = self.global_pool(x32)
        global_ctx = global_ctx.expand_as(x32)

        x32 = self.arm32(x32) + global_ctx
        x32 = self.conv32(x32)
        x32 = F.interpolate(x32, scale_factor=2, mode='bilinear', align_corners=True)

        x16 = self.arm16(x16)
        x16 = self.conv16(x16)
        x16 = x16 + x32
        x16 = F.interpolate(x16, scale_factor=2, mode='bilinear', align_corners=True)

        return x16


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_sp, x_cp):
        x = torch.cat([x_sp, x_cp], dim=1)
        x = self.conv(x)
        attn = self.pool(x)
        attn = self.fc1(attn)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)
        x = x + x * attn
        return x


class BiSeNetV1(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        """
        BiSeNet V1: Bilateral Segmentation Network
        Paper: https://arxiv.org/abs/1808.00897

        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            num_classes: Number of output classes
                        - 1 for binary segmentation
                        - >=2 for multi-class
        """
        super(BiSeNetV1, self).__init__()
        self.num_classes = num_classes
        self.spatial_path = SpatialPath(in_channels, 128)
        self.context_path = ContextPath(out_channels=128)
        self.ffm = FeatureFusionModule(256, 256)
        self.head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        size = x.size()[2:]
        x_sp = self.spatial_path(x)
        x_cp = self.context_path(x)
        x = self.ffm(x_sp, x_cp)
        x = self.head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


def build_model(num_classes=1):
    return BiSeNetV1(in_channels=3, num_classes=num_classes)