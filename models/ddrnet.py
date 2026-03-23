import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x + identity)
        return x


class DAPPM(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DAPPM, self).__init__()
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        )
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(5, 2, 2),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(9, 4, 4),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(17, 8, 8),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        )
        self.process1 = ConvBNReLU(mid_channels, mid_channels, kernel_size=3)
        self.process2 = ConvBNReLU(mid_channels, mid_channels, kernel_size=3)
        self.process3 = ConvBNReLU(mid_channels, mid_channels, kernel_size=3)
        self.process4 = ConvBNReLU(mid_channels, mid_channels, kernel_size=3)
        self.compress = ConvBNReLU(mid_channels * 5, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        size = x.size()[2:]
        s0 = self.scale0(x)
        s1 = self.process1(F.interpolate(self.scale1(x), size, mode='bilinear', align_corners=True) + s0)
        s2 = self.process2(F.interpolate(self.scale2(x), size, mode='bilinear', align_corners=True) + s1)
        s3 = self.process3(F.interpolate(self.scale3(x), size, mode='bilinear', align_corners=True) + s2)
        s4 = self.process4(F.interpolate(self.scale4(x), size, mode='bilinear', align_corners=True) + s3)
        x = self.compress(torch.cat([s0, s1, s2, s3, s4], dim=1))
        return x


class DDRNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, planes=32):
        """
        DDRNet-23-slim: Deep Dual-Resolution Network
        Paper: https://arxiv.org/abs/2101.06085

        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            num_classes: Number of output classes
                        - 1 for binary segmentation
                        - >=2 for multi-class
            planes: Base channel width (32 for slim, 64 for full)
        """
        super(DDRNet, self).__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, planes, stride=2),
            ConvBNReLU(planes, planes, stride=2),
        )
        self.layer1 = self._make_layer(planes, planes, 2)
        self.layer2 = self._make_layer(planes, planes * 2, 2, stride=2)

        self.high_branch = self._make_layer(planes * 2, planes * 2, 2)
        self.low_branch = self._make_layer(planes * 2, planes * 4, 2, stride=2)

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, 1, bias=False),
            nn.BatchNorm2d(planes * 2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(planes * 2, planes * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(planes * 4)
        )

        self.high_branch2 = self._make_layer(planes * 2, planes * 2, 2)
        self.low_branch2 = self._make_layer(planes * 4, planes * 8, 2, stride=2)

        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, 1, bias=False),
            nn.BatchNorm2d(planes * 2)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(planes * 2, planes * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(planes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(planes * 8)
        )

        self.dappm = DAPPM(planes * 16, planes * 4, planes * 4)
        self.high_to_low = self._make_layer(planes * 8, planes * 16, 1, stride=2)

        self.final_layer = nn.Sequential(
            ConvBNReLU(planes * 4, planes * 4),
            nn.Conv2d(planes * 4, num_classes, 1)
        )
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        size = x.size()[2:]

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x_h = self.high_branch(x)
        x_l = self.low_branch(x)

        x_h = self.relu(x_h + F.interpolate(self.compression3(x_l),
                         x_h.size()[2:], mode='bilinear', align_corners=True))
        x_l = self.relu(x_l + self.down3(x_h))

        x_h = self.high_branch2(x_h)
        x_l = self.low_branch2(x_l)

        x_h = self.relu(x_h + F.interpolate(self.compression4(x_l),
                         x_h.size()[2:], mode='bilinear', align_corners=True))
        x_l = self.relu(x_l + self.down4(x_h))

        x_l = self.high_to_low(x_l)
        x_l = self.dappm(x_l)
        x_l = F.interpolate(x_l, x_h.size()[2:], mode='bilinear', align_corners=True)

        x = self.final_layer(x_h + x_l)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


def build_model(num_classes=1):
    return DDRNet(in_channels=3, num_classes=num_classes, planes=32)