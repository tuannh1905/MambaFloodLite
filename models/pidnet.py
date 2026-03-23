import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, no_relu=False):
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
        self.no_relu = no_relu

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        if not self.no_relu:
            x = self.relu(x)
        return x


class PagFM(nn.Module):
    """Pixel-attention-guided fusion module"""
    def __init__(self, in_channels, mid_channels, after_relu=False):
        super(PagFM, self).__init__()
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, input_size[2:], mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        attn = torch.sigmoid(torch.sum(x_k * y_q, dim=1, keepdim=True))
        x = attn * F.interpolate(y, input_size[2:], mode='bilinear', align_corners=False) + (1 - attn) * x
        return x


class Bag(nn.Module):
    """Boundary-attention-guided fusion module"""
    def __init__(self, in_channels, out_channels):
        super(Bag, self).__init__()
        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size=3)

    def forward(self, p, i, d):
        edge_attn = torch.sigmoid(d)
        x = edge_attn * p + (1 - edge_attn) * i
        x = self.conv(x)
        return x


class PIDNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, planes=32, ppm_planes=96, head_planes=128):
        """
        PIDNet: Parallel Interaction and Detail Prediction Network
        Paper: https://arxiv.org/abs/2206.02066

        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            num_classes: Number of output classes
                        - 1 for binary segmentation
                        - >=2 for multi-class
            planes: Base channel width (32 for PIDNet-S, 64 for PIDNet-M/L)
        """
        super(PIDNet, self).__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, planes, stride=2),
            ConvBNReLU(planes, planes, stride=2),
        )

        self.p_branch = nn.Sequential(
            self._make_layer(planes, planes, 2),
            self._make_layer(planes, planes * 2, 2, stride=2),
            self._make_layer(planes * 2, planes * 2, 2),
        )

        self.i_branch = nn.Sequential(
            self._make_layer(planes, planes, 2),
            self._make_layer(planes, planes * 2, 2, stride=2),
            self._make_layer(planes * 2, planes * 4, 2, stride=2),
        )

        self.d_branch = nn.Sequential(
            self._make_layer(planes, planes, 2),
            self._make_layer(planes, planes, 2, stride=2),
        )

        self.pag1 = PagFM(planes * 2, planes)
        self.pag2 = PagFM(planes * 2, planes)

        self.bag = Bag(planes * 2, planes * 2)

        self.ppm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(planes * 4, ppm_planes, kernel_size=1, padding=0),
        )

        self.head = nn.Sequential(
            ConvBNReLU(planes * 2, head_planes),
            nn.Conv2d(head_planes, num_classes, 1)
        )

        self.edge_head = nn.Sequential(
            ConvBNReLU(planes, planes),
            nn.Conv2d(planes, 1, 1)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        size = x.size()[2:]
        x = self.stem(x)

        x_p = self.p_branch(x)
        x_i = self.i_branch(x)
        x_d = self.d_branch(x)

        x_i_up = F.interpolate(x_i, x_p.size()[2:], mode='bilinear', align_corners=False)
        x_p = self.pag1(x_p, x_i_up)

        x_p_down = F.interpolate(x_p, x_i.size()[2:], mode='bilinear', align_corners=False)
        x_i_context = F.interpolate(
            self.ppm(x_i).expand_as(x_i), x_i.size()[2:], mode='bilinear', align_corners=False
        )
        x_i = x_i + x_i_context

        x_i_up = F.interpolate(x_i, x_p.size()[2:], mode='bilinear', align_corners=False)
        x_p = self.pag2(x_p, x_i_up)

        x_d_up = F.interpolate(x_d, x_p.size()[2:], mode='bilinear', align_corners=False)
        x = self.bag(x_p, x_i_up, x_d_up)

        x = self.head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=False)
        return x


def build_model(num_classes=1):
    return PIDNet(in_channels=3, num_classes=num_classes, planes=32,
                  ppm_planes=96, head_planes=128)