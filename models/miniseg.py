import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        return self.act(self.bn(self.conv(input)))

class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DilatedParallelConvBlockD2, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv1 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, dilation=1, groups=out_planes, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=2, dilation=2, groups=out_planes, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        output = self.conv0(input)
        return self.bn(self.conv1(output) + self.conv2(output))

class DilatedParallelConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DilatedParallelConvBlock, self).__init__()
        inter_planes = out_planes // 4
        self.conv1x1_down = nn.Conv2d(in_planes, inter_planes, 1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=1, dilation=1, groups=inter_planes, bias=False)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=2, dilation=2, groups=inter_planes, bias=False)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=4, dilation=4, groups=inter_planes, bias=False)
        self.conv4 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=8, dilation=8, groups=inter_planes, bias=False)
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1)
        self.conv1x1_fuse = nn.Conv2d(out_planes, out_planes, 1, padding=0, groups=4, bias=False)
        self.attention = nn.Conv2d(out_planes, 4, 1, padding=0, groups=4, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        output = self.conv1x1_down(input)
        d1 = self.conv1(output)
        d2 = self.conv2(output)
        d3 = self.conv3(output)
        d4 = self.conv4(output)
        p = self.pool(output)
        d1, d2, d3, d4 = d1 + p, d1 + p + d2, d1 + p + d2 + d3, d1 + p + d2 + d3 + d4
        
        att = torch.sigmoid(self.attention(torch.cat([d1, d2, d3, d4], 1)))
        d1 = d1 + d1 * att[:, 0].unsqueeze(1)
        d2 = d2 + d2 * att[:, 1].unsqueeze(1)
        d3 = d3 + d3 * att[:, 2].unsqueeze(1)
        d4 = d4 + d4 * att[:, 3].unsqueeze(1)
        
        output = self.conv1x1_fuse(torch.cat([d1, d2, d3, d4], 1))
        return self.act(self.bn(output))

class DownsamplerBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(DownsamplerBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, 1, stride=1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(out_planes, out_planes, 5, stride=stride, padding=2, groups=out_planes, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        return self.act(self.bn(self.conv1(self.conv0(input))))

def split(x):
    c = x.size(1)
    c1 = c // 2
    return x[:, :c1, :, :].contiguous(), x[:, c1:, :, :].contiguous()

class MiniSeg(nn.Module):
    def __init__(self, num_classes=1, P1=2, P2=3, P3=8, P4=6, aux=False):
        super(MiniSeg, self).__init__()
        self.D1, self.D2, self.D3, self.D4 = P1//2, P2//2, P3//2, P4//2
        self.aux = aux

        # Level 1
        self.long1, self.down1 = DownsamplerBlock(3, 8, stride=2), ConvBlock(3, 8, stride=2)
        self.level1 = nn.ModuleList([ConvBlock(8, 8) for _ in range(P1)])
        self.level1_long = nn.ModuleList([DownsamplerBlock(8, 8, stride=1) for _ in range(self.D1)])
        self.cat1 = nn.Sequential(nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16))

        # Level 2
        self.long2, self.down2 = DownsamplerBlock(8, 24, stride=2), DilatedParallelConvBlock(8, 24, stride=2)
        self.level2 = nn.ModuleList([DilatedParallelConvBlock(24, 24) for _ in range(P2)])
        self.level2_long = nn.ModuleList([DownsamplerBlock(24, 24, stride=1) for _ in range(self.D2)])
        self.cat2 = nn.Sequential(nn.Conv2d(48, 48, 1, bias=False), nn.BatchNorm2d(48))

        # Level 3
        self.long3, self.down3 = DownsamplerBlock(24, 32, stride=2), DilatedParallelConvBlock(24, 32, stride=2)
        self.level3 = nn.ModuleList([DilatedParallelConvBlock(32, 32) for _ in range(P3)])
        self.level3_long = nn.ModuleList([DownsamplerBlock(32, 32, stride=1) for _ in range(self.D3)])
        self.cat3 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64))

        # Level 4
        self.long4, self.down4 = DownsamplerBlock(32, 64, stride=2), DilatedParallelConvBlock(32, 64, stride=2)
        self.level4 = nn.ModuleList([DilatedParallelConvBlock(64, 64) for _ in range(P4)])
        self.level4_long = nn.ModuleList([DownsamplerBlock(64, 64, stride=1) for _ in range(self.D4)])

        # Upsampling / Decoder
        self.up4_conv4 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.PReLU(64))
        self.up3_conv4, self.up3_conv3 = DilatedParallelConvBlockD2(64, 32), nn.Sequential(nn.Conv2d(32, 32, 1), nn.BatchNorm2d(32), nn.PReLU(32))
        self.up2_conv3, self.up2_conv2 = DilatedParallelConvBlockD2(32, 24), nn.Sequential(nn.Conv2d(24, 24, 1), nn.BatchNorm2d(24), nn.PReLU(24))
        self.up1_conv2, self.up1_conv1 = DilatedParallelConvBlockD2(24, 8), nn.Sequential(nn.Conv2d(8, 8, 1), nn.BatchNorm2d(8), nn.PReLU(8))

        self.pred1 = nn.Sequential(nn.Dropout2d(0.01), nn.Conv2d(8, num_classes, 1))

    def forward(self, input):
        # Forward pass giữ nguyên logic "long" và "down" của tác giả
        l1, d1 = self.long1(input), self.down1(input)
        add1 = d1 + l1
        for i, layer in enumerate(self.level1):
            d1 = layer(add1) + d1
            if i < self.D1: l1 = self.level1_long[i](add1) + l1
            add1 = d1 + l1
        l1_cat, r1_cat = split(self.cat1(torch.cat([l1, d1], 1)))

        l2, d2 = self.long2(l1_cat + l1), self.down2(r1_cat + d1)
        add2 = d2 + l2
        for i, layer in enumerate(self.level2):
            d2 = layer(add2) + d2
            if i < self.D2: l2 = self.level2_long[i](add2) + l2
            add2 = d2 + l2
        l2_cat, r2_cat = split(self.cat2(torch.cat([l2, d2], 1)))

        l3, d3 = self.long3(l2_cat + l2), self.down3(r2_cat + d2)
        add3 = d3 + l3
        for i, layer in enumerate(self.level3):
            d3 = layer(add3) + d3
            if i < self.D3: l3 = self.level3_long[i](add3) + l3
            add3 = d3 + l3
        l3_cat, r3_cat = split(self.cat3(torch.cat([l3, d3], 1)))

        l4, d4 = self.long4(l3_cat + l3), self.down4(r3_cat + d3)
        add4 = d4 + l4
        for i, layer in enumerate(self.level4):
            d4 = layer(add4) + d4
            if i < self.D4: l4 = self.level4_long[i](add4) + l4
            add4 = d4 + l4

        # Decoder path
        up4 = self.up4_conv4(d4)
        up3 = self.up3_act_func(up4, d3, self.up3_conv4, self.up3_conv3)
        up2 = self.up3_act_func(up3, d2, self.up2_conv3, self.up2_conv2)
        up1 = self.up3_act_func(up2, d1, self.up1_conv2, self.up1_conv1)

        final = F.interpolate(self.pred1(up1), input.size()[2:], mode='bilinear', align_corners=False)
        return final

    def up3_act_func(self, upper, current, conv_upper, conv_curr):
        upper_up = F.interpolate(upper, current.size()[2:], mode='bilinear', align_corners=False)
        return F.relu(conv_upper(upper_up) + conv_curr(current))

def build_model(num_classes=1):
    return MiniSeg(num_classes=num_classes, aux=False)