import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1
        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = torch.cat(out_list, dim=1)
        return out

class STDCNet813(nn.Module):
    def __init__(self, base=64, layers=[2, 2, 2], block_num=4):
        super(STDCNet813, self).__init__()
        self.features = self._make_layers(base, layers, block_num, CatBottleneck)
        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]
        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2, i+1)), base*int(math.pow(2, i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2, i+2)), base*int(math.pow(2, i+2)), block_num, 1))
        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        return feat2, feat4, feat8, feat16, feat32

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        size_array = [int(s) for s in feat.size()[2:]]
        atten = torch.nn.functional.avg_pool2d(feat, size_array)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

class ContextPath(nn.Module):
    def __init__(self, input_size=512):
        super(ContextPath, self).__init__()
        self.backbone = STDCNet813()
        self.arm16 = AttentionRefinementModule(512, 128)
        inplanes = 1024
        self.arm32 = AttentionRefinementModule(inplanes, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)
        self.input_size = input_size
        if self.input_size == 512:
            self.H8, self.W8 = 64, 128
            self.H16, self.W16 = 32, 64
            self.H32, self.W32 = 16, 32
        elif self.input_size == 1024:
            self.H8, self.W8 = 128, 256
            self.H16, self.W16 = 64, 128
            self.H32, self.W32 = 32, 64
        elif self.input_size == 720:
            self.H8, self.W8 = 90, 120
            self.H16, self.W16 = 45, 60
            self.H32, self.W32 = 23, 30
        else:
            self.H8 = input_size // 8
            self.W8 = input_size // 4
            self.H16 = input_size // 16
            self.W16 = input_size // 8
            self.H32 = input_size // 32
            self.W32 = input_size // 16

    def forward(self, x):
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        size_array = [int(s) for s in feat32.size()[2:]]
        avg = torch.nn.functional.avg_pool2d(feat32, size_array)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (self.H32, self.W32), mode='nearest')
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (self.H16, self.W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)
        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (self.H8, self.W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)
        return feat2, feat4, feat8, feat16, feat16_up, feat32_up

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan//4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        size_array = [int(s) for s in feat.size()[2:]]
        atten = torch.nn.functional.avg_pool2d(feat, size_array)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

class BiSeNetSTDCModel(nn.Module):
    def __init__(self, num_classes=1, input_size=512):
        super(BiSeNetSTDCModel, self).__init__()
        self.input_size = input_size
        self.cp = ContextPath(input_size=self.input_size)
        conv_out_inplanes = 128
        sp8_inplanes = 256
        inplane = sp8_inplanes + conv_out_inplanes
        self.ffm = FeatureFusionModule(inplane, 256)
        self.conv_out = BiSeNetOutput(256, 256, num_classes)
        if self.input_size == 512:
            self.H, self.W = 512, 1024
        elif self.input_size == 1024:
            self.H, self.W = 1024, 2048
        elif self.input_size == 720:
            self.H, self.W = 720, 960
        else:
            self.H, self.W = input_size, input_size * 2

    def forward(self, x):
        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        feat_out = self.conv_out(feat_fuse)
        feat_out = F.interpolate(feat_out, (self.H, self.W), mode='nearest')
        return feat_out

def build_model(num_classes=1):
    return BiSeNetSTDCModel(num_classes=num_classes)