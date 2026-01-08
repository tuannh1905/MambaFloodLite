import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    def __init__(self):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(3, 13, kernel_size=3, stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.pool(x)
        return torch.cat([conv_out, pool_out], dim=1)


class BottleneckEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling=False, dilated=False, 
                 asymmetric=False, normal=False, drate=0.1):
        super(BottleneckEncoder, self).__init__()
        
        self.downsampling = downsampling
        internal_channels = out_channels // 4
        
        # Main branch
        if downsampling:
            self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=2, 
                                   stride=2, padding=0, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1, 
                                   stride=1, padding=0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(internal_channels, momentum=0.1)
        self.prelu1 = nn.PReLU(internal_channels)
        
        # Middle convolution
        if normal:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, 
                                   stride=1, padding=1, bias=True)
        elif asymmetric:
            self.conv2a = nn.Conv2d(internal_channels, internal_channels, kernel_size=(5, 1), 
                                    stride=1, padding=(2, 0), bias=False)
            self.conv2b = nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, 5), 
                                    stride=1, padding=(0, 2), bias=True)
        elif dilated:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, 
                                   stride=1, padding=dilated, dilation=dilated, bias=True)
        
        self.bn2 = nn.BatchNorm2d(internal_channels, momentum=0.1)
        self.prelu2 = nn.PReLU(internal_channels)
        
        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, 
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.dropout = nn.Dropout2d(p=drate)
        
        self.prelu_out = nn.PReLU(out_channels)
        
        # Skip connection
        if downsampling:
            self.pool_skip = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pad_channels = out_channels - in_channels
        
        self.asymmetric = asymmetric
        self.normal = normal
        self.dilated = dilated
        
    def forward(self, x):
        # Skip connection
        if self.downsampling:
            skip = self.pool_skip(x)
            if self.pad_channels > 0:
                # Pad channels: (batch, channels, height, width)
                # Create zero tensor with extra channels
                batch, channels, height, width = skip.shape
                padding = torch.zeros(batch, self.pad_channels, height, width, 
                                     device=skip.device, dtype=skip.dtype)
                skip = torch.cat([skip, padding], dim=1)
        else:
            skip = x
        
        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        
        if self.asymmetric:
            out = self.conv2a(out)
            out = self.conv2b(out)
        else:
            out = self.conv2(out)
        
        out = self.bn2(out)
        out = self.prelu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        
        out = out + skip
        out = self.prelu_out(out)
        
        return out


class BottleneckDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, upsampling=False, normal=False):
        super(BottleneckDecoder, self).__init__()
        
        self.upsampling = upsampling
        internal_channels = out_channels // 4
        
        # Skip connection
        if upsampling:
            self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                       stride=1, padding=0, bias=False)
            self.upsample_skip = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Main branch
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1, 
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels, momentum=0.1)
        self.prelu1 = nn.PReLU(internal_channels)
        
        if upsampling:
            self.conv2 = nn.ConvTranspose2d(internal_channels, internal_channels, 
                                            kernel_size=3, stride=2, padding=1, 
                                            output_padding=1, bias=True)
        elif normal:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, 
                                   stride=1, padding=1, bias=True)
        
        self.bn2 = nn.BatchNorm2d(internal_channels, momentum=0.1)
        self.prelu2 = nn.PReLU(internal_channels)
        
        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, 
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=0.1)
        
        self.relu_out = nn.ReLU()
        
        self.normal = normal
        
    def forward(self, x):
        # Skip connection
        if self.upsampling:
            skip = self.conv_skip(x)
            skip = self.upsample_skip(skip)
        else:
            skip = x
        
        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        
        if self.upsampling or self.normal:
            out = self.conv2(out)
        
        out = self.bn2(out)
        out = self.prelu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = out + skip
        out = self.relu_out(out)
        
        return out


class ENET(nn.Module):
    def __init__(self, num_classes=1):
        super(ENET, self).__init__()
        
        print('. . . . .Building ENet. . . . .')
        
        self.initial = InitialBlock()
        
        # Stage 1
        self.bottleneck1_0 = BottleneckEncoder(16, 64, downsampling=True, normal=True, drate=0.01)
        self.bottleneck1_1 = BottleneckEncoder(64, 64, normal=True, drate=0.01)
        self.bottleneck1_2 = BottleneckEncoder(64, 64, normal=True, drate=0.01)
        self.bottleneck1_3 = BottleneckEncoder(64, 64, normal=True, drate=0.01)
        self.bottleneck1_4 = BottleneckEncoder(64, 64, normal=True, drate=0.01)
        
        # Stage 2
        self.bottleneck2_0 = BottleneckEncoder(64, 128, downsampling=True, normal=True)
        self.bottleneck2_1 = BottleneckEncoder(128, 128, normal=True)
        self.bottleneck2_2 = BottleneckEncoder(128, 128, dilated=2)
        self.bottleneck2_3 = BottleneckEncoder(128, 128, asymmetric=True)
        self.bottleneck2_4 = BottleneckEncoder(128, 128, dilated=4)
        self.bottleneck2_5 = BottleneckEncoder(128, 128, normal=True)
        self.bottleneck2_6 = BottleneckEncoder(128, 128, dilated=8)
        self.bottleneck2_7 = BottleneckEncoder(128, 128, asymmetric=True)
        self.bottleneck2_8 = BottleneckEncoder(128, 128, dilated=16)
        
        # Stage 3
        self.bottleneck3_0 = BottleneckEncoder(128, 128, normal=True)
        self.bottleneck3_1 = BottleneckEncoder(128, 128, dilated=2)
        self.bottleneck3_2 = BottleneckEncoder(128, 128, asymmetric=True)
        self.bottleneck3_3 = BottleneckEncoder(128, 128, dilated=4)
        self.bottleneck3_4 = BottleneckEncoder(128, 128, normal=True)
        self.bottleneck3_5 = BottleneckEncoder(128, 128, dilated=8)
        self.bottleneck3_6 = BottleneckEncoder(128, 128, asymmetric=True)
        self.bottleneck3_7 = BottleneckEncoder(128, 128, dilated=16)
        
        # Stage 4
        self.bottleneck4_0 = BottleneckDecoder(128, 64, upsampling=True)
        self.bottleneck4_1 = BottleneckDecoder(64, 64, normal=True)
        self.bottleneck4_2 = BottleneckDecoder(64, 64, normal=True)
        
        # Stage 5
        self.bottleneck5_0 = BottleneckDecoder(64, 16, upsampling=True)
        self.bottleneck5_1 = BottleneckDecoder(16, 16, normal=True)
        
        # Final upsampling
        self.final_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, 
                                             stride=2, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        print('. . . . .Build Completed. . . . .')
        
    def forward(self, x):
        x = self.initial(x)
        
        # Stage 1
        x = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)
        
        # Stage 2
        x = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)
        
        # Stage 3
        x = self.bottleneck3_0(x)
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)
        
        # Stage 4
        x = self.bottleneck4_0(x)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)
        
        # Stage 5
        x = self.bottleneck5_0(x)
        x = self.bottleneck5_1(x)
        
        # Final
        x = self.final_conv(x)
        x = self.sigmoid(x)
        
        return x

def build_model(num_classes=1):
    return ENET(num_classes=num_classes)