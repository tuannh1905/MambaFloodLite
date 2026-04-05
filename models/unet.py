import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNetModel(nn.Module):
    # ✓ ĐÃ SỬA: Thêm input_size=256
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256, 512], input_size=256):
        """
        UNet model for segmentation
        
        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            num_classes: Number of output classes
                        - 1 for binary segmentation (output: 1 channel with sigmoid)
                        - >=2 for multi-class (output: num_classes channels with softmax)
            features: Feature map sizes for each encoder/decoder level
            input_size: Input image size (MUST be divisible by 16 due to 4 MaxPool layers)
        """
        super(UNetModel, self).__init__()
        
        self.num_classes = num_classes
        
        # ✓ KIỂM TRA TOÁN HỌC: U-Net có 4 lần downsample (MaxPool2d stride 2) -> 2^4 = 16
        if input_size % 16 != 0:
            raise ValueError(f"UNet yêu cầu input_size chia hết cho 16. Kích thước {input_size} không hợp lệ.")
        
        self.inc = DoubleConv(in_channels, features[0])
        
        self.down1 = DownSampling(features[0], features[1])
        self.down2 = DownSampling(features[1], features[2])
        self.down3 = DownSampling(features[2], features[3])
        
        self.bottleneck = DownSampling(features[3], features[3] * 2)
        
        self.up1 = UpSampling(features[3] * 2, features[3])
        self.up2 = UpSampling(features[3], features[2])
        self.up3 = UpSampling(features[2], features[1])
        self.up4 = UpSampling(features[1], features[0])
        
        # Output layer with num_classes channels
        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# ✓ ĐÃ SỬA: Hàm build_model nhận thêm input_size
def build_model(num_classes=1, input_size=256):
    return UNetModel(in_channels=3, num_classes=num_classes, input_size=input_size)