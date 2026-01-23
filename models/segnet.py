import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """Encoder block with conv-bn-relu pattern"""
    def __init__(self, in_channels, out_channels, num_convs):
        super(EncoderBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.convs = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.convs(x)


class DecoderBlock(nn.Module):
    """Decoder block with conv-bn-relu pattern (no ReLU in paper but code has it)"""
    def __init__(self, in_channels, out_channels, num_convs):
        super(DecoderBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            # Paper says no ReLU in decoder, but reference code has it
            layers.append(nn.ReLU(inplace=True))
        
        self.convs = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.convs(x)


class SegNetModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        """
        SegNet architecture following VGG16 encoder
        
        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            num_classes: Number of output classes
        """
        super(SegNetModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Encoder (following VGG16 architecture)
        # Stage 1: 2 convs, 64 filters
        self.encoder1 = EncoderBlock(in_channels, 64, num_convs=2)
        
        # Stage 2: 2 convs, 128 filters
        self.encoder2 = EncoderBlock(64, 128, num_convs=2)
        
        # Stage 3: 3 convs, 256 filters
        self.encoder3 = EncoderBlock(128, 256, num_convs=3)
        
        # Stage 4: 3 convs, 512 filters
        self.encoder4 = EncoderBlock(256, 512, num_convs=3)
        
        # Stage 5: 3 convs, 512 filters
        self.encoder5 = EncoderBlock(512, 512, num_convs=3)
        
        # Decoder (symmetric to encoder)
        # Stage 5: 3 convs, 512 filters
        self.decoder5 = DecoderBlock(512, 512, num_convs=3)
        
        # Stage 4: 3 convs, 256 filters (last conv reduces to 256)
        self.decoder4_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: 3 convs, 128 filters (last conv reduces to 128)
        self.decoder3_convs = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: 2 convs, 64 filters (last conv reduces to 64)
        self.decoder2_convs = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 2 convs, final output
        self.decoder1_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer (no batch norm, no relu)
        self.decoder1_conv2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Encoder with pooling indices
        x1 = self.encoder1(x)
        x1_pooled, indices1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        
        x2 = self.encoder2(x1_pooled)
        x2_pooled, indices2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        
        x3 = self.encoder3(x2_pooled)
        x3_pooled, indices3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        
        x4 = self.encoder4(x3_pooled)
        x4_pooled, indices4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        
        x5 = self.encoder5(x4_pooled)
        x5_pooled, indices5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)
        
        # Decoder with unpooling using stored indices
        d5 = F.max_unpool2d(x5_pooled, indices5, kernel_size=2, stride=2)
        d5 = self.decoder5(d5)
        
        d4 = F.max_unpool2d(d5, indices4, kernel_size=2, stride=2)
        d4 = self.decoder4_convs(d4)
        
        d3 = F.max_unpool2d(d4, indices3, kernel_size=2, stride=2)
        d3 = self.decoder3_convs(d3)
        
        d2 = F.max_unpool2d(d3, indices2, kernel_size=2, stride=2)
        d2 = self.decoder2_convs(d2)
        
        d1 = F.max_unpool2d(d2, indices1, kernel_size=2, stride=2)
        d1 = self.decoder1_conv1(d1)
        output = self.decoder1_conv2(d1)
        
        return output


def build_model(num_classes=1):
    """
    Build SegNet model
    
    Args:
        num_classes: Number of output classes
                    - 1 for binary segmentation
                    - >1 for multi-class segmentation
    """
    return SegNetModel(in_channels=3, num_classes=num_classes)