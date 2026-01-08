import torch
import torch.nn as nn
import torch.nn.functional as F
from .propose_model.model.encoder import TripleBranchEncoderBlock
from .propose_model.model.decoder import DecoderBlock
from .propose_model.module.eesp_bottleneck import EESPBottleneck


class MambaUNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.use_checkpoint = True
        self.num_classes = num_classes  # ✓ Store num_classes
        
        # Initial 1×1 convolution
        self.initial_conv = nn.Conv2d(in_channels, 16, kernel_size=1, bias=False)
        
        # Encoder với dual branches
        self.e1 = TripleBranchEncoderBlock(16, 16, kernel_size=3)
        self.e2 = TripleBranchEncoderBlock(16, 32, kernel_size=3)
        self.e3 = TripleBranchEncoderBlock(32, 64, kernel_size=3)
        self.e4 = TripleBranchEncoderBlock(64, 128, kernel_size=3)
        
        # EESP Bottleneck từ ESPNetv2
        self.bottleneck = EESPBottleneck(
            in_channels=128, 
            out_channels=256, 
            num_blocks=1,
            branches=4
        )
        
        # Decoder với CSAM
        self.d4 = DecoderBlock(256, 128, 128, kernel_size=3)
        self.d3 = DecoderBlock(128, 64, 64, kernel_size=3)
        self.d2 = DecoderBlock(64, 32, 32, kernel_size=3)
        self.d1 = DecoderBlock(32, 16, 16, kernel_size=3)
        
        # ✓ Reduce maps cho deep supervision - support multi-class
        self.reduce4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.reduce3 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.reduce2 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.reduce1 = nn.Conv2d(16, num_classes, kernel_size=1)
        
        # ✓ Final fusion - output num_classes channels
        self.out = nn.Conv2d(num_classes * 4, num_classes, kernel_size=1)
    
    def forward(self, x):
        _, _, H, W = x.shape  # Should be 512×512
        
        # Initial conv: (B, 3, 512, 512) -> (B, 16, 512, 512)
        x = self.initial_conv(x)
        
        # Encoder
        x, skip1 = self.e1(x)  # (B, 16, 256, 256)
        x, skip2 = self.e2(x)  # (B, 32, 128, 128)
        x, skip3 = self.e3(x)  # (B, 64, 64, 64)
        x, skip4 = self.e4(x)  # (B, 128, 32, 32)
        
        # Bottleneck: (B, 128, 32, 32) -> (B, 256, 32, 32)
        x = self.bottleneck(x)
        
        # Decoder
        d4 = self.d4(x, skip4)   # (B, 128, 64, 64)
        d3 = self.d3(d4, skip3)  # (B, 64, 128, 128)
        d2 = self.d2(d3, skip2)  # (B, 32, 256, 256)
        d1 = self.d1(d2, skip1)  # (B, 16, 512, 512)
        
        # Reduce maps - output shape: (B, num_classes, H, W)
        out4 = self.reduce4(d4)  # (B, num_classes, 64, 64)
        out3 = self.reduce3(d3)  # (B, num_classes, 128, 128)
        out2 = self.reduce2(d2)  # (B, num_classes, 256, 256)
        out1 = self.reduce1(d1)  # (B, num_classes, 512, 512)
        
        # Upsample tất cả về 512×512
        out4 = F.interpolate(out4, size=(H, W), mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=(H, W), mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)
        
        # Concatenate: (B, num_classes*4, H, W)
        concat_out = torch.cat([out1, out2, out3, out4], dim=1)
        
        # Final output: (B, num_classes, H, W)
        final_out = self.out(concat_out)
        
        return final_out


def build_model(num_classes=1):
    return MambaUNet(in_channels=3, num_classes=num_classes)