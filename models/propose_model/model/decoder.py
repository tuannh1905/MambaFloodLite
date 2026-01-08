import torch
import torch.nn as nn
from ..module.csam import CSAM

class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, reduction=2, kernel_size=3):  # Thêm kernel_size
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.csam = CSAM(skip_c)
        
        # Bottleneck: giảm channels
        hidden = (in_c + skip_c) // reduction
        
        self.pw1 = nn.Conv2d(in_c + skip_c, hidden, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.pw2 = nn.Conv2d(hidden, out_c, kernel_size=1, bias=False)
    
    def forward(self, x, skip):
        x = self.up(x)
        skip = self.csam(skip)
        x = torch.cat([skip, x], dim=1)
        
        x = self.pw1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pw2(x)
        
        return x