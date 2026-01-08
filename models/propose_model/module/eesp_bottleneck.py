import torch
import torch.nn as nn
import torch.nn.functional as F


class EESP(nn.Module):
    """
    EESP Module từ ESPNetv2 paper (ĐÚNG CHUẨN)
    
    Architecture:
    1. Group Pointwise reduce (GConv 1x1)
    2. Split -> K branches Depthwise Dilated Conv (dilation: 1,2,4,8,16)
    3. Hierarchical Feature Fusion (HFF) - cộng dồn
    4. Concat all branches
    5. Group Pointwise expand (GConv 1x1)  
    6. Residual connection
    """
    def __init__(self, in_channels, out_channels, stride=1, branches=5, reduce_ratio=4):
        super().__init__()
        
        assert out_channels % branches == 0, "out_channels must be divisible by branches"
        
        self.stride = stride
        self.branches = branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Reduced channels
        reduced_channels = out_channels // reduce_ratio
        branch_channels = out_channels // branches
        
        # 1. Group Pointwise reduce
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.PReLU(reduced_channels)
        )
        
        # 2. Depthwise Dilated Convolutions - K branches
        self.dw_branches = nn.ModuleList()
        for i in range(branches):
            dilation = 2 ** i  # 1, 2, 4, 8, 16
            self.dw_branches.append(
                nn.Conv2d(
                    reduced_channels,
                    reduced_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    dilation=dilation,
                    groups=reduced_channels,  # Depthwise
                    bias=False
                )
            )
        
        # 3. Branch outputs batch norm
        self.bn = nn.ModuleList([nn.BatchNorm2d(reduced_channels) for _ in range(branches)])
        
        # 4. Group Pointwise expand
        self.expand = nn.Sequential(
            nn.Conv2d(reduced_channels * branches, out_channels, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels)
        )
        
        # 5. Shortcut
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        # 1. Reduce
        x = self.reduce(x)
        
        # 2. Depthwise branches với HFF (Hierarchical Feature Fusion)
        outputs = []
        hff = 0  # Hierarchical accumulation
        
        for i, (dw, bn) in enumerate(zip(self.dw_branches, self.bn)):
            if i == 0:
                out = dw(x)
                out = bn(out)
                hff = out
            else:
                out = dw(x)
                out = bn(out)
                hff = hff + out  # HFF: cộng dồn
                out = hff
            
            outputs.append(out)
        
        # 3. Concatenate all branches
        x = torch.cat(outputs, dim=1)
        
        # 4. Expand
        x = self.expand(x)
        
        # 5. Residual
        x = x + self.shortcut(identity)
        
        return x


class EESPBottleneck(nn.Module):
    """
    Stack nhiều EESP modules cho bottleneck layer
    """
    def __init__(self, in_channels, out_channels, num_blocks=2, branches=4):
        super().__init__()
        
        # First EESP: in_channels -> out_channels
        self.blocks = nn.ModuleList([
            EESP(in_channels, out_channels, stride=1, branches=branches)
        ])
        
        # Remaining blocks: out_channels -> out_channels
        for _ in range(num_blocks - 1):
            self.blocks.append(
                EESP(out_channels, out_channels, stride=1, branches=branches)
            )
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    print("Testing EESP Module (ĐÚNG CHUẨN ESPNetv2):")
    print("="*60)
    
    # Test single EESP
    model = EESP(in_channels=128, out_channels=256, stride=1, branches=4)
    x = torch.randn(2, 128, 32, 32)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}\n")
    
    # Test EESPBottleneck
    print("Testing EESPBottleneck (2 blocks):")
    model2 = EESPBottleneck(in_channels=128, out_channels=256, num_blocks=2, branches=4)
    out2 = model2(x)
    print(f"Input: {x.shape} -> Output: {out2.shape}")
    
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"Parameters: {params2:,}")
    
    print("\n" + "="*60)
    print("EESP Module đặc điểm:")
    print("✓ Group Pointwise Convolution (reduce/expand)")
    print("✓ Depthwise Dilated Convolution branches (1,2,4,8...)")
    print("✓ Hierarchical Feature Fusion (HFF)")
    print("✓ Residual connection")
    print("="*60)