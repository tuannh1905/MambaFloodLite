import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# PICO-UNET V2: BẢN PAPER - TỐI ƯU HÓA CHO ESP32-S3
# ==============================================================================

class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), padding=(0, k_size//2), bias=False)
        self.hardsigmoid = nn.Hardsigmoid() 

    def forward(self, x):
        B, C, _, _ = x.shape 
        y = torch.mean(x, dim=[2, 3], keepdim=True)              
        y = y.permute(0, 2, 3, 1) 
        y = self.hardsigmoid(self.conv(y))                     
        y = y.permute(0, 3, 1, 2) 
        return x * y

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x

class SquareDW(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.dw(x))

class MultiScale_PFCU_DG_v2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Kernel Factorization: Biến 5x5, 7x7 thành các chuỗi 3x3
        self.branch_3 = SquareDW(dim, kernel_size=3)
        self.branch_5 = nn.Sequential(SquareDW(dim, 3), SquareDW(dim, 3))
        self.branch_7 = nn.Sequential(SquareDW(dim, 3), SquareDW(dim, 3), SquareDW(dim, 3))
        
        self.pw_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(dim)
        
        self.shuffle = ChannelShuffle(groups=4) # Nhóm 4 để trộn đặc trưng hiệu quả
        self.eca = ECABlock(dim) 
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        b3, b5, b7 = self.branch_3(x), self.branch_5(x), self.branch_7(x)
        fused = self.bn_fuse(self.pw_fuse(b3 + b5 + b7))
        fused = self.shuffle(fused)
        return self.eca(self.act(fused + x))

class EncoderBlock_v2(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pfcu_dg = MultiScale_PFCU_DG_v2(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))
        
        self.same_channels = (in_c == out_c)
        if not self.same_channels:
            self.pw = nn.Sequential(
                nn.Conv2d(in_c, out_c - in_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c - in_c)
            )
            self.down_pw = nn.MaxPool2d((2, 2))
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        feat = self.pfcu_dg(x)
        if self.same_channels:
            return self.act(self.down_pool(feat)), feat
        else:
            feat_pw = self.pw(feat)
            skip = torch.cat([feat, feat_pw], dim=1)
            x = self.act(torch.cat([self.down_pool(feat), self.down_pw(feat_pw)], dim=1))
            return x, skip

class LightInvertedBottleneck(nn.Module):
    def __init__(self, dim, expand_ratio=2):
        super().__init__()
        hid = dim * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(dim, hid, 1, bias=False),
            nn.BatchNorm2d(hid), nn.ReLU6(True),
            nn.Conv2d(hid, hid, 3, padding=1, groups=hid, bias=False),
            nn.BatchNorm2d(hid), nn.ReLU6(True),
            nn.Conv2d(hid, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
    def forward(self, x): return x + self.conv(x)

class NearestUpsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x): return self.refine(self.up(x))

class SimpleConcatFusion(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU6(True)
        )
    def forward(self, x, skip): return self.fuse(torch.cat([x, skip], dim=1))

class LightDecoderBlock_NoUAFM(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = NearestUpsample(in_c)
        self.fusion = SimpleConcatFusion(in_c, in_c, out_c)
        gc = max(out_c // 4, 4)
        self.refine = nn.Sequential(
            nn.Conv2d(out_c, gc, 1, bias=False), nn.BatchNorm2d(gc), nn.ReLU6(True),
            SquareDW(gc, 5), ECABlock(gc),
            nn.Conv2d(gc, out_c, 1, bias=False), nn.BatchNorm2d(out_c)
        )
        self.act = nn.ReLU6(True)
    def forward(self, x, skip):
        up = self.up(x)
        fused = self.fusion(up, skip)
        return self.act(self.refine(fused) + fused)

class PicoUNet_v2_Paper(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 16, 3, padding=1)
        self.e1 = EncoderBlock_v2(16, 32)
        self.e2 = EncoderBlock_v2(32, 64)
        self.e3 = EncoderBlock_v2(64, 128)
        self.e4 = EncoderBlock_v2(128, 128) # Cap tại 128
        self.bottleneck = LightInvertedBottleneck(128)
        self.d4 = LightDecoderBlock_NoUAFM(128, 128)
        self.d3 = LightDecoderBlock_NoUAFM(128, 64)
        self.d2 = LightDecoderBlock_NoUAFM(64, 32)
        self.d1 = LightDecoderBlock_NoUAFM(32, 16)
        self.conv_out = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        x = self.bottleneck(x)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        return self.conv_out(x)

# ==============================================================================
# 5. HÀM BUILD MODEL (TRẢ RA MODEL ĐỂ TRAINING)
# ==============================================================================
def build_model(num_classes=1):
    """
    Khởi tạo Pico-UNet v2 dành cho Paper nghiên cứu.
    Đã tối ưu hóa kernel 3x3 và giới hạn 128 channels.
    """
    model = PicoUNet_v2_Paper(num_classes=num_classes)
    return model

if __name__ == "__main__":
    # Test thử kích thước tensor
    model = build_model(num_classes=1)
    test_input = torch.randn(1, 3, 128, 128)
    output = model(test_input)
    print(f"Input: {test_input.shape}")
    print(f"Output: {output.shape}")
    
    # Kiểm tra tổng số tham số (Rất quan trọng cho Paper)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")