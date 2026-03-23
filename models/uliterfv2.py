import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Module Lõi của ULite: Axial Depthwise Convolution
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        # Dùng padding='same' cực kỳ tiện lợi cho mọi dilation rate
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        # Cộng song song đặc trưng 2 trục vào ảnh gốc
        return x + self.dw_h(x) + self.dw_w(x)

# 2. Attention Module: Light Spatial & Channel Attention (CBAM-inspired)
class AttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Channel Attention (Squeeze and Excitation-like)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False), # Giảm channels
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False), # Phục hồi channels
            nn.Sigmoid()
        )
        # Spatial Attention (Simple Convolution)
        self.sa = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        out = x * self.ca(x)
        # Spatial Attention
        out = out * self.sa(out)
        return out

# 3. Block trích xuất đặc trưng Đa tỷ lệ (Multiscale Dilated Axial)
class MultiscaleAxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel=(5,5)):
        super().__init__()
        # Ta dùng 3 nhánh song song với 3 dilation rate khác nhau
        self.axial1 = AxialDW(dim, mixer_kernel, dilation=1)
        self.axial2 = AxialDW(dim, mixer_kernel, dilation=2)
        self.axial3 = AxialDW(dim, mixer_kernel, dilation=4)
        
        # Pointwise để tổng hợp thông tin đa tỷ lệ
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        # Cộng đặc trưng từ 3 nhánh song song
        out = self.axial1(x) + self.axial2(x) + self.axial3(x)
        out = self.pw(out)
        return out

# 4. Block lai tạo mới cho ERFNet, tích hợp Attention và Multiscale
class MAG_NonBottleneck_1D(nn.Module):
    def __init__(self, chann, dropprob, mixer_kernel=(5, 5), use_multiscale=False):
        super().__init__()
        self.use_multiscale = use_multiscale
        
        # Nhánh 1: Pointwise (mix channels) -> AxialDW/MultiscaleAxial -> BN -> ReLU
        self.pw1 = nn.Conv2d(chann, chann, kernel_size=1, bias=False)
        if use_multiscale:
            self.axial_block = MultiscaleAxialDW(chann, mixer_kernel=mixer_kernel)
        else:
            self.axial_block = AxialDW(chann, mixer_kernel=mixer_kernel, dilation=1)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        # Nhánh 2: Pointwise -> AxialDW (mix spatial cục bộ) -> BN
        self.pw2 = nn.Conv2d(chann, chann, kernel_size=1, bias=False)
        self.axial2 = AxialDW(chann, mixer_kernel=mixer_kernel, dilation=1)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        # Module Attention và Dropout
        self.attn = AttentionModule(chann)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        out = self.pw1(input)
        out = self.axial_block(out)
        out = F.relu(self.bn1(out))

        out = self.pw2(out)
        out = self.axial2(out)
        out = self.bn2(out)

        # Áp dụng Attention
        out = self.attn(out)

        if self.dropout.p != 0:
            out = self.dropout(out)
        
        # Residual connection
        return F.relu(out + input)

# 5. Các Block thu phóng cơ bản của ERFNet
class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)

class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

# 6. Ráp thành mạng hoàn chỉnh
class MAG_DANet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Kích thước kernel dọc/ngang. Tăng lên (7,7) vì AxialDW rất nhẹ!
        mk = (7, 7) 

        # --- ENCODER ---
        self.initial_block = DownsamplerBlock(3, 16)
        self.enc_layers = nn.ModuleList()
        self.enc_layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5): 
            self.enc_layers.append(MAG_NonBottleneck_1D(64, 0.03, mixer_kernel=mk, use_multiscale=False)) 

        self.enc_layers.append(DownsamplerBlock(64, 128))

        # Khối lõi với Dilation đa tỷ lệ
        for x in range(0, 4): # Số block ít hơn nhưng mỗi block "chất" hơn
            self.enc_layers.append(MAG_NonBottleneck_1D(128, 0.3, mixer_kernel=mk, use_multiscale=True))

        # --- DECODER ---
        self.dec_layers = nn.ModuleList()
        self.dec_layers.append(UpsamplerBlock(128, 64))
        self.dec_layers.append(MAG_NonBottleneck_1D(64, 0, mixer_kernel=mk, use_multiscale=False))
        self.dec_layers.append(MAG_NonBottleneck_1D(64, 0, mixer_kernel=mk, use_multiscale=False))

        self.dec_layers.append(UpsamplerBlock(64, 16))
        self.dec_layers.append(MAG_NonBottleneck_1D(16, 0, mixer_kernel=mk, use_multiscale=False))
        self.dec_layers.append(MAG_NonBottleneck_1D(16, 0, mixer_kernel=mk, use_multiscale=False))

        # Head trả về trực tiếp logits đồng bộ với BCEDiceLoss
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        x = self.initial_block(input)
        
        for layer in self.enc_layers:
            x = layer(x)
            
        for layer in self.dec_layers:
            x = layer(x)
            
        x = self.output_conv(x)
        return x

def build_model(num_classes=1):
    return MAG_DANet(num_classes=num_classes)