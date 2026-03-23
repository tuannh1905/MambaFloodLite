import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. Module ECA (Efficient Channel Attention)
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # Tính toán kích thước kernel tự động dựa trên số kênh (công thức gốc từ bài báo ECA-Net)
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [b, c, h, w]
        b, c, h, w = x.size()
        
        # Lấy global average pool [b, c, 1, 1]
        y = self.avg_pool(x)
        
        # Reshape để dùng Conv1d: [b, 1, c]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Attention weight [b, c, 1, 1]
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)

# 2. Module lõi từ ULite: Axial Depthwise Convolution
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

# 3. Block lai tạo mới (Axial + ERFNet) tích hợp ECA
class AxialECA_NonBottleneck_1D(nn.Module):
    def __init__(self, chann, dropprob, dilated, mixer_kernel=(5, 5)):
        super().__init__()
        
        # Nhánh 1
        self.pw1 = nn.Conv2d(chann, chann, kernel_size=1, bias=False)
        self.axial1 = AxialDW(chann, mixer_kernel=mixer_kernel, dilation=1)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        # Nhánh 2
        self.pw2 = nn.Conv2d(chann, chann, kernel_size=1, bias=False)
        self.axial2 = AxialDW(chann, mixer_kernel=mixer_kernel, dilation=dilated)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        # Tích hợp ECA vào cuối nhánh đặc trưng
        self.eca = ECA(channels=chann)
        
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        out = self.pw1(input)
        out = self.axial1(out)
        out = F.relu(self.bn1(out))

        out = self.pw2(out)
        out = self.axial2(out)
        out = self.bn2(out)

        # Áp dụng ECA để tinh chỉnh kênh trước khi dropout và residual add
        out = self.eca(out)

        if self.dropout.p != 0:
            out = self.dropout(out)
        
        return F.relu(out + input)

# 4. Các Block thu phóng cơ bản của ERFNet
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

# 5. Ráp thành mạng hoàn chỉnh
class ERF_AxialECA(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (5, 5) 

        # --- ENCODER ---
        self.initial_block = DownsamplerBlock(3, 16)
        self.enc_layers = nn.ModuleList()
        self.enc_layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5): 
            self.enc_layers.append(AxialECA_NonBottleneck_1D(64, 0.03, dilated=1, mixer_kernel=mk)) 

        self.enc_layers.append(DownsamplerBlock(64, 128))

        # Khối lõi đón Context với Dilation tăng dần
        for x in range(0, 2):
            self.enc_layers.append(AxialECA_NonBottleneck_1D(128, 0.3, dilated=2, mixer_kernel=mk))
            self.enc_layers.append(AxialECA_NonBottleneck_1D(128, 0.3, dilated=4, mixer_kernel=mk))
            self.enc_layers.append(AxialECA_NonBottleneck_1D(128, 0.3, dilated=8, mixer_kernel=mk))
            self.enc_layers.append(AxialECA_NonBottleneck_1D(128, 0.3, dilated=16, mixer_kernel=mk))

        # --- DECODER ---
        self.dec_layers = nn.ModuleList()
        self.dec_layers.append(UpsamplerBlock(128, 64))
        self.dec_layers.append(AxialECA_NonBottleneck_1D(64, 0, dilated=1, mixer_kernel=mk))
        self.dec_layers.append(AxialECA_NonBottleneck_1D(64, 0, dilated=1, mixer_kernel=mk))

        self.dec_layers.append(UpsamplerBlock(64, 16))
        self.dec_layers.append(AxialECA_NonBottleneck_1D(16, 0, dilated=1, mixer_kernel=mk))
        self.dec_layers.append(AxialECA_NonBottleneck_1D(16, 0, dilated=1, mixer_kernel=mk))

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
    return ERF_AxialECA(num_classes=num_classes)