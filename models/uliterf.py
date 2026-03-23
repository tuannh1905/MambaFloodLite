import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Module lõi từ ULite: Axial Depthwise Convolution
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        # Dùng padding='same' cực kỳ tiện lợi cho mọi dilation rate
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        # Cộng song song đặc trưng 2 trục vào ảnh gốc
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

# 2. Block lai tạo mới thay thế cho non_bottleneck_1d của ERFNet
class Axial_NonBottleneck_1D(nn.Module):
    def __init__(self, chann, dropprob, dilated, mixer_kernel=(5, 5)):
        super().__init__()
        
        # Nhánh 1: Pointwise (mix channels) -> AxialDW (mix spatial cục bộ) -> BN -> ReLU
        self.pw1 = nn.Conv2d(chann, chann, kernel_size=1, bias=False)
        self.axial1 = AxialDW(chann, mixer_kernel=mixer_kernel, dilation=1)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        # Nhánh 2: Pointwise -> AxialDW (mix spatial diện rộng với Dilation) -> BN
        self.pw2 = nn.Conv2d(chann, chann, kernel_size=1, bias=False)
        self.axial2 = AxialDW(chann, mixer_kernel=mixer_kernel, dilation=dilated)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        out = self.pw1(input)
        out = self.axial1(out)
        out = F.relu(self.bn1(out))

        out = self.pw2(out)
        out = self.axial2(out)
        out = self.bn2(out)

        if self.dropout.p != 0:
            out = self.dropout(out)
        
        # Residual connection
        return F.relu(out + input)

# 3. Các Block thu phóng cơ bản của ERFNet
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

# 4. Ráp thành mạng hoàn chỉnh
class ERF_AxialNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Kích thước kernel dọc/ngang. Tăng lên (5,5) vì AxialDW rất nhẹ!
        mk = (5, 5) 

        # --- ENCODER ---
        self.initial_block = DownsamplerBlock(3, 16)
        self.enc_layers = nn.ModuleList()
        self.enc_layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5): 
            self.enc_layers.append(Axial_NonBottleneck_1D(64, 0.03, dilated=1, mixer_kernel=mk)) 

        self.enc_layers.append(DownsamplerBlock(64, 128))

        # Khối lõi đón Context với Dilation tăng dần
        for x in range(0, 2):
            self.enc_layers.append(Axial_NonBottleneck_1D(128, 0.3, dilated=2, mixer_kernel=mk))
            self.enc_layers.append(Axial_NonBottleneck_1D(128, 0.3, dilated=4, mixer_kernel=mk))
            self.enc_layers.append(Axial_NonBottleneck_1D(128, 0.3, dilated=8, mixer_kernel=mk))
            self.enc_layers.append(Axial_NonBottleneck_1D(128, 0.3, dilated=16, mixer_kernel=mk))

        # --- DECODER ---
        self.dec_layers = nn.ModuleList()
        self.dec_layers.append(UpsamplerBlock(128, 64))
        self.dec_layers.append(Axial_NonBottleneck_1D(64, 0, dilated=1, mixer_kernel=mk))
        self.dec_layers.append(Axial_NonBottleneck_1D(64, 0, dilated=1, mixer_kernel=mk))

        self.dec_layers.append(UpsamplerBlock(64, 16))
        self.dec_layers.append(Axial_NonBottleneck_1D(16, 0, dilated=1, mixer_kernel=mk))
        self.dec_layers.append(Axial_NonBottleneck_1D(16, 0, dilated=1, mixer_kernel=mk))

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
    return ERF_AxialNet(num_classes=num_classes)