import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsamplerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0, use_prelu=False):
        super(DownsamplerBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, 
                             kernel_size=3, stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.act = nn.PReLU(out_channels) if use_prelu else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None
    
    def forward(self, x):
        output_conv = self.conv(x)
        output_pool = self.pool(x)
        output = torch.cat([output_conv, output_pool], dim=1)
        output = self.bn(output)
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.act(output)
        return output


class NonBottleneck1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 dropout_prob=0.0, use_prelu=False, dilation=1):
        super(NonBottleneck1D, self).__init__()
        
        assert in_channels == out_channels
        
        padding = (kernel_size - 1) // 2
        
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=(kernel_size, 1), 
                                   stride=1, padding=(padding, 0), bias=True)
        self.act1 = nn.PReLU(out_channels) if use_prelu else nn.ReLU(inplace=True)
        self.conv1x2 = nn.Conv2d(out_channels, out_channels, 
                                kernel_size=(1, kernel_size), 
                                stride=1, padding=(0, padding), bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.act2 = nn.PReLU(out_channels) if use_prelu else nn.ReLU(inplace=True)
        
        if dilation == 1:
            self.conv2x1 = nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=(kernel_size, 1), 
                                    stride=1, padding=(padding, 0), bias=True)
            self.act3 = nn.PReLU(out_channels) if use_prelu else nn.ReLU(inplace=True)
            self.conv2x2 = nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=(1, kernel_size), 
                                    stride=1, padding=(0, padding), bias=True)
        else:
            dilated_padding = padding * dilation
            self.conv2x1 = nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=(kernel_size, 1), 
                                    stride=1, padding=(dilated_padding, 0),
                                    dilation=(dilation, 1), bias=True)
            self.act3 = nn.PReLU(out_channels) if use_prelu else nn.ReLU(inplace=True)
            self.conv2x2 = nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=(1, kernel_size), 
                                    stride=1, padding=(0, dilated_padding),
                                    dilation=(1, dilation), bias=True)
        
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None
        self.act_final = nn.PReLU(out_channels) if use_prelu else nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        output = self.conv1x1_1(x)
        output = self.act1(output)
        output = self.conv1x2(output)
        output = self.bn1(output)
        output = self.act2(output)
        output = self.conv2x1(output)
        output = self.act3(output)
        output = self.conv2x2(output)
        output = self.bn2(output)
        if self.dropout is not None:
            output = self.dropout(output)
        output = output + identity
        output = self.act_final(output)
        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplerBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 
                                       kernel_size=3, stride=2, 
                                       padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.act(output)
        return output


class ERFNet_Encoder(nn.Module):
    def __init__(self, num_classes=20, use_prelu=False):
        super(ERFNet_Encoder, self).__init__()
        drop = 0.3
        
        self.downsample1 = DownsamplerBlock(3, 16, dropout_prob=0.0, use_prelu=use_prelu)
        self.downsample2 = DownsamplerBlock(16, 64, dropout_prob=drop/10.0, use_prelu=use_prelu)
        
        self.layer1 = nn.ModuleList()
        for i in range(5):
            self.layer1.append(NonBottleneck1D(64, 64, kernel_size=3, 
                                              dropout_prob=drop/10.0, 
                                              use_prelu=use_prelu, dilation=1))
        
        self.downsample3 = DownsamplerBlock(64, 128, dropout_prob=drop, use_prelu=use_prelu)
        
        self.layer2 = nn.ModuleList()
        for i in range(2):
            self.layer2.append(NonBottleneck1D(128, 128, kernel_size=3, 
                                              dropout_prob=drop, 
                                              use_prelu=use_prelu, dilation=2))
            self.layer2.append(NonBottleneck1D(128, 128, kernel_size=3, 
                                              dropout_prob=drop, 
                                              use_prelu=use_prelu, dilation=4))
            self.layer2.append(NonBottleneck1D(128, 128, kernel_size=3, 
                                              dropout_prob=drop, 
                                              use_prelu=use_prelu, dilation=8))
            self.layer2.append(NonBottleneck1D(128, 128, kernel_size=3, 
                                              dropout_prob=drop, 
                                              use_prelu=use_prelu, dilation=16))
        
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1, bias=True)
    
    def forward(self, x):
        output = self.downsample1(x)
        output = self.downsample2(output)
        for layer in self.layer1:
            output = layer(output)
        output = self.downsample3(output)
        for layer in self.layer2:
            output = layer(output)
        output = self.classifier(output)
        return output


class ERFNet(nn.Module):
    def __init__(self, num_classes=20, encoder_weights=None, use_prelu=False):
        super(ERFNet, self).__init__()
        
        self.encoder = ERFNet_Encoder(num_classes, use_prelu)
        
        if encoder_weights is not None:
            self.encoder.load_state_dict(torch.load(encoder_weights))
        
        self.encoder.classifier = nn.Identity()
        
        self.decoder = nn.ModuleList()
        self.decoder.append(UpsamplerBlock(128, 64))
        self.decoder.append(NonBottleneck1D(64, 64, kernel_size=3, 
                                           dropout_prob=0.0, 
                                           use_prelu=use_prelu, dilation=1))
        self.decoder.append(NonBottleneck1D(64, 64, kernel_size=3, 
                                           dropout_prob=0.0, 
                                           use_prelu=use_prelu, dilation=1))
        self.decoder.append(UpsamplerBlock(64, 16))
        self.decoder.append(NonBottleneck1D(16, 16, kernel_size=3, 
                                           dropout_prob=0.0, 
                                           use_prelu=use_prelu, dilation=1))
        self.decoder.append(NonBottleneck1D(16, 16, kernel_size=3, 
                                           dropout_prob=0.0, 
                                           use_prelu=use_prelu, dilation=1))
        
        self.final_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, 
                                            stride=2, padding=0, bias=True)
    
    def forward(self, x):
        output = self.encoder.downsample1(x)
        output = self.encoder.downsample2(output)
        for layer in self.encoder.layer1:
            output = layer(output)
        output = self.encoder.downsample3(output)
        for layer in self.encoder.layer2:
            output = layer(output)
        for layer in self.decoder:
            output = layer(output)
        output = self.final_conv(output)
        return output


class ERFNetModel(ERFNet):
    pass

def build_model(num_classes=1):
    return ERFNet(num_classes=num_classes, use_prelu=False)