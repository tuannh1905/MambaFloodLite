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
    """Decoder block with conv-bn-relu pattern"""
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
            layers.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


def _max_unpool(x, indices, kernel_size=2, stride=2):
    """
    Wrapper quanh F.max_unpool2d để bypass deterministic constraint.
    max_unpool2d chưa được PyTorch đăng ký là deterministic dù thực tế
    kết quả hoàn toàn deterministic (chỉ đặt giá trị vào đúng vị trí indices).
    Note: deterministic mode is temporarily disabled for this operation only.
    """
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    out = F.max_unpool2d(x, indices, kernel_size=kernel_size, stride=stride)
    torch.use_deterministic_algorithms(prev)
    return out


class SegNetModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, input_size=256):
        """
        SegNet architecture following VGG16 encoder.
        All blocks are symmetric between encoder and decoder
        per the original paper (Badrinarayanan et al., 2017).

        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            num_classes: Number of output classes
            input_size: Input image size (MUST be divisible by 32 due to 5 MaxPool layers)
        """
        super(SegNetModel, self).__init__()

        self.num_classes = num_classes
        
        # ✓ KIỂM TRA TOÁN HỌC: SegNet có 5 lớp MaxPool (2^5 = 32)
        if input_size % 32 != 0:
            raise ValueError(f"SegNet yêu cầu input_size chia hết cho 32. Kích thước {input_size} sẽ gây lỗi Unpool.")

        # Encoder (following VGG16 architecture)
        self.encoder1 = EncoderBlock(in_channels, 64,  num_convs=2)
        self.encoder2 = EncoderBlock(64,  128, num_convs=2)
        self.encoder3 = EncoderBlock(128, 256, num_convs=3)
        self.encoder4 = EncoderBlock(256, 512, num_convs=3)
        self.encoder5 = EncoderBlock(512, 512, num_convs=3)

        # Decoder (symmetric to encoder)
        self.decoder5 = DecoderBlock(512, 512, num_convs=3)

        self.decoder4_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.decoder3_convs = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

        self.decoder2_convs = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,  64, kernel_size=3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
        )

        # FIX: decoder1 now has 2 convs to match encoder1 (symmetric per paper)
        self.decoder1_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        # Final output layer — kernel_size=1 per paper
        self.decoder1_conv2 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
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

        # Decoder
        d5 = _max_unpool(x5_pooled, indices5)
        d5 = self.decoder5(d5)

        d4 = _max_unpool(d5, indices4)
        d4 = self.decoder4_convs(d4)

        d3 = _max_unpool(d4, indices3)
        d3 = self.decoder3_convs(d3)

        d2 = _max_unpool(d3, indices2)
        d2 = self.decoder2_convs(d2)

        d1 = _max_unpool(d2, indices1)
        d1 = self.decoder1_conv1(d1)
        output = self.decoder1_conv2(d1)

        return output


def build_model(num_classes=1, input_size=256):
    """
    Build SegNet model tương thích với file __init__.py mới
    """
    return SegNetModel(in_channels=3, num_classes=num_classes, input_size=input_size)