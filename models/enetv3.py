# Model K + LDA Attention + Gated Skip Connections
# LDA được cắm sau stage 2 và stage 3
# Gated Skip Connections truyền thông tin từ Stage 1 và Stage 2 sang Decoder
# Không thay đổi bất kỳ bottleneck nào

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Helpers (Không đổi)
# ─────────────────────────────────────────────

def _max_unpool(unpool_layer, x, indices, output_size):
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    out = unpool_layer(x, indices, output_size=output_size)
    torch.use_deterministic_algorithms(prev)
    return out


def _depthwise_separable_conv(channels, kernel_size=3, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding,
                  dilation=dilation, groups=channels, bias=False),
        nn.BatchNorm2d(channels), nn.PReLU(),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(channels), nn.PReLU()
    )


def _depthwise_separable_conv_asymmetric(channels, kernel_size=5, padding=2):
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                  padding=(padding, 0), groups=channels, bias=False),
        nn.BatchNorm2d(channels), nn.PReLU(),
        nn.Conv2d(channels, channels, kernel_size=(1, kernel_size),
                  padding=(0, padding), groups=channels, bias=False),
        nn.BatchNorm2d(channels), nn.PReLU(),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(channels), nn.PReLU()
    )


# ─────────────────────────────────────────────
# Attention module (LDA - Không đổi)
# ─────────────────────────────────────────────

class LDA(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.shared = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.PReLU(),
        )

        self.fc_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.fc_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        xh = self.pool_h(x)
        xw = self.pool_w(x).permute(0, 1, 3, 2)
        combined = torch.cat([xh, xw], dim=2)
        combined = self.shared(combined)
        ah, aw = torch.split(combined, [H, W], dim=2)
        aw = aw.permute(0, 1, 3, 2)
        attn_h = self.sigmoid(self.fc_h(ah))
        attn_w = self.sigmoid(self.fc_w(aw))
        return x * attn_h * attn_w


# ─────────────────────────────────────────────
# Gated Skip Connection (NEW cải tiến)
# ─────────────────────────────────────────────

class GatedSkipConnection(nn.Module):
    """Gated Skip Connection 
    Sử dụng thông tin từ Decoder để lọc (gate) các đặc trưng từ Encoder 
    trước khi cộng vào để khôi phục ranh giới.
    """
    def __init__(self, enc_channels, dec_channels):
        super().__init__()
        # Để đảm bảo số channel enc khớp với dec nếu có sai lệch
        self.adapt_conv = nn.Sequential(
            nn.Conv2d(enc_channels, dec_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dec_channels)
        )
        # Nhánh tạo Gate
        self.gate = nn.Sequential(
            nn.Conv2d(dec_channels * 2, dec_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.Sigmoid()
        )

    def forward(self, enc, dec):
        enc_feat = self.adapt_conv(enc)
        # Nối enc và dec để tạo mặt nạ lọc
        cat = torch.cat([enc_feat, dec], dim=1)
        mask = self.gate(cat)
        # Chỉ lấy những phần đặc trưng enc mà gate cho phép
        return dec + (mask * enc_feat)


# ─────────────────────────────────────────────
# ENet blocks (Không đổi)
# ─────────────────────────────────────────────

class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.main_branch = nn.Conv2d(in_channels, out_channels - in_channels,
                                     kernel_size=3, stride=2, padding=1, bias=False)
        self.ext_branch = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_activation = nn.PReLU()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat([main, ext], dim=1)
        return self.out_activation(self.batch_norm(out))


class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.01):
        super().__init__()
        internal_channels = out_channels // 4
        self.main_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.main_avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(internal_channels), nn.PReLU()
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(internal_channels), nn.PReLU()
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.PReLU()
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()
        self.out_channels = out_channels

    def _pad_channels(self, x, target_channels):
        n, ch, h, w = x.size()
        pad_ch = target_channels - ch
        if pad_ch > 0:
            padding = torch.zeros(n, pad_ch, h, w, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        return x

    def forward(self, x):
        max_out, max_indices = self.main_maxpool(x)
        avg_out = self.main_avgpool(x)
        max_out = self._pad_channels(max_out, self.out_channels)
        avg_out = self._pad_channels(avg_out, self.out_channels)
        main = max_out + avg_out
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(main + ext), max_indices


class RegularBottleneck(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, dilation=1,
                 asymmetric=False, dropout_prob=0.1):
        super().__init__()
        internal_channels = channels // 4
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels), nn.PReLU()
        )
        if asymmetric:
            self.ext_conv2 = _depthwise_separable_conv_asymmetric(
                internal_channels, kernel_size=kernel_size, padding=padding)
        else:
            self.ext_conv2 = _depthwise_separable_conv(
                internal_channels, kernel_size=kernel_size,
                padding=padding, dilation=dilation)
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.PReLU()
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(main + ext)


class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super().__init__()
        internal_channels = in_channels // 4
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.main_unpool = nn.MaxUnpool2d(kernel_size=2)
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels), nn.PReLU()
        )
        self.ext_tconv = nn.ConvTranspose2d(internal_channels, internal_channels,
                                            kernel_size=2, stride=2, bias=False)
        self.ext_tconv_bn = nn.BatchNorm2d(internal_channels)
        self.ext_tconv_act = nn.PReLU()
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()

    def forward(self, x, max_indices, output_size):
        main = self.main_conv(x)
        main = _max_unpool(self.main_unpool, main, max_indices, output_size)
        ext = self.ext_conv1(x)
        ext = self.ext_tconv(ext, output_size=output_size)
        ext = self.ext_tconv_bn(ext)
        ext = self.ext_tconv_act(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(main + ext)


class RegularBottleneckDecoder(nn.Module):
    def __init__(self, channels, dropout_prob=0.1):
        super().__init__()
        internal_channels = channels // 4
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels), nn.PReLU()
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(internal_channels), nn.PReLU()
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.PReLU()
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(main + ext)


# ─────────────────────────────────────────────
# Model cải tiến với Gated Skip Connections
# ─────────────────────────────────────────────

class ENetModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # Encoder Stage 1
        self.initial_block = InitialBlock(in_channels, out_channels=16)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, dropout_prob=0.01)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01)

        # Encoder Stage 2
        self.downsample2_0 = DownsamplingBottleneck(64, 128, dropout_prob=0.1)
        self.regular2_1  = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated2_2  = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.dilated2_4  = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1)
        self.regular2_5  = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated2_6  = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.dilated2_8  = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)

        # Encoder Stage 3
        self.regular3_0  = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated3_1  = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1)
        self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.dilated3_3  = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1)
        self.regular3_4  = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated3_5  = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1)
        self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.dilated3_7  = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)

        # Decoder Stage 4
        self.attn_before_up4 = LDA(128)
        self.gated_skip_2 = GatedSkipConnection(enc_channels=128, dec_channels=128) # Kết nối từ cuối Stage 2
        self.upsample4_0  = UpsamplingBottleneck(128, 64, dropout_prob=0.1)
        self.regular4_1   = RegularBottleneckDecoder(64, dropout_prob=0.1)
        self.regular4_2   = RegularBottleneckDecoder(64, dropout_prob=0.1)

        # Decoder Stage 5
        self.attn_before_up5 = LDA(64)
        self.gated_skip_1 = GatedSkipConnection(enc_channels=64, dec_channels=64)  # Kết nối từ cuối Stage 1
        self.upsample5_0  = UpsamplingBottleneck(64, 16, dropout_prob=0.1)
        self.regular5_1   = RegularBottleneckDecoder(16, dropout_prob=0.1)

        self.transposed_conv = nn.ConvTranspose2d(
            16, num_classes, kernel_size=3, stride=2, padding=1,
            output_padding=1, bias=False
        )

    def forward(self, x):
        # Initial
        x = self.initial_block(x)
        stage1_size = x.size()

        # Stage 1
        x, max_idx1 = self.downsample1_0(x)
        x = self.regular1_1(x); x = self.regular1_2(x)
        x = self.regular1_3(x); x = self.regular1_4(x)
        feat_stage1 = x # Lưu đặc trưng stage 1 để skip

        # Stage 2
        stage2_size = x.size()
        x, max_idx2 = self.downsample2_0(x)
        x = self.regular2_1(x);   x = self.dilated2_2(x)
        x = self.asymmetric2_3(x); x = self.dilated2_4(x)
        x = self.regular2_5(x);   x = self.dilated2_6(x)
        x = self.asymmetric2_7(x); x = self.dilated2_8(x)
        feat_stage2 = x # Lưu đặc trưng stage 2 để skip

        # Stage 3
        x = self.regular3_0(x);   x = self.dilated3_1(x)
        x = self.asymmetric3_2(x); x = self.dilated3_3(x)
        x = self.regular3_4(x);   x = self.dilated3_5(x)
        x = self.asymmetric3_6(x); x = self.dilated3_7(x)

        # Decoder Stage 4
        x = self.attn_before_up4(x)
        x = self.gated_skip_2(feat_stage2, x) # Gated Skip từ Stage 2
        x = self.upsample4_0(x, max_idx2, output_size=stage2_size)
        x = self.regular4_1(x); x = self.regular4_2(x)

        # Decoder Stage 5
        x = self.attn_before_up5(x)
        x = self.gated_skip_1(feat_stage1, x) # Gated Skip từ Stage 1
        x = self.upsample5_0(x, max_idx1, output_size=stage1_size)
        x = self.regular5_1(x)

        return self.transposed_conv(x)


def build_model(num_classes=1):
    return ENetModel(in_channels=3, num_classes=num_classes)


# ─────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model(num_classes=1)
    model.eval()

    dummy = torch.randn(2, 3, 256, 512)
    with torch.no_grad():
        out = model(dummy)

    total = sum(p.numel() for p in model.parameters())
    gated = (sum(p.numel() for p in model.gated_skip_1.parameters()) + 
             sum(p.numel() for p in model.gated_skip_2.parameters()))

    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(out.shape)}")
    print(f"Params (total)      : {total:,}")
    print(f"Params (Gated Skip) : {gated:,} ({100*gated/total:.2f}%)")