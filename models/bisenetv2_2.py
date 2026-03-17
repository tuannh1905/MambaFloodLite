import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat

# =========================================================================
# [CẢI TIẾN 1]: THU HẸP NHÁNH DETAIL BRANCH
# =========================================================================
class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch, self).__init__()
        # Giảm số kênh từ [64, 64, 128] xuống [16, 32, 64]
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 16, 3, stride=2),
            ConvBNReLU(16, 16, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(16, 32, 3, stride=2),
            ConvBNReLU(32, 32, 3, stride=1),
            ConvBNReLU(32, 32, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(32, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat

# =========================================================================
# [CẢI TIẾN 2]: ÉP CÂN SEMANTIC BRANCH (Giảm Base Channels & Exp Ratio)
# =========================================================================
class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        # Hạ base channel từ 16 xuống 8
        self.conv = ConvBNReLU(3, 8, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(8, 4, 1, stride=1, padding=0),
            ConvBNReLU(4, 8, 3, stride=2),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(16, 8, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class GELayerS1(nn.Module):
    # Exp ratio mặc định giờ là 2 thay vì 6
    def __init__(self, in_chan, out_chan, exp_ratio=2):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        return self.relu(feat)

class GELayerS2(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=2):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=2, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=2, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        return self.relu(feat)

# =========================================================================
# [CẢI TIẾN 4]: BỎ STAGE 5 Ở SEMANTIC BRANCH
# =========================================================================
class SegmentBranch(nn.Module):
    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock() # Output: 8 channels
        self.S3 = nn.Sequential(GELayerS2(8, 16), GELayerS1(16, 16))
        self.S4 = nn.Sequential(GELayerS2(16, 32), GELayerS1(32, 32))
        # Đã loại bỏ S5_4 và S5_5 (CEBlock) để tránh mất chi tiết ở 1/32

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        return feat2, feat3, feat4 # Trả về feat4 làm Output cuối của nhánh


# =========================================================================
# [CẢI TIẾN 3]: DSC-HÓA TOÀN BỘ BGALayer
# =========================================================================
class BGALayer(nn.Module):
    def __init__(self, in_d=64, in_s=32, out_chan=64):
        super(BGALayer, self).__init__()
        # Lớp gốc vốn đã dùng DSC, ta giữ nguyên và scale theo in_d
        self.left1 = nn.Sequential(
            nn.Conv2d(in_d, in_d, kernel_size=3, stride=1, padding=1, groups=in_d, bias=False),
            nn.BatchNorm2d(in_d),
            nn.Conv2d(in_d, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # Lớp gốc dùng Conv thường -> Chuyển thành DSC
        self.left2 = nn.Sequential(
            nn.Conv2d(in_d, in_d, kernel_size=3, stride=2, padding=1, groups=in_d, bias=False),
            nn.BatchNorm2d(in_d),
            nn.Conv2d(in_d, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        )
        # Lớp gốc dùng Conv thường -> Chuyển thành DSC
        self.right1 = nn.Sequential(
            nn.Conv2d(in_s, in_s, kernel_size=3, stride=1, padding=1, groups=in_s, bias=False),
            nn.BatchNorm2d(in_s),
            nn.Conv2d(in_s, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # Lớp gốc vốn đã dùng DSC, ta giữ nguyên và scale theo in_s
        self.right2 = nn.Sequential(
            nn.Conv2d(in_s, in_s, kernel_size=3, stride=1, padding=1, groups=in_s, bias=False),
            nn.BatchNorm2d(in_s),
            nn.Conv2d(in_s, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
        # [QUAN TRỌNG]: Vì nhánh Semantic giờ dừng ở 1/16, còn Detail ở 1/8
        # Ta chỉ cần upsample scale_factor=2 thay vì 4 như bản gốc
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_d, x_s):
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        
        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        
        out = self.conv(left + right)
        return out


class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor_final = up_factor // 2 if aux else up_factor
        
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
            ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, n_classes, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor_final, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        return self.conv_out(feat)

# =========================================================================
# MAIN MODEL
# =========================================================================
class BiSeNetV2_Slim(nn.Module):
    def __init__(self, n_classes, aux_mode='train'):
        super(BiSeNetV2_Slim, self).__init__()
        self.aux_mode = aux_mode
        
        self.detail = DetailBranch() # Output 64 channels
        self.segment = SegmentBranch() # Output 32 channels
        
        self.bga = BGALayer(in_d=64, in_s=32, out_chan=64)
        
        # Hạ mid_channel của Head từ 1024 xuống 256 để cân bằng với 64 channel đầu vào
        self.head = SegmentHead(64, 256, n_classes, up_factor=8, aux=False)
        
        if self.aux_mode == 'train':
            self.aux2 = SegmentHead(8, 64, n_classes, up_factor=4)
            self.aux3 = SegmentHead(16, 64, n_classes, up_factor=8)
            self.aux4 = SegmentHead(32, 64, n_classes, up_factor=16)

        self.init_weights()

    def forward(self, x):
        feat_d = self.detail(x)
        feat2, feat3, feat4 = self.segment(x)
        
        feat_head = self.bga(feat_d, feat4)
        logits = self.head(feat_head)

        if self.aux_mode == 'train':
            return logits
        return logits

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # NOTE: Bỏ qua load_pretrain vì ta đã ép cân triệt để, kiến trúc không còn
        # khớp với backbone ImageNet gốc của tác giả nữa. Model sẽ train từ đầu (From Scratch).
        print("✓ BiSeNetV2_Slim initialized with random weights (from scratch).")


def build_model(num_classes=1):
    return BiSeNetV2_Slim(n_classes=num_classes, aux_mode='eval')