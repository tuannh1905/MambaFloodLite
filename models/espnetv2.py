import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

# =================================================================================
# PHẦN 1: CÁC LỚP BỔ TRỢ (Được gom từ thư mục nn_layers của repo gốc)
# =================================================================================
class EfficientPWConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels)
        )
    def forward(self, x): return self.conv(x)

class EfficientPyrPool(nn.Module):
    def __init__(self, in_planes, proj_planes, out_planes, last_layer_br=True):
        super().__init__()
        self.proj_1x1 = nn.Conv2d(in_planes, proj_planes, 1, bias=False)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.pool2 = nn.AvgPool2d(3, stride=2, padding=1)
        self.pool3 = nn.AvgPool2d(5, stride=2, padding=2)
        self.merge = nn.Sequential(
            nn.Conv2d(proj_planes * 4, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.PReLU(out_planes) if last_layer_br else nn.Identity()
        )

    def forward(self, x):
        w, h = x.size(2), x.size(3)
        p0 = self.proj_1x1(x)
        p1 = F.interpolate(self.pool1(p0), size=(w, h), mode='bilinear', align_corners=True)
        p2 = F.interpolate(self.pool2(p0), size=(w, h), mode='bilinear', align_corners=True)
        p3 = F.interpolate(self.pool3(p0), size=(w, h), mode='bilinear', align_corners=True)
        return self.merge(torch.cat([p0, p1, p2, p3], 1))

class EESP(nn.Module):
    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7):
        super(EESP, self).__init__()
        self.stride = stride
        n = nOut // k
        n1 = nOut - (k - 1) * n
        self.proj_1x1 = nn.Sequential(nn.Conv2d(nIn, n, 1, bias=False), nn.BatchNorm2d(n), nn.PReLU(n))
        
        self.s_pyramid = nn.ModuleList()
        for i in range(k):
            dilation = min(2 ** i, r_lim)
            self.s_pyramid.append(nn.Conv2d(n, n1 if i == 0 else n, 3, stride, dilation, dilation, groups=n1 if i == 0 else n, bias=False))
        
        self.merge = nn.Sequential(nn.BatchNorm2d(nOut), nn.PReLU(nOut))
        self.pw_1x1 = nn.Sequential(nn.Conv2d(nOut, nOut, 1, groups=k, bias=False), nn.BatchNorm2d(nOut))
        self.act = nn.PReLU(nOut)

    def forward(self, x):
        out_1x1 = self.proj_1x1(x)
        output = [self.s_pyramid[0](out_1x1)]
        for i in range(1, len(self.s_pyramid)):
            output.append(self.s_pyramid[i](out_1x1) + output[i-1])
        output = self.pw_1x1(self.merge(torch.cat(output, 1)))
        if self.stride == 1 and x.shape == output.shape: output = output + x
        return self.act(output)

class DownSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.eesp = EESP(in_channels + 3, out_channels, stride=2)
    def forward(self, x, x_img):
        # Input reinforcement: Ghép ảnh gốc đã thu nhỏ vào feature map
        if x_img.size(2) != x.size(2):
            x_img = F.interpolate(x_img, size=x.size()[2:], mode='bilinear', align_corners=True)
        return self.eesp(torch.cat([x, x_img], 1))


# =================================================================================
# PHẦN 2: BASE NETWORK (ImageNet Model)
# =================================================================================
class EESPNet(nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.input_reinforcement = True
        config = [32, 64, 128, 256, 512]
        self.config = [int(c * scale) for c in config]
        self.config[0] = 32 # Tầng đầu không scale theo thiết kế gốc
        
        self.level1 = nn.Sequential(nn.Conv2d(3, self.config[0], 3, 2, 1, bias=False), nn.BatchNorm2d(self.config[0]), nn.PReLU(self.config[0]))
        self.level2_0 = DownSampler(self.config[0], self.config[1])
        self.level3_0 = DownSampler(self.config[1], self.config[2])
        self.level3 = nn.ModuleList([EESP(self.config[2], self.config[2]) for _ in range(3)])
        self.level4_0 = DownSampler(self.config[2], self.config[3])
        self.level4 = nn.ModuleList([EESP(self.config[3], self.config[3]) for _ in range(4)])
        
        # Để tránh lỗi 'del' trong hàm init của Segmentation Network
        self.classifier = nn.Identity()
        self.level5 = nn.Identity()
        self.level5_0 = nn.Identity()


# =================================================================================
# PHẦN 3: ESPNetv2 SEGMENTATION NETWORK
# =================================================================================
class ESPNetv2Segmentation(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for Semantic Segmentation.
    '''
    # ✓ SỬA TẠI ĐÂY: Thêm input_size để tương thích với build_model framework
    def __init__(self, classes=1, dataset='pascal', scale=2.0, input_size=256):
        super().__init__()
        
        # ✓ KIỂM TRA TOÁN HỌC: Mạng downsample 4 lần (l1, l2_0, l3_0, l4_0) -> 2^4 = 16
        if input_size % 16 != 0:
            raise ValueError(f"ESPNetv2 yêu cầu input_size chia hết cho 16. Kích thước {input_size} không hợp lệ.")

        # =============================================================
        #                        BASE NETWORK
        # =============================================================
        self.base_net = EESPNet(scale=scale) # ImageNet model
        del self.base_net.classifier
        del self.base_net.level5
        del self.base_net.level5_0
        config = self.base_net.config

        #=============================================================
        #                    SEGMENTATION NETWORK
        #=============================================================
        dec_feat_dict={
            'pascal': 16,
            'city': 16,
            'coco': 32,
            'floodvn': 16 # Thêm bộ dữ liệu của chúng ta vào đây
        }
        
        # Xử lý an toàn nếu truyền tên dataset lạ
        base_dec_planes = dec_feat_dict.get(dataset, 16) 
        
        dec_planes = [4*base_dec_planes, 3*base_dec_planes, 2*base_dec_planes, classes]
        pyr_plane_proj = min(classes // 2 + 1, base_dec_planes) # +1 để tránh loi proj_planes=0 khi classes=1

        self.bu_dec_l1 = EfficientPyrPool(in_planes=config[3], proj_planes=pyr_plane_proj, out_planes=dec_planes[0])
        self.bu_dec_l2 = EfficientPyrPool(in_planes=dec_planes[0], proj_planes=pyr_plane_proj, out_planes=dec_planes[1])
        self.bu_dec_l3 = EfficientPyrPool(in_planes=dec_planes[1], proj_planes=pyr_plane_proj, out_planes=dec_planes[2])
        self.bu_dec_l4 = EfficientPyrPool(in_planes=dec_planes[2], proj_planes=pyr_plane_proj, out_planes=dec_planes[3], last_layer_br=False)

        self.merge_enc_dec_l2 = EfficientPWConv(config[2], dec_planes[0])
        self.merge_enc_dec_l3 = EfficientPWConv(config[1], dec_planes[1])
        self.merge_enc_dec_l4 = EfficientPWConv(config[0], dec_planes[2])

        self.bu_br_l2 = nn.Sequential(nn.BatchNorm2d(dec_planes[0]), nn.PReLU(dec_planes[0]))
        self.bu_br_l3 = nn.Sequential(nn.BatchNorm2d(dec_planes[1]), nn.PReLU(dec_planes[1]))
        self.bu_br_l4 = nn.Sequential(nn.BatchNorm2d(dec_planes[2]), nn.PReLU(dec_planes[2]))

        self.init_params()

    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x_size = (x.size(2), x.size(3))
        enc_out_l1 = self.base_net.level1(x)  # 112
        if not self.base_net.input_reinforcement:
            del x
            x = None

        enc_out_l2 = self.base_net.level2_0(enc_out_l1, x)  # 56

        enc_out_l3_0 = self.base_net.level3_0(enc_out_l2, x)  # down-sample
        for i, layer in enumerate(self.base_net.level3):
            if i == 0:
                enc_out_l3 = layer(enc_out_l3_0)
            else:
                enc_out_l3 = layer(enc_out_l3)

        enc_out_l4_0 = self.base_net.level4_0(enc_out_l3, x)  # down-sample
        for i, layer in enumerate(self.base_net.level4):
            if i == 0:
                enc_out_l4 = layer(enc_out_l4_0)
            else:
                enc_out_l4 = layer(enc_out_l4)

        # bottom-up decoding
        bu_out = self.bu_dec_l1(enc_out_l4)

        # Decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l3_proj = self.merge_enc_dec_l2(enc_out_l3)
        bu_out = enc_out_l3_proj + bu_out
        bu_out = self.bu_br_l2(bu_out)
        bu_out = self.bu_dec_l2(bu_out)

        # decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l2_proj = self.merge_enc_dec_l3(enc_out_l2)
        bu_out = enc_out_l2_proj + bu_out
        bu_out = self.bu_br_l3(bu_out)
        bu_out = self.bu_dec_l3(bu_out)

        # decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l1_proj = self.merge_enc_dec_l4(enc_out_l1)
        bu_out = enc_out_l1_proj + bu_out
        bu_out = self.bu_br_l4(bu_out)
        bu_out  = self.bu_dec_l4(bu_out)

        return F.interpolate(bu_out, size=x_size, mode='bilinear', align_corners=True)


# =================================================================================
# PHẦN 4: TEMPLATE BUILD MODEL
# =================================================================================
# ✓ SỬA TẠI ĐÂY: Hàm build_model giờ nhận tham số input_size
def build_model(num_classes=1, input_size=256):
    '''
    Khởi tạo mô hình ESPNetv2 với cấu hình scale s=2.0 
    để đảm bảo số lượng tham số đạt mức chuẩn ~2.1M như thiết kế gốc.
    '''
    return ESPNetv2Segmentation(classes=num_classes, dataset='floodvn', scale=2.0, input_size=input_size)