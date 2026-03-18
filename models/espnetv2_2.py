import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

# =================================================================================
# [CẢI TIẾN 3]: ECA MODULE (MẮT THẦN CHỐNG PHẢN CHIẾU)
# =================================================================================
class ECAModule(nn.Module):
    """
    Efficient Channel Attention: Giúp mạng tự động đánh trọng số cao cho các kênh 
    chứa đặc trưng nước đục, lờ đi các kênh chứa bầu trời/bóng râm.
    """
    def __init__(self, k_size=3):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        # Reshape để dùng Conv1D thay vì Fully Connected (Tiết kiệm cực nhiều tham số)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# =================================================================================
# PHẦN 1: CÁC LỚP BỔ TRỢ 
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
    # [CẢI TIẾN 2]: Dilated Pruning - Chỉnh r_lim=8 để chỉ sinh ra các luồng [1, 2, 4, 8]
    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=8):
        super(EESP, self).__init__()
        self.stride = stride
        n = nOut // k
        n1 = nOut - (k - 1) * n
        self.proj_1x1 = nn.Sequential(nn.Conv2d(nIn, n, 1, bias=False), nn.BatchNorm2d(n), nn.PReLU(n))
        
        self.s_pyramid = nn.ModuleList()
        for i in range(k):
            # Các luồng Dilation sẽ là: 1, 2, 4, 8 (Loại bỏ 16, 32 gây răng cưa)
            dilation = min(2 ** i, r_lim)
            self.s_pyramid.append(nn.Conv2d(n, n1 if i == 0 else n, 3, stride, dilation, dilation, groups=n1 if i == 0 else n, bias=False))
        
        self.merge = nn.Sequential(nn.BatchNorm2d(nOut), nn.PReLU(nOut))
        self.pw_1x1 = nn.Sequential(nn.Conv2d(nOut, nOut, 1, groups=k, bias=False), nn.BatchNorm2d(nOut))
        
        # [CẢI TIẾN 3]: Tích hợp ECA Module ngay sau Hierarchical Feature Fusion
        self.eca = ECAModule(k_size=3)
        self.act = nn.PReLU(nOut)

    def forward(self, x):
        out_1x1 = self.proj_1x1(x)
        output = [self.s_pyramid[0](out_1x1)]
        for i in range(1, len(self.s_pyramid)):
            output.append(self.s_pyramid[i](out_1x1) + output[i-1])
        
        output = self.pw_1x1(self.merge(torch.cat(output, 1)))
        
        # [CẢI TIẾN 3]: Chạy qua ECA Module để lọc nhiễu channel
        output = self.eca(output)
        
        if self.stride == 1 and x.shape == output.shape: output = output + x
        return self.act(output)

class DownSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.eesp = EESP(in_channels + 3, out_channels, stride=2)
    def forward(self, x, x_img):
        if x_img.size(2) != x.size(2):
            x_img = F.interpolate(x_img, size=x.size()[2:], mode='bilinear', align_corners=True)
        return self.eesp(torch.cat([x, x_img], 1))


# =================================================================================
# PHẦN 2: BASE NETWORK 
# =================================================================================
class EESPNet(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.input_reinforcement = True
        config = [32, 64, 128, 256, 512]
        self.config = [int(c * scale) for c in config]
        self.config[0] = 32 
        
        self.level1 = nn.Sequential(nn.Conv2d(3, self.config[0], 3, 2, 1, bias=False), nn.BatchNorm2d(self.config[0]), nn.PReLU(self.config[0]))
        self.level2_0 = DownSampler(self.config[0], self.config[1])
        self.level3_0 = DownSampler(self.config[1], self.config[2])
        self.level3 = nn.ModuleList([EESP(self.config[2], self.config[2]) for _ in range(3)])
        self.level4_0 = DownSampler(self.config[2], self.config[3])
        self.level4 = nn.ModuleList([EESP(self.config[3], self.config[3]) for _ in range(4)])
        
        self.classifier = nn.Identity()
        self.level5 = nn.Identity()
        self.level5_0 = nn.Identity()


# =================================================================================
# PHẦN 3: ESPNetv2 SEGMENTATION NETWORK
# =================================================================================
class ESPNetv2Segmentation(nn.Module):
    def __init__(self, classes=1, dataset='pascal', scale=1.0, aux=False):
        super().__init__()
        self.aux = aux

        self.base_net = EESPNet(scale=scale) 
        del self.base_net.classifier
        del self.base_net.level5
        del self.base_net.level5_0
        config = self.base_net.config

        dec_feat_dict={
            'pascal': 16,
            'city': 16,
            'coco': 32,
            'floodvn': 16,
            'floodkaggle': 16 # Thêm Kaggle dataset
        }
        base_dec_planes = dec_feat_dict.get(dataset, 16) 
        
        dec_planes = [4*base_dec_planes, 3*base_dec_planes, 2*base_dec_planes, classes]
        pyr_plane_proj = min(classes // 2 + 1, base_dec_planes)

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

        # =========================================================================
        # [CẢI TIẾN 4]: AUXILIARY BOUNDARY HEAD (Nhánh dự đoán đường biên)
        # =========================================================================
        if self.aux:
            # Lấy đặc trưng từ l2 (độ phân giải 1/4) để vẽ đường biên cho sắc nét
            self.boundary_head = nn.Sequential(
                nn.Conv2d(config[1], 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.PReLU(64),
                nn.Conv2d(64, classes, kernel_size=1)
            )

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
        
        # --- Encoder ---
        enc_out_l1 = self.base_net.level1(x)  # 1/2
        if not self.base_net.input_reinforcement: x = None

        enc_out_l2 = self.base_net.level2_0(enc_out_l1, x)  # 1/4

        enc_out_l3_0 = self.base_net.level3_0(enc_out_l2, x)  # 1/8
        for i, layer in enumerate(self.base_net.level3):
            if i == 0: enc_out_l3 = layer(enc_out_l3_0)
            else: enc_out_l3 = layer(enc_out_l3)

        enc_out_l4_0 = self.base_net.level4_0(enc_out_l3, x)  # 1/16
        for i, layer in enumerate(self.base_net.level4):
            if i == 0: enc_out_l4 = layer(enc_out_l4_0)
            else: enc_out_l4 = layer(enc_out_l4)

        # --- Decoder (Bottom-up) ---
        bu_out = self.bu_dec_l1(enc_out_l4)
        
        bu_out = self.upsample(bu_out)
        enc_out_l3_proj = self.merge_enc_dec_l2(enc_out_l3)
        bu_out = enc_out_l3_proj + bu_out
        bu_out = self.bu_br_l2(bu_out)
        bu_out = self.bu_dec_l2(bu_out)

        bu_out = self.upsample(bu_out)
        enc_out_l2_proj = self.merge_enc_dec_l3(enc_out_l2)
        bu_out = enc_out_l2_proj + bu_out
        bu_out = self.bu_br_l3(bu_out)
        bu_out = self.bu_dec_l3(bu_out)

        bu_out = self.upsample(bu_out)
        enc_out_l1_proj = self.merge_enc_dec_l4(enc_out_l1)
        bu_out = enc_out_l1_proj + bu_out
        bu_out = self.bu_br_l4(bu_out)
        main_out  = self.bu_dec_l4(bu_out)

        main_out = F.interpolate(main_out, size=x_size, mode='bilinear', align_corners=True)

        # [CẢI TIẾN 4]: Trả về thêm Boundary Map khi đang Train
        if self.aux and self.training:
            bound_out = self.boundary_head(enc_out_l2)
            bound_out = F.interpolate(bound_out, size=x_size, mode='bilinear', align_corners=True)
            return main_out, bound_out
            
        return main_out


# =================================================================================
# PHẦN 4: TEMPLATE BUILD MODEL
# =================================================================================
def build_model(num_classes=1):
    '''
    [CẢI TIẾN 1]: Width Scaling - Đặt hệ số scale=0.75 
    Kéo tham số từ 836k xuống mốc lý tưởng ~0.47M.
    Để aux=False cho lúc Test/Inference chạy nhẹ mượt.
    '''
    return ESPNetv2Segmentation(classes=num_classes, dataset='floodkaggle', scale=0.75, aux=False)