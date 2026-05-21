import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# [GIỮ NGUYÊN TOÀN BỘ CÁC KHỐI CƠ BẢN CỦA BẠN TẠI ĐÂY]
# get_activation, SquareDW, NearestUpsample, Straight3x3Block, 
# EncoderBlock, ConcatDecoderBlock_NoAttn, SerialBottleneck_NoAttn
# ==============================================================================

# ==============================================================================
# ABLATION: ĐƯA AUX HEAD LÊN TẦNG NÔNG (E2) ĐỂ HỌC VIỀN
# ==============================================================================
class Ablation_ShallowAux(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        
        self.conv_in = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # Encoder
        self.e1 = EncoderBlock(32, 64,   act_type='relu6')   # Out: s1 (64 channels)
        self.e2 = EncoderBlock(64, 128,  act_type='relu6')   # Out: s2 (128 channels)
        self.e3 = EncoderBlock(128, 128, act_type='relu6')   # Out: s3 (128 channels)
        self.e4 = EncoderBlock(128, 128, act_type='relu6')   # Out: s4 (128 channels)
        
        # Nhánh phụ (Auxiliary Head) chĩa ra từ E2 (Tầng nông)
        # s2 có 128 channels, ta thiết kế Aux Head nhận 128 channels
        aux_dim = 64
        self.aux_head_shallow = nn.Sequential(
            nn.Conv2d(128, aux_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aux_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(aux_dim, num_classes, kernel_size=1)
        )
        
        # Bottleneck & Decoder 
        self.bottleneck = SerialBottleneck_NoAttn(128, act_type='relu6')
        
        self.d4 = ConcatDecoderBlock_NoAttn(in_c=128, skip_c=128, out_c=128, act_type='relu6') 
        self.d3 = ConcatDecoderBlock_NoAttn(in_c=128, skip_c=128, out_c=64,  act_type='relu6')  
        self.d2 = ConcatDecoderBlock_NoAttn(in_c=64,  skip_c=128, out_c=32,  act_type='relu6')   
        self.d1 = ConcatDecoderBlock_NoAttn(in_c=32,  skip_c=64,  out_c=16,  act_type='relu6')   
        
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[2:] 

        x = self.conv_in(x)
        
        x, s1 = self.e1(x)
        x, s2 = self.e2(x) # Lấy đặc trưng tầng nông s2
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        
        # Xử lý Aux Head (Chỉ kích hoạt lúc Train)
        aux_out = None
        if self.training:
            # Truyền s2 vào nhánh Aux thay vì s4
            aux_out = self.aux_head_shallow(s2)
            aux_out = F.interpolate(aux_out, size=input_shape, mode='bilinear', align_corners=False)
        
        # Luồng chính tiếp tục
        x = self.bottleneck(x)
        
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        
        main_out = self.conv_out(x)
        
        if self.training:
            return main_out, aux_out
        return main_out

def build_model(num_classes=1, input_size=128):
    return Ablation_ShallowAux(num_classes=num_classes, input_size=input_size)