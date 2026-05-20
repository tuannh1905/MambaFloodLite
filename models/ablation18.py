import torch
import torch.nn as nn
import torch.nn.functional as F

# (Giữ nguyên các khối get_activation, SquareDW, NearestUpsample, Straight3x3Block, EncoderBlock, ConcatDecoderBlock, SerialBottleneck như bản Base)

# ... [BẠN DÁN CÁC KHỐI CƠ BẢN VÀO ĐÂY] ...

# ==============================================================================
# 5. ABLATION 18: SHALLOW BOUNDARY-ONLY SUPERVISION (CHỈ DÙNG AUX Ở E2)
# ==============================================================================
class Ablation18_ShallowAuxOnly(nn.Module):
    def __init__(self, num_classes=1, input_size=128):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        self.e1 = EncoderBlock(16, 32)   # out: 32 channels
        self.e2 = EncoderBlock(32, 64)   # out: 64 channels
        self.e3 = EncoderBlock(64, 128)  # out: 128 channels
        self.e4 = EncoderBlock(128, 128) # out: 128 channels
        
        # Aux 1: Đặt ở E2 (Tầng nông, độ phân giải cao). 
        # Cắt bỏ hoàn toàn Aux ở E4, dồn toàn lực bắt mạng học ranh giới ở E2.
        self.aux_head_detail = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        self.bottleneck = SerialBottleneck_NoAttn(128)
        
        self.d4 = ConcatDecoderBlock_NoAttn(128, 128, 128) 
        self.d3 = ConcatDecoderBlock_NoAttn(128, 128, 64)  
        self.d2 = ConcatDecoderBlock_NoAttn(64, 64, 32)   
        self.d1 = ConcatDecoderBlock_NoAttn(32, 32, 16)   
        
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[2:] 

        x = self.conv_in(x)
        x, s1 = self.e1(x)
        x, s2 = self.e2(x) # Lấy s2 cho Detail Aux
        x, s3 = self.e3(x)
        x, s4 = self.e4(x) 
        
        out_detail = None
        
        if self.training:
            # Chỉ tính toán 1 nhánh Aux duy nhất tại E2
            out_detail = self.aux_head_detail(s2)
            # Phóng to về kích thước gốc
            out_detail = F.interpolate(out_detail, size=input_shape, mode='bilinear', align_corners=False)
        
        x = self.bottleneck(x)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        
        main_out = self.conv_out(x)
        
        if self.training:
            # Tuple trả về: (Nhánh chính, Nhánh chi tiết viền E2)
            return main_out, out_detail
        return main_out

def build_model(num_classes=1, input_size=128):
    return Ablation18_ShallowAuxOnly(num_classes=num_classes, input_size=input_size)