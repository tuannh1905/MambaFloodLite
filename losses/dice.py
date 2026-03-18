import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, bce_weight=0.2, dice_weight=0.8, num_classes=1):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        # Có thể thêm pos_weight vào BCE nếu vùng lũ cực kỳ nhỏ
        # ví dụ: pos_weight = torch.tensor([5.0])
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        # 1. Tính BCE Loss (giữ nguyên)
        loss_bce = self.bce(inputs, targets)
        
        # 2. Tính Dice Loss theo TỪNG ẢNH (Image-level Dice)
        inputs_sig = torch.sigmoid(inputs)
        
        # Thay vì .view(-1), ta giữ lại chiều batch_size: (Batch, H*W)
        batch_size = inputs_sig.size(0)
        inputs_flat = inputs_sig.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # Tính intersection và union theo từng dòng (từng ảnh trong batch)
        intersection = (inputs_flat * targets_flat).sum(dim=1)
        union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        # dice_score là một vector có độ dài bằng batch_size
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Lấy 1 trừ đi trung bình Dice của cả batch
        loss_dice = 1.0 - dice_score.mean()
        
        # 3. Kết hợp với tỷ lệ mới ưu tiên Dice hơn
        return (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)

def build_loss(num_classes=1):
    # Bạn có thể thử nghiệm bce_weight=0.2, dice_weight=0.8
    # Hoặc nếu Kaggle quá khó, đẩy lên bce_weight=0.1, dice_weight=0.9
    return BCEDiceLoss(smooth=1.0, bce_weight=0.2, dice_weight=0.8, num_classes=num_classes)