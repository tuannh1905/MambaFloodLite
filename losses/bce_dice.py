import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=1):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        # BCEWithLogitsLoss nhận đầu vào chưa qua sigmoid
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        # 1. Tính BCE Loss
        loss_bce = self.bce(inputs, targets)
        
        # 2. Tính Dice Loss (Cần tự thêm sigmoid)
        inputs_sig = torch.sigmoid(inputs)
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        loss_dice = 1 - dice
        
        # 3. Kết hợp 0.5 BCE + 0.5 DICE
        return 0.5 * loss_bce + 0.5 * loss_dice

def build_loss(num_classes=1):
    return BCEDiceLoss()