# File: losses/boundary_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size=5, dice_weight=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.dice_weight = dice_weight

    def dice_loss(self, pred, target, smooth=1e-5):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, pred_boundary, gt_mask):
        gt_mask = gt_mask.float()
        
        dilated = F.max_pool2d(gt_mask, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        eroded = -F.max_pool2d(-gt_mask, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        gt_boundary = dilated - eroded
        gt_boundary = (gt_boundary > 0.5).float()
        
        pos_pixels = gt_boundary.sum()
        neg_pixels = (1 - gt_boundary).sum()
        
        if pos_pixels > 0:
            pos_weight = neg_pixels / pos_pixels
            pos_weight = torch.clamp(pos_weight, min=1.0, max=50.0) 
        else:
            pos_weight = torch.tensor(1.0, device=pred_boundary.device)
            
        loss_bce = F.binary_cross_entropy_with_logits(pred_boundary, gt_boundary, pos_weight=pos_weight)
        loss_dice = self.dice_loss(pred_boundary, gt_boundary)
        
        return loss_bce + self.dice_weight * loss_dice

# Thêm hàm này để __init__.py có thể gọi được
def build_loss(num_classes=1):
    # Trả về instance của BoundaryLoss với các tham số mặc định tối ưu cho bài toán của bạn
    return BoundaryLoss(kernel_size=5, dice_weight=0.5)