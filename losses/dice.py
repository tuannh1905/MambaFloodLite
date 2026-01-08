import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        if self.num_classes == 1:
            # Binary
            inputs = torch.sigmoid(inputs)
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            
            intersection = (inputs * targets).sum()
            dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
            return 1 - dice
        else:
            # Multi-class
            inputs = F.softmax(inputs, dim=1)  # (B, C, H, W)
            targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)
            
            dice_scores = []
            for cls in range(self.num_classes):
                input_cls = inputs[:, cls].reshape(-1)
                target_cls = targets_one_hot[:, cls].reshape(-1)
                
                intersection = (input_cls * target_cls).sum()
                dice = (2. * intersection + self.smooth) / (input_cls.sum() + target_cls.sum() + self.smooth)
                dice_scores.append(dice)
            
            return 1 - torch.mean(torch.stack(dice_scores))

def build_loss(num_classes=1):
    return DiceLoss(num_classes=num_classes)