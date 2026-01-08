import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, num_classes=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        if self.num_classes == 1:
            # Binary
            inputs = torch.sigmoid(inputs)
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            
            TP = (inputs * targets).sum()
            FP = ((1 - targets) * inputs).sum()
            FN = (targets * (1 - inputs)).sum()
            
            tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            return 1 - tversky
        else:
            # Multi-class
            inputs = F.softmax(inputs, dim=1)
            targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
            
            tversky_scores = []
            for cls in range(self.num_classes):
                input_cls = inputs[:, cls].reshape(-1)
                target_cls = targets_one_hot[:, cls].reshape(-1)
                
                TP = (input_cls * target_cls).sum()
                FP = ((1 - target_cls) * input_cls).sum()
                FN = (target_cls * (1 - input_cls)).sum()
                
                tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
                tversky_scores.append(tversky)
            
            return 1 - torch.mean(torch.stack(tversky_scores))

def build_loss(num_classes=1):
    return TverskyLoss(num_classes=num_classes)