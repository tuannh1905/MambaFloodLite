import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    """
    Tversky Loss (Salehi et al., 2017)
    - alpha: hệ số phạt False Positives
    - beta:  hệ số phạt False Negatives
    alpha + beta = 1
    alpha=0.3, beta=0.7 → ưu tiên recall (phạt FN nhiều hơn), phù hợp segmentation y tế
    alpha=beta=0.5       → tương đương Dice Loss
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5, num_classes=1):
        super().__init__()
        self.alpha = alpha   # FP penalty
        self.beta  = beta    # FN penalty
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs  = torch.sigmoid(inputs)
        inputs  = inputs.view(-1)
        targets = targets.view(-1)

        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


def build_loss(num_classes=1):
    return TverskyLoss(num_classes=num_classes)
