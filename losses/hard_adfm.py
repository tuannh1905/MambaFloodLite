import torch
import torch.nn as nn


def hard_sigmoid(x, low=-3.0, high=3.0):
    return torch.clamp((x - low) / (high - low), min=0.0, max=1.0)


class HardAFMLoss(nn.Module):
    def __init__(self, smooth=1.0, hs_low=-3.0, hs_high=3.0, num_classes=1):
        super(HardAFMLoss, self).__init__()
        self.smooth = smooth
        self.hs_low = hs_low
        self.hs_high = hs_high

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()

        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        recall    = (tp + self.smooth) / (tp + fn + self.smooth)
        sfmi      = precision * recall

        hs        = hard_sigmoid(sfmi, low=self.hs_low, high=self.hs_high)
        loss      = (1 - sfmi) * (1 - hs)
        return loss


def build_loss(num_classes=1):
    return HardAFMLoss(num_classes=num_classes)