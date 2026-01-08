# bce.py
import torch.nn as nn

def build_loss(num_classes=1):
    return nn.BCEWithLogitsLoss()