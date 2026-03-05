import torch
import torch.nn as nn
import numpy as np
from transformers import SamModel, SamProcessor
import torch.nn.functional as F


class SAMPoint(nn.Module):
    def __init__(self, num_points=30, pretrained="facebook/sam-vit-base"):
        super(SAMPoint, self).__init__()
        self.sam = SamModel.from_pretrained(pretrained)
        self.processor = SamProcessor.from_pretrained(pretrained)
        self.num_points = num_points

        for name, param in self.sam.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

    def _make_grid_points(self, size, device):
        n = int(self.num_points ** 0.5)
        xs = torch.linspace(0, size - 1, n)
        ys = torch.linspace(0, size - 1, n)
        grid = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1).reshape(-1, 2)
        return grid.long().tolist()

    def forward(self, x):
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        B, C, H, W = x.shape
        device = x.device

        points = self._make_grid_points(H, device)
        input_points = torch.tensor(points, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        input_points = input_points.expand(B, 1, -1, -1).to(device)

        outputs = self.sam(
            pixel_values=x,
            input_points=input_points,
            multimask_output=False
        )

        return outputs.pred_masks.squeeze(1)


def build_model(num_classes=1, seed=42):
    return SAMPoint(num_points=30)