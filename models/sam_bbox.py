import torch
import torch.nn as nn
from transformers import SamModel, SamProcessor
import torch.nn.functional as F


class SAMBbox(nn.Module):
    def __init__(self, pretrained="facebook/sam-vit-base"):
        super(SAMBbox, self).__init__()
        self.sam = SamModel.from_pretrained(pretrained)
        self.processor = SamProcessor.from_pretrained(pretrained)

        for name, param in self.sam.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

    def forward(self, x):
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        B, C, H, W = x.shape
        device = x.device

        boxes = torch.tensor([[0, 0, W, H]], dtype=torch.float32)
        boxes = boxes.unsqueeze(1).expand(B, 1, -1).to(device)

        outputs = self.sam(
            pixel_values=x,
            input_boxes=boxes,
            multimask_output=False
        )

        return outputs.pred_masks.squeeze(1)


def build_model(num_classes=1, seed=42):
    return SAMBbox()