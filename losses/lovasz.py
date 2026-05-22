# lovasz.py  ─  Standalone Lovász-Hinge Loss (Binary Segmentation)
#
# Paper: "The Lovász-Softmax Loss: A Tractable Surrogate for the
#         Optimization of the Intersection-Over-Union Measure in
#         Neural Networks" (Berman et al., CVPR 2018)
#
# Ý tưởng: thay vì tối ưu surrogate loss gián tiếp, Lovász-Hinge tối ưu
# trực tiếp Jaccard/IoU thông qua Lovász extension của hàm tập hợp.
# Đặc biệt hiệu quả với các vùng nhỏ, phân mảnh, và khi mIoU là metric chính.
#
# So sánh nhanh với Focal/Dice:
#   Focal  → trị hard-negatives, imbalance màu sắc / texture
#   Dice   → F1 proxy, tốt cho recall/precision cân bằng
#   Lovász → IoU proxy trực tiếp, tốt cho structural continuity, viền mảnh

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Core: Lovász gradient (Berman et al. 2018, Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Tính Lovász gradient cho ground-truth đã được sắp xếp theo thứ tự giảm dần
    của hinge errors.

    Args:
        gt_sorted: (N,) – nhãn binary {0,1} đã reorder theo perm của hinge error

    Returns:
        jaccard: (N,) – gradient của Lovász extension
    """
    p   = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union        = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard      = 1.0 - intersection / union          # (N,)
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]       # discrete derivative
    return jaccard


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Lovász-Hinge trên vector 1-D (sau khi flatten).

    Args:
        logits: (N,) – raw logit, CHƯA qua sigmoid
        labels: (N,) – nhãn binary {0, 1}
    """
    if len(labels) == 0:
        return logits.sum() * 0.0

    signs  = 2.0 * labels.float() - 1.0          # {-1, +1}
    errors = 1.0 - logits * signs                 # hinge error (N,)

    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm.data]

    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────────────────
class LovaszHingeLoss(nn.Module):
    """
    Lovász-Hinge Loss cho binary segmentation.

    Supports two per-image reduction strategies:
        per_image=False  (default): flatten toàn bộ batch → 1 loss value
                                    ổn định hơn khi batch nhỏ
        per_image=True           : tính loss từng ảnh rồi average
                                    tốt hơn khi class ratio khác nhau giữa ảnh
    """
    def __init__(self, per_image: bool = False, num_classes: int = 1):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) hoặc (B, H, W) – raw logit
            targets : (B, 1, H, W) hoặc (B, H, W) – nhãn binary {0, 1}
        Returns:
            scalar loss
        """
        if self.per_image:
            # Tính riêng từng ảnh trong batch rồi lấy mean
            B = logits.shape[0]
            losses = [
                _lovasz_hinge_flat(logits[i].view(-1), targets[i].view(-1))
                for i in range(B)
            ]
            return torch.stack(losses).mean()
        else:
            return _lovasz_hinge_flat(logits.view(-1), targets.view(-1))


# ─────────────────────────────────────────────────────────────────────────────
# Build helper (đồng bộ format với các loss khác trong repo)
# ─────────────────────────────────────────────────────────────────────────────
def build_loss(num_classes: int = 1):
    """
    per_image=False  → phù hợp batch nhỏ (batch_size <= 4)
    per_image=True   → phù hợp khi mỗi ảnh có class distribution khác nhau
    """
    return LovaszHingeLoss(per_image=False, num_classes=num_classes)
