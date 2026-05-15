import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. BINARY FOCAL LOSS (Trị bóng mây, mái nhà - Hard Negatives)
# ==============================================================================
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha 
        self.gamma = gamma 
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, 1, H, W) - Chưa qua sigmoid
        # targets: (B, 1, H, W) - Nhãn gốc (0 hoặc 1)
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        
        # Công thức Focal Loss: -alpha * (1 - pt)^gamma * log(pt)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==============================================================================
# 2. LOVÁSZ-HINGE LOSS (Trị phân mảnh cấu trúc, tối ưu mIOU trực tiếp)
# ==============================================================================
def lovasz_grad(gt_sorted):
    """Tính gradient của Lovasz extension dựa trên các lỗi đã được sắp xếp."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels):
    """Hàm lõi tính Lovasz Hinge cho vector 1D."""
    if len(labels) == 0:
        return logits.sum() * 0.
        
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs) # Hinge error
    
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        # Lovász yêu cầu dữ liệu phải được ép phẳng (flatten) thành 1D
        logits = logits.view(-1)
        targets = targets.view(-1)
        return lovasz_hinge_flat(logits, targets)

# ==============================================================================
# 3. HÀM MAIN LOSS TỔNG HỢP 
# ==============================================================================
class MainLovaszFocalLoss(nn.Module):
    def __init__(self, num_classes=1, focal_weight=0.5, lovasz_weight=0.5):
        """
        num_classes: Truyền vào để đồng bộ format, thiết kế hiện tại dùng cho Binary (num_classes=1)
        focal_weight: Trọng số cho Focal Loss (trị nhiễu màu sắc)
        lovasz_weight: Trọng số cho Lovász Loss (nối liền các vũng nước)
        """
        super().__init__()
        self.num_classes = num_classes
        self.focal = BinaryFocalLoss(alpha=0.5, gamma=2.0)
        self.lovasz = LovaszHingeLoss()
        self.fw = focal_weight
        self.lw = lovasz_weight

    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_lovasz = self.lovasz(logits, targets)
        return self.fw * loss_focal + self.lw * loss_lovasz

# ==============================================================================
# 4. HÀM BUILD LOSS THEO FORMAT
# ==============================================================================
def build_loss(num_classes=1):
    # Khởi tạo loss với trọng số mặc định là 50/50
    return MainLovaszFocalLoss(num_classes=num_classes, focal_weight=0.5, lovasz_weight=0.5)