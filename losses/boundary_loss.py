import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size=5):
        """
        Hàm loss chuyên trị đường viền.
        kernel_size: Độ dày của đường viền muốn trích xuất (Mặc định: 5 pixel).
                     Ảnh Kaggle vệ tinh thường mờ, nên để viền hơi dày một chút để dễ học.
        """
        super(BoundaryLoss, self).__init__()
        self.kernel_size = kernel_size

    def get_boundary(self, mask):
        """
        Tự động trích xuất viền từ Mask Segmentation (0 và 1)
        Bằng công thức: Boundary = Dilation(Mask) - Erosion(Mask)
        """
        # Đảm bảo mask có dạng 4D: (Batch, Channel, Height, Width)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
            
        mask = mask.float()

        # Dilation (Phình to mask)
        dilation = F.max_pool2d(mask, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        
        # Erosion (Co rút mask) -> Bằng cách đảo dấu và dùng max_pool
        erosion = -F.max_pool2d(-mask, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        
        # Lấy phần rìa
        boundary = dilation - erosion
        return boundary

    def forward(self, pred_boundary, target_mask):
        """
        pred_boundary: Logits dự đoán từ nhánh Aux (B, 1, H, W)
        target_mask: Ground Truth Mask của vùng ngập (B, 1, H, W)
        """
        # 1. Trích xuất viền thực tế từ Mask
        gt_boundary = self.get_boundary(target_mask)

        # 2. Tính toán trọng số để cân bằng (Viền vs Nền)
        # Vì viền rất mỏng, ta phải phạt thật nặng nếu model đoán sai viền
        pos_pixels = gt_boundary.sum()
        neg_pixels = (1 - gt_boundary).sum()
        
        if pos_pixels > 0:
            pos_weight = neg_pixels / pos_pixels
            # Clamp (khóa) trọng số để tránh nổ Gradient nếu viền quá bé
            pos_weight = torch.clamp(pos_weight, min=1.0, max=50.0) 
        else:
            pos_weight = torch.tensor(1.0).to(pred_boundary.device)

        # 3. Tính BCE Loss với trọng số
        bce_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = bce_weighted(pred_boundary, gt_boundary)
        
        return loss


# =========================================================================
# HÀM KẾT HỢP: JOINT LOSS (DÙNG CHO FILE TRAINER.PY)
# =========================================================================
class JointEdgeSegLoss(nn.Module):
    """
    Hàm Loss tổng hợp: Kết hợp giữa Loss phân vùng chính và Loss viền phụ.
    """
    def __init__(self, main_loss_fn, edge_weight=0.4, edge_kernel=5):
        super(JointEdgeSegLoss, self).__init__()
        self.main_loss = main_loss_fn          # Hàm loss chính của bạn (VD: BCEDiceLoss)
        self.edge_loss = BoundaryLoss(kernel_size=edge_kernel)
        self.edge_weight = edge_weight         # Trọng số nhánh phụ (Thường để 0.4 -> 0.8)

    def forward(self, preds, target):
        # Nếu model trả về 1 Tuple (Lúc Train có Aux Branch)
        if isinstance(preds, (list, tuple)):
            main_pred = preds[0]
            edge_pred = preds[1]
            
            loss_seg = self.main_loss(main_pred, target)
            loss_edge = self.edge_loss(edge_pred, target)
            
            # Trả về tổng Loss
            return loss_seg + self.edge_weight * loss_edge
            
        # Nếu model chỉ trả về 1 Tensor (Lúc Inference hoặc model không có Aux)
        else:
            return self.main_loss(preds, target)