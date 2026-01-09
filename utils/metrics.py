import torch
import numpy as np

def calculate_miou(all_preds, all_labels, num_classes, threshold=0.5):
    """
    Calculate mean Intersection over Union (mIOU) with proper batch handling
    
    Args:
        all_preds: List of prediction arrays from batches
        all_labels: List of label arrays from batches
        num_classes: Number of classes (1 for binary, >1 for multi-class)
        threshold: Threshold for binary segmentation (default: 0.5)
    
    Returns:
        float: mIOU score
    """
    # Convert list of batches to single array
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if num_classes == 1:
        # Binary segmentation: shape (N, 1, H, W) or (N, H, W)
        # Ensure shape is (N, H, W) by squeezing channel dimension if present
        if all_preds.ndim == 4:  # (N, 1, H, W)
            all_preds = all_preds.squeeze(1)
        if all_labels.ndim == 4:  # (N, 1, H, W)
            all_labels = all_labels.squeeze(1)
        
        # Apply threshold
        all_preds = (all_preds > threshold).astype(np.uint8)
        all_labels = all_labels.astype(np.uint8)
        
        # Flatten all spatial dimensions: (N, H, W) -> (N*H*W,)
        all_preds_flat = all_preds.reshape(-1)
        all_labels_flat = all_labels.reshape(-1)
        
        # Calculate IoU
        intersection = np.logical_and(all_preds_flat, all_labels_flat).sum()
        union = np.logical_or(all_preds_flat, all_labels_flat).sum()
        
        if union == 0:
            return 0.0
        return float(intersection / union)
    
    else:
        # Multi-class segmentation: shape (N, C, H, W)
        # Get predicted class for each pixel: (N, C, H, W) -> (N, H, W)
        preds_class = np.argmax(all_preds, axis=1)
        
        # Labels shape should be (N, H, W)
        if all_labels.ndim == 4:  # If (N, 1, H, W), squeeze
            all_labels = all_labels.squeeze(1)
        
        # Flatten spatial dimensions for easier computation
        preds_class_flat = preds_class.reshape(-1)
        labels_flat = all_labels.reshape(-1)
        
        # Calculate IoU for each class
        ious = []
        for cls in range(num_classes):
            pred_mask = (preds_class_flat == cls)
            label_mask = (labels_flat == cls)
            
            intersection = np.logical_and(pred_mask, label_mask).sum()
            union = np.logical_or(pred_mask, label_mask).sum()
            
            if union > 0:
                ious.append(float(intersection / union))
        
        # Return mean IoU across classes that exist in the dataset
        return float(np.mean(ious)) if ious else 0.0


def calculate_dice_score(all_preds, all_labels, num_classes, threshold=0.5):
    """
    Calculate Dice Score (F1 Score) for segmentation
    
    Args:
        all_preds: List of prediction arrays from batches
        all_labels: List of label arrays from batches
        num_classes: Number of classes
        threshold: Threshold for binary segmentation
    
    Returns:
        float: Dice score
    """
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if num_classes == 1:
        if all_preds.ndim == 4:
            all_preds = all_preds.squeeze(1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)
        
        all_preds = (all_preds > threshold).astype(np.uint8)
        all_labels = all_labels.astype(np.uint8)
        
        all_preds_flat = all_preds.reshape(-1)
        all_labels_flat = all_labels.reshape(-1)
        
        intersection = np.logical_and(all_preds_flat, all_labels_flat).sum()
        dice = (2.0 * intersection) / (all_preds_flat.sum() + all_labels_flat.sum() + 1e-8)
        
        return float(dice)
    
    else:
        preds_class = np.argmax(all_preds, axis=1)
        
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)
        
        preds_class_flat = preds_class.reshape(-1)
        labels_flat = all_labels.reshape(-1)
        
        dice_scores = []
        for cls in range(num_classes):
            pred_mask = (preds_class_flat == cls)
            label_mask = (labels_flat == cls)
            
            intersection = np.logical_and(pred_mask, label_mask).sum()
            dice = (2.0 * intersection) / (pred_mask.sum() + label_mask.sum() + 1e-8)
            
            if pred_mask.sum() > 0 or label_mask.sum() > 0:
                dice_scores.append(float(dice))
        
        return float(np.mean(dice_scores)) if dice_scores else 0.0


def calculate_pixel_accuracy(all_preds, all_labels, num_classes, threshold=0.5):
    """
    Calculate pixel-wise accuracy
    
    Args:
        all_preds: List of prediction arrays from batches
        all_labels: List of label arrays from batches
        num_classes: Number of classes
        threshold: Threshold for binary segmentation
    
    Returns:
        float: Pixel accuracy
    """
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if num_classes == 1:
        if all_preds.ndim == 4:
            all_preds = all_preds.squeeze(1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)
        
        all_preds = (all_preds > threshold).astype(np.uint8)
        all_labels = all_labels.astype(np.uint8)
    else:
        all_preds = np.argmax(all_preds, axis=1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)
    
    all_preds_flat = all_preds.reshape(-1)
    all_labels_flat = all_labels.reshape(-1)
    
    correct = (all_preds_flat == all_labels_flat).sum()
    total = all_preds_flat.size
    
    return float(correct / total)