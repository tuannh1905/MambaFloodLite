import torch
import numpy as np

def calculate_miou(all_preds, all_labels, num_classes, threshold=0.5):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if num_classes == 1:
        all_preds = (all_preds > threshold).astype(np.uint8)
        all_labels = all_labels.astype(np.uint8)
        
        intersection = np.logical_and(all_preds, all_labels).sum()
        union = np.logical_or(all_preds, all_labels).sum()
        
        if union == 0:
            return 0.0
        return intersection / union
    else:
        preds_class = np.argmax(all_preds, axis=1)
        
        ious = []
        for cls in range(num_classes):
            pred_mask = (preds_class == cls)
            label_mask = (all_labels == cls)
            
            intersection = np.logical_and(pred_mask, label_mask).sum()
            union = np.logical_or(pred_mask, label_mask).sum()
            
            if union > 0:
                ious.append(intersection / union)
        
        return np.mean(ious) if ious else 0.0