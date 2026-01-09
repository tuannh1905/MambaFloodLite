import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random

class FloodSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', size=512, seed=42, num_classes=1, dataset_type='floodvn'):
        self.dataset_type = dataset_type
        self.size = size
        self.seed = seed
        self.num_classes = num_classes
        self.split = split  # ADD THIS
        
        random.seed(seed)
        np.random.seed(seed)
        
        if dataset_type == 'floodnet':
            self.root_dir = os.path.join(root_dir, split)
            self.images_dir = os.path.join(self.root_dir, 'image')
            self.labels_dir = os.path.join(self.root_dir, 'mask')
        else:
            self.root_dir = os.path.join(root_dir, split)
            self.images_dir = os.path.join(self.root_dir, 'images')
            self.labels_dir = os.path.join(self.root_dir, 'labels')
        
        self.images = []
        if os.path.exists(self.images_dir):
            for img_name in sorted(os.listdir(self.images_dir)):
                ext = os.path.splitext(img_name)[1].lower()
                if ext in [".png", ".jpg", ".jpeg"]:
                    if dataset_type == 'floodnet':
                        mask_name = img_name
                        mask_path = os.path.join(self.labels_dir, mask_name)
                    else:
                        label_name = os.path.splitext(img_name)[0] + ".json"
                        mask_path = os.path.join(self.labels_dir, label_name)
                    
                    if os.path.exists(mask_path):
                        self.images.append(img_name)
        
        if split == 'train':
            self.transform = A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(size, size),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # ==================== CRITICAL FIX ====================
        # Set unique seed per sample for deterministic augmentation
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        
        # Formula: base_seed + sample_index + worker_offset
        # This ensures each sample gets same augmentation across runs
        sample_seed = self.seed + idx + worker_id * 100000
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        # ======================================================
        
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.dataset_type == 'floodnet':
            mask_path = os.path.join(self.labels_dir, img_name)
            mask = np.array(Image.open(mask_path))
        else:
            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(self.labels_dir, base_name + '.json')
            
            with open(label_path, 'r') as f:
                label_data = json.load(f)
            
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            if 'shapes' in label_data:
                for shape in label_data['shapes']:
                    points = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [points], 1)
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image'].float() / 255.0
        
        if self.num_classes == 1:
            mask = transformed['mask'].unsqueeze(0).float()
        else:
            mask = transformed['mask'].long()
        
        return image, mask


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(dataset, batch_size=4, size=256, seed=42, num_classes=1):
    print(f"Using {dataset} dataset with {num_classes} classes")
    
    train_dataset = FloodSegmentationDataset(dataset, 'train', size, seed, num_classes, dataset)
    val_dataset = FloodSegmentationDataset(dataset, 'val', size, seed, num_classes, dataset)
    test_dataset = FloodSegmentationDataset(dataset, 'test', size, seed, num_classes, dataset)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    
    return train_loader, val_loader, test_loader