import os
import torch
import json
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class FloodSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', size=512, seed=42, num_classes=1, dataset_type='floodvn'):
        self.dataset_type = dataset_type
        self.size = size
        self.seed = seed
        self.num_classes = num_classes
        self.split = split
        
        # ✅ SỬA: Cả hai dataset đều dùng folder images và labels
        self.root_dir = os.path.join(root_dir, split)
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.labels_dir = os.path.join(self.root_dir, 'labels')
        
        self.images = sorted([
            img for img in os.listdir(self.images_dir)
            if os.path.splitext(img)[1].lower() in [".png", ".jpg", ".jpeg"]
        ])
        
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
        img_name = self.images[idx]
        
        # CRITICAL: Deterministic per-sample seed
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        
        filename_hash = hash(img_name) % 1000000
        sample_seed = self.seed + filename_hash + worker_id * 10000000
        
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           
         # FloodVN dùng file JSON
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
        
        # ✅ THÊM: Kiểm tra mask sau transform
        if transformed['mask'] is None:
            raise ValueError(f"Mask bị None sau khi transform tại index {idx}")
        
        if self.num_classes == 1:
            mask = transformed['mask'].unsqueeze(0).float()
        else:
            mask = transformed['mask'].long()
        
        return image, mask

def seed_worker(worker_id):
    """Initialize worker with deterministic seed based on global seed + worker_id"""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)

def get_dataloaders(dataset, batch_size=4, size=256, seed=42, num_classes=1, dataset_type='floodvn'):
    """
    ✅ FIXED: Added dataset_type parameter
    Create DataLoaders with STRICT reproducibility using num_workers=4
    """
    
    train_dataset = FloodSegmentationDataset(
        dataset, 'train', size, seed, num_classes, dataset_type=dataset_type
    )
    val_dataset = FloodSegmentationDataset(
        dataset, 'val', size, seed, num_classes, dataset_type=dataset_type
    )
    test_dataset = FloodSegmentationDataset(
        dataset, 'test', size, seed, num_classes, dataset_type=dataset_type
    )
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=False
    )
    
    return train_loader, val_loader, test_loader