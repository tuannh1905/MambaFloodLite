import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class FloodSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', size=512, seed=42, num_classes=1, dataset_type='floodkaggle'):
        self.dataset_type = dataset_type
        self.size = size
        self.seed = seed
        self.num_classes = num_classes
        self.split = split

        self.root_dir   = os.path.join(root_dir, split)
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.masks_dir  = os.path.join(self.root_dir, 'masks')

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

        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id if worker_info is not None else 0

        filename_hash = hash(img_name) % 1000000
        sample_seed   = self.seed + filename_hash + worker_id * 10000000

        np.random.seed(sample_seed)
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)

        img_path = os.path.join(self.images_dir, img_name)
        image    = cv2.imread(img_path)
        image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.masks_dir, base_name + '.png')
        mask_img  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask      = (mask_img > 127).astype(np.uint8)

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image'].float() / 255.0

        if transformed['mask'] is None:
            raise ValueError(f"Mask bị None sau khi transform tại index {idx}")

        if self.num_classes == 1:
            mask = transformed['mask'].unsqueeze(0).float()
        else:
            mask = transformed['mask'].long()

        return image, mask


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)


def get_dataloaders(dataset, batch_size=4, size=256, seed=42, num_classes=1, dataset_type='floodkaggle'):
    train_dataset = FloodSegmentationDataset(dataset, 'train', size, seed, num_classes, dataset_type=dataset_type)
    val_dataset   = FloodSegmentationDataset(dataset, 'val',   size, seed, num_classes, dataset_type=dataset_type)
    test_dataset  = FloodSegmentationDataset(dataset, 'test',  size, seed, num_classes, dataset_type=dataset_type)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=torch.cuda.is_available(),
        drop_last=True, worker_init_fn=seed_worker,
        generator=g, persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available(),
        drop_last=False, worker_init_fn=seed_worker,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available(),
        drop_last=False, worker_init_fn=seed_worker,
        persistent_workers=False
    )

    return train_loader, val_loader, test_loader