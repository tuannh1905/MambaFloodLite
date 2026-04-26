import os
import sys
import time
import json
import random
import zipfile
import argparse
import warnings

import cv2
import gdown
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2


FLOODSCENE_CLASSES = {0: 0, 85: 1, 170: 2, 255: 3}
NUM_CLASSES = 4
CLASS_NAMES = ["background", "sky", "building", "flood"]

DATASETS = {
    "floodscene": {
        "id": "12f0UmV46uwzjMyYPpfc_S0fHLRe4LKIf",
        "dir": "floodscene",
    }
}


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(False)
    warnings.filterwarnings("ignore", category=UserWarning)
    print(f"✓ Seed set to {seed}")


def download_dataset(name):
    cfg = DATASETS[name]
    folder = cfg["dir"]
    if os.path.exists(folder):
        print(f"{name} exists. Skipping.")
        return
    url = f'https://drive.google.com/uc?id={cfg["id"]}'
    output = f"{name}.zip"
    print(f"Downloading {name}...")
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile(output, "r") as z:
        z.extractall(folder)
    os.remove(output)
    print(f"✓ {name} ready.")


class FloodSceneFullDataset(Dataset):
    def __init__(self, root_dir, split="train", size=256, seed=42):
        self.size = size
        self.seed = seed
        self.split = split

        self.root_dir = os.path.join(root_dir, split)
        self.images_dir = os.path.join(self.root_dir, "images")
        self.masks_dir = os.path.join(self.root_dir, "masks")

        self.images = sorted([
            f for f in os.listdir(self.images_dir)
            if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg"]
        ])

        self.lut = np.zeros(256, dtype=np.uint8)
        for px, cls in FLOODSCENE_CLASSES.items():
            self.lut[px] = cls

        if split == "train":
            self.transform = A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                ToTensorV2()
            ], is_check_shapes=False)
        else:
            self.transform = A.Compose([
                A.Resize(size, size),
                ToTensorV2()
            ], is_check_shapes=False)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

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

        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.masks_dir, base_name + ".png")
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask_img.shape[:2] != image.shape[:2]:
            mask_img = cv2.resize(mask_img, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        mask = self.lut[mask_img].astype(np.uint8)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"].float() / 255.0
        mask = transformed["mask"].long()

        return image, mask


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)


def get_dataloaders(dataset_path, batch_size=4, size=256, seed=42):
    train_dataset = FloodSceneFullDataset(dataset_path, "train", size, seed)
    val_dataset = FloodSceneFullDataset(dataset_path, "val", size, seed)
    test_dataset = FloodSceneFullDataset(dataset_path, "test", size, seed)

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


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=4):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss_ce = self.ce(inputs, targets)

        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = probs.sum(dims) + targets_one_hot.sum(dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss_dice = 1.0 - dice_per_class.mean()

        return 0.5 * loss_ce + 0.5 * loss_dice


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.ce(inputs, targets)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, num_classes=4):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def get_loss(loss_name, num_classes=4):
    loss_name = loss_name.lower()
    if loss_name in ("bce_dice", "bce-dice", "bcedice", "dice_ce", "ce_dice"):
        return CrossEntropyDiceLoss(num_classes=num_classes)
    elif loss_name in ("ce", "cross_entropy", "crossentropy"):
        return CrossEntropyLoss(num_classes=num_classes)
    elif loss_name in ("focal",):
        return FocalLoss(num_classes=num_classes)
    else:
        print(f"Unknown loss '{loss_name}', falling back to CrossEntropyDiceLoss.")
        return CrossEntropyDiceLoss(num_classes=num_classes)


def calculate_miou(all_preds, all_labels, num_classes=4):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    preds_class = np.argmax(all_preds, axis=1)
    if all_labels.ndim == 4:
        all_labels = all_labels.squeeze(1)

    preds_flat = preds_class.reshape(-1)
    labels_flat = all_labels.reshape(-1)

    ious = []
    for cls in range(num_classes):
        pred_mask = preds_flat == cls
        label_mask = labels_flat == cls
        intersection = np.logical_and(pred_mask, label_mask).sum()
        union = np.logical_or(pred_mask, label_mask).sum()
        if union > 0:
            ious.append(float(intersection / union))

    return float(np.mean(ious)) if ious else 0.0


def calculate_dice_score(all_preds, all_labels, num_classes=4):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    preds_class = np.argmax(all_preds, axis=1)
    if all_labels.ndim == 4:
        all_labels = all_labels.squeeze(1)

    preds_flat = preds_class.reshape(-1)
    labels_flat = all_labels.reshape(-1)

    dice_scores = []
    for cls in range(num_classes):
        pred_mask = preds_flat == cls
        label_mask = labels_flat == cls
        intersection = np.logical_and(pred_mask, label_mask).sum()
        dice = (2.0 * intersection) / (pred_mask.sum() + label_mask.sum() + 1e-8)
        if pred_mask.sum() > 0 or label_mask.sum() > 0:
            dice_scores.append(float(dice))

    return float(np.mean(dice_scores)) if dice_scores else 0.0


def calculate_pixel_accuracy(all_preds, all_labels, num_classes=4):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    preds_class = np.argmax(all_preds, axis=1)
    if all_labels.ndim == 4:
        all_labels = all_labels.squeeze(1)

    preds_flat = preds_class.reshape(-1)
    labels_flat = all_labels.reshape(-1)

    return float((preds_flat == labels_flat).sum() / preds_flat.size)


def calculate_per_class_iou(all_preds, all_labels, num_classes=4):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    preds_class = np.argmax(all_preds, axis=1)
    if all_labels.ndim == 4:
        all_labels = all_labels.squeeze(1)

    preds_flat = preds_class.reshape(-1)
    labels_flat = all_labels.reshape(-1)

    per_class = {}
    for cls in range(num_classes):
        pred_mask = preds_flat == cls
        label_mask = labels_flat == cls
        intersection = np.logical_and(pred_mask, label_mask).sum()
        union = np.logical_or(pred_mask, label_mask).sum()
        per_class[CLASS_NAMES[cls]] = float(intersection / union) if union > 0 else 0.0

    return per_class


def calculate_model_complexity(model, input_size=(1, 3, 256, 256), device="cuda"):
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)

    dummy_input = torch.randn(input_size).to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.no_grad():
        _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        memory_allocated = 0.0

    total_ops = 0
    hooks = []

    def count_ops_hook(module, input, output):
        nonlocal total_ops
        if isinstance(module, torch.nn.Conv2d):
            batch_size = input[0].size(0)
            out_h = output.size(2)
            out_w = output.size(3)
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            total_ops += batch_size * kernel_ops * out_h * out_w * module.out_channels
        elif isinstance(module, torch.nn.Linear):
            batch_size = input[0].size(0)
            total_ops += batch_size * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(count_ops_hook))

    with torch.no_grad():
        _ = model(dummy_input)

    for hook in hooks:
        hook.remove()

    gflops = total_ops / 1e9

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "memory_mb": memory_allocated,
        "gflops": gflops,
    }


def measure_inference_time(model, input_size=(1, 3, 256, 256), device="cuda", warmup=10, iterations=100):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times)
    avg_time = np.mean(times)

    return {
        "avg_time_s": avg_time,
        "std_time_s": np.std(times),
        "min_time_s": np.min(times),
        "max_time_s": np.max(times),
        "fps": 1.0 / avg_time,
        "latency_ms": avg_time * 1000,
    }


def train_floodscene(model_name, loss_name, size, epochs, batch_size, lr,
                     dataset_path, output_path, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    set_seed(seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_path=dataset_path, batch_size=batch_size, size=size, seed=seed
    )
    print(f"✓ DataLoaders ready")
    print(f"  Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}")

    set_seed(seed)

    from models import get_model
    model = get_model(model_name, num_classes=NUM_CLASSES, seed=seed, input_size=size)
    model = model.to(device).float()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model: {model_name} | Params: {total_params:,} | Trainable: {trainable_params:,}")

    print(f"\n{'='*70}")
    print("MODEL COMPLEXITY ANALYSIS")
    print(f"{'='*70}")
    complexity = calculate_model_complexity(model, input_size=(1, 3, size, size), device=device)
    print(f"Total Parameters:     {complexity['total_params']:,}")
    print(f"Trainable Parameters: {complexity['trainable_params']:,}")
    print(f"Model Size:           {complexity['model_size_mb']:.2f} MB")
    print(f"Peak Memory Usage:    {complexity['memory_mb']:.2f} MB")
    print(f"GFLOPs:               {complexity['gflops']:.4f}")

    print(f"\n{'='*70}")
    print("INFERENCE PERFORMANCE")
    print(f"{'='*70}")
    inference_stats = measure_inference_time(model, input_size=(1, 3, size, size), device=device)
    print(f"Average Inference Time: {inference_stats['avg_time_s']*1000:.4f} ms (± {inference_stats['std_time_s']*1000:.4f} ms)")
    print(f"Min Inference Time:     {inference_stats['min_time_s']*1000:.4f} ms")
    print(f"Max Inference Time:     {inference_stats['max_time_s']*1000:.4f} ms")
    print(f"FPS:                    {inference_stats['fps']:.2f}")
    print(f"Latency:                {inference_stats['latency_ms']:.4f} ms")
    print(f"{'='*70}\n")

    set_seed(seed)

    criterion = get_loss(loss_name, num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"✓ Loss: {loss_name} | Optimizer: Adam | LR: {lr}")

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, f"{model_name}_{loss_name}_floodscene_full_s{seed}.pth")

    best_val_loss = float("inf")

    print(f"\n{'='*70}")
    print(f"TRAINING START — {epochs} EPOCHS | FloodScene Full ({NUM_CLASSES} classes)")
    print(f"{'='*70}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, ncols=100)
        for images, masks in pbar:
            images = images.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)

            optimizer.zero_grad(set_to_none=False)
            outputs = model(images)

            aux_weight = 1.0 if "litev8" in model_name.lower() else 0.4

            if isinstance(outputs, tuple):
                loss = criterion(outputs[0], masks)
                for aux_out in outputs[1:]:
                    loss += aux_weight * criterion(aux_out, masks)
            else:
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, ncols=100):
                images = images.to(device, non_blocking=False)
                masks = masks.to(device, non_blocking=False)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}", end="")
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "seed": seed,
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "complexity": complexity,
                "inference_stats": inference_stats,
                "config": {
                    "model": model_name, "loss": loss_name,
                    "dataset": "floodscene_full", "num_classes": NUM_CLASSES,
                    "batch_size": batch_size, "lr": lr, "size": size,
                },
            }
            torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
            print(" ✓ Best")
        else:
            print()

    print(f"\n{'='*70}")
    print("TESTING")
    print(f"{'='*70}")

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing", ncols=100):
            images = images.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)

            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            loss = criterion(outputs, masks)
            test_loss += loss.item()

            preds = torch.softmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(masks.cpu().numpy())

    miou = calculate_miou(all_preds, all_labels, NUM_CLASSES)
    dice = calculate_dice_score(all_preds, all_labels, NUM_CLASSES)
    pixel_acc = calculate_pixel_accuracy(all_preds, all_labels, NUM_CLASSES)
    per_class_iou = calculate_per_class_iou(all_preds, all_labels, NUM_CLASSES)
    avg_test_loss = test_loss / len(test_loader)

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Test Loss:        {avg_test_loss:.10f}")
    print(f"mIOU:             {miou:.10f}")
    print(f"Dice Score:       {dice:.10f}")
    print(f"Pixel Accuracy:   {pixel_acc:.10f}")
    print(f"Best Val Loss:    {best_val_loss:.10f}")
    print(f"\nPer-Class IoU:")
    for cls_name, iou_val in per_class_iou.items():
        print(f"  {cls_name:<12}: {iou_val:.6f}")
    print(f"\n{'='*70}")
    print("MODEL COMPLEXITY")
    print(f"{'='*70}")
    print(f"Parameters:       {complexity['total_params']:,}")
    print(f"Model Size:       {complexity['model_size_mb']:.2f} MB")
    print(f"Memory Usage:     {complexity['memory_mb']:.2f} MB")
    print(f"GFLOPs:           {complexity['gflops']:.4f}")
    print(f"\n{'='*70}")
    print("INFERENCE PERFORMANCE")
    print(f"{'='*70}")
    print(f"Avg Inference:    {inference_stats['avg_time_s']*1000:.4f} ms")
    print(f"FPS:              {inference_stats['fps']:.2f}")
    print(f"Latency:          {inference_stats['latency_ms']:.4f} ms")
    print(f"\n{'='*70}")
    print(f"Saved:            {save_path}")
    print(f"{'='*70}\n")

    return {
        "test_loss": avg_test_loss,
        "miou": miou,
        "dice": dice,
        "pixel_accuracy": pixel_acc,
        "best_val_loss": best_val_loss,
        "per_class_iou": per_class_iou,
        "model_path": save_path,
        "complexity": complexity,
        "inference_stats": inference_stats,
    }


def verify_reproducibility(args, num_runs=2):
    print(f"\n{'#'*70}")
    print("REPRODUCIBILITY TEST")
    print(f"{'#'*70}\n")

    results = []
    for run in range(num_runs):
        print(f"\nRUN {run+1}/{num_runs} with seed={args.seed}")
        set_seed(args.seed)
        result = train_floodscene(
            model_name=args.model, loss_name=args.loss, size=args.size,
            epochs=min(args.epochs, 5), batch_size=args.batch_size, lr=args.lr,
            dataset_path=args.dataset, output_path=os.path.join(args.output_path, f"repro_run{run}"),
            seed=args.seed,
        )
        results.append(result)

    print(f"\n{'='*70}")
    print("VERIFICATION RESULTS")
    print(f"{'='*70}")
    passed = True
    for i in range(1, num_runs):
        loss_diff = abs(results[0]["test_loss"] - results[i]["test_loss"])
        miou_diff = abs(results[0]["miou"] - results[i]["miou"])
        print(f"Run 1 vs {i+1}: Loss diff={loss_diff:.10f} | mIOU diff={miou_diff:.10f}")
        if loss_diff > 1e-6 or miou_diff > 1e-6:
            print("  ❌ FAIL")
            passed = False
        else:
            print("  ✅ PASS")

    print(f"{'='*70}")
    print("✅ ALL TESTS PASSED" if passed else "❌ REPRODUCIBILITY FAILED")
    print(f"{'='*70}\n")
    return passed


def run_multiseed_experiments(args, seeds):
    print(f"\n{'#'*70}")
    print(f"MULTI-SEED EXPERIMENT — Seeds: {seeds}")
    print(f"{'#'*70}\n")

    results = []
    for seed in seeds:
        print(f"\n{'='*70}\nSEED: {seed}\n{'='*70}\n")
        set_seed(seed)
        result = train_floodscene(
            model_name=args.model, loss_name=args.loss, size=args.size,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            dataset_path=args.dataset, output_path=os.path.join(args.output_path, f"seed_{seed}"),
            seed=seed,
        )
        results.append({
            "seed": seed,
            "test_loss": result["test_loss"],
            "miou": result["miou"],
            "dice": result["dice"],
            "pixel_accuracy": result["pixel_accuracy"],
            "best_val_loss": result["best_val_loss"],
            "per_class_iou": result["per_class_iou"],
            "total_params": result["complexity"]["total_params"],
            "model_size_mb": result["complexity"]["model_size_mb"],
            "memory_mb": result["complexity"]["memory_mb"],
            "gflops": result["complexity"]["gflops"],
            "fps": result["inference_stats"]["fps"],
            "latency_ms": result["inference_stats"]["latency_ms"],
            "avg_inference_ms": result["inference_stats"]["avg_time_s"] * 1000,
        })

    losses = [r["test_loss"] for r in results]
    mious = [r["miou"] for r in results]
    dices = [r["dice"] for r in results]
    pixel_acc = [r["pixel_accuracy"] for r in results]
    val_losses = [r["best_val_loss"] for r in results]
    fps_list = [r["fps"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    per_class_means = {}
    for cls_name in CLASS_NAMES:
        vals = [r["per_class_iou"][cls_name] for r in results]
        per_class_means[cls_name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    print(f"\n{'='*70}\nSTATISTICS FOR PAPER\n{'='*70}")
    print(f"Test Loss:      {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f"mIOU:           {np.mean(mious):.4f} ± {np.std(mious):.4f}")
    print(f"Dice:           {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Pixel Accuracy: {np.mean(pixel_acc):.4f} ± {np.std(pixel_acc):.4f}")
    print(f"FPS:            {np.mean(fps_list):.2f} ± {np.std(fps_list):.2f}")
    print(f"Latency:        {np.mean(latencies):.4f} ± {np.std(latencies):.4f} ms")
    print(f"\nPer-Class IoU:")
    for cls_name, stats in per_class_means.items():
        print(f"  {cls_name:<12}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"\nLaTeX:")
    print(f"mIOU: ${np.mean(mious):.4f} \\pm {np.std(mious):.4f}$")
    print(f"Dice: ${np.mean(dices):.4f} \\pm {np.std(dices):.4f}$")
    print(f"FPS:  ${np.mean(fps_list):.2f} \\pm {np.std(fps_list):.2f}$")
    print(f"{'='*70}\n")

    result_file = os.path.join(args.output_path, f"{args.model}_floodscene_full_multiseed.json")
    with open(result_file, "w") as f:
        json.dump({
            "config": {
                "model": args.model, "dataset": "floodscene_full",
                "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
                "loss": args.loss, "epochs": args.epochs,
                "batch_size": args.batch_size, "lr": args.lr, "size": args.size,
            },
            "seeds": seeds,
            "results": results,
            "statistics": {
                "test_loss": {"mean": float(np.mean(losses)), "std": float(np.std(losses))},
                "miou": {"mean": float(np.mean(mious)), "std": float(np.std(mious))},
                "dice": {"mean": float(np.mean(dices)), "std": float(np.std(dices))},
                "pixel_accuracy": {"mean": float(np.mean(pixel_acc)), "std": float(np.std(pixel_acc))},
                "val_loss": {"mean": float(np.mean(val_losses)), "std": float(np.std(val_losses))},
                "fps": {"mean": float(np.mean(fps_list)), "std": float(np.std(fps_list))},
                "latency_ms": {"mean": float(np.mean(latencies)), "std": float(np.std(latencies))},
                "per_class_iou": per_class_means,
            },
            "complexity": {
                "total_params": results[0]["total_params"],
                "model_size_mb": results[0]["model_size_mb"],
                "memory_mb": results[0]["memory_mb"],
                "gflops": results[0]["gflops"],
            },
        }, f, indent=2)

    print(f"Results saved: {result_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="FloodScene Full Multi-Class Benchmark")
    parser.add_argument("--dataset", type=str, default="floodscene",
                        choices=["floodscene"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--loss", type=str, default="bce_dice")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify_repro", action="store_true")
    parser.add_argument("--multiseed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024])
    args = parser.parse_args()

    set_seed(args.seed)

    if args.download:
        download_dataset(args.dataset)

    dataset_path = DATASETS[args.dataset]["dir"]

    print("=" * 70)
    print(f"FloodScene FULL ({NUM_CLASSES} classes): {CLASS_NAMES}")
    print(f"Model: {args.model} | Size: {args.size} | Loss: {args.loss}")
    print(f"Epochs: {args.epochs} | BS: {args.batch_size} | LR: {args.lr}")
    print("=" * 70)

    if args.verify_repro:
        args.dataset = dataset_path
        verify_reproducibility(args, num_runs=2)
        return

    if args.multiseed:
        args.dataset = dataset_path
        run_multiseed_experiments(args, seeds=args.seeds)
        return

    train_floodscene(
        model_name=args.model, loss_name=args.loss, size=args.size,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        dataset_path=dataset_path, output_path=args.output_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
