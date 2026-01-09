import os
import torch
import torch.nn as nn
from tqdm import tqdm

def set_seed(seed):
    """Reset seed - call before every major operation"""
    import random
    import numpy as np
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_segmentation(model_name, loss_name, size, epochs, batch_size, lr, 
                       dataset, output_path, seed, num_classes=1):
    """Train with ABSOLUTE reproducibility - num_workers=4"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    set_seed(seed)
    
    from utils.dataloader import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset, batch_size=batch_size, size=size, 
        seed=seed, num_classes=num_classes
    )
    
    print(f"✓ DataLoaders: num_workers=4, persistent_workers=False")
    print(f"  Train: {len(train_loader)} batches (drop_last=True)")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")
    
    set_seed(seed)
    
    from models import get_model
    model = get_model(model_name, num_classes=num_classes, seed=seed)
    model = model.to(device)
    model = model.float()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model: {model_name}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    set_seed(seed)
    
    from losses import get_loss
    criterion = get_loss(loss_name, num_classes=num_classes)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        fused=False
    )
    
    print(f"✓ Optimizer: Adam (fused=False for determinism)")
    print(f"✓ Loss: {loss_name}")
    
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, f'{model_name}_{loss_name}_{dataset}_s{seed}.pth')
    
    best_val_loss = float('inf')
    
    print(f"\n{'='*70}")
    print(f"TRAINING START - {epochs} EPOCHS")
    print(f"{'='*70}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                   leave=False, ncols=100)
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)
            
            optimizer.zero_grad(set_to_none=False)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                                     leave=False, ncols=100):
                images = images.to(device, non_blocking=False)
                masks = masks.to(device, non_blocking=False)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {avg_train_loss:.6f} | "
              f"Val: {avg_val_loss:.6f}", end='')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'seed': seed,
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'config': {
                    'model': model_name,
                    'loss': loss_name,
                    'dataset': dataset,
                    'batch_size': batch_size,
                    'lr': lr,
                    'size': size,
                    'num_classes': num_classes
                }
            }
            
            torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
            print(" ✓ Best")
        else:
            print()
    
    print(f"\n{'='*70}")
    print("TESTING")
    print(f"{'='*70}")
    
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing", ncols=100):
            images = images.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            
            if num_classes == 1:
                preds = torch.sigmoid(outputs)
            else:
                preds = torch.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(masks.cpu().numpy())
    
    from utils.metrics import calculate_miou
    miou = calculate_miou(all_preds, all_labels, num_classes)
    
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Test Loss:     {avg_test_loss:.10f}")
    print(f"mIOU:          {miou:.10f}")
    print(f"Best Val Loss: {best_val_loss:.10f}")
    print(f"Saved:         {save_path}")
    print(f"{'='*70}\n")
    
    return {
        'test_loss': avg_test_loss,
        'miou': miou,
        'best_val_loss': best_val_loss,
        'model_path': save_path
    }