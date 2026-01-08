import os
import torch
import torch.nn as nn
from tqdm import tqdm
from segmentation.models import get_model
from segmentation.losses import get_loss
from segmentation.utils.dataloader import get_dataloaders
from segmentation.utils.metrics import calculate_miou

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def train_segmentation(model_name, loss_name, size, epochs, batch_size, lr, 
                       dataset, output_path, seed, num_classes=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\n{'='*70}")
    print(f"LOADING DATASET: {dataset.upper()}")
    print(f"{'='*70}")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        size=size,
        seed=seed,
        num_classes=num_classes
    )
    
    print(f"\n{'='*70}")
    print(f"LOADING MODEL: {model_name.upper()}")
    print(f"{'='*70}")
    print(f"Number of classes: {num_classes} ({'binary' if num_classes == 1 else 'multi-class'})")
    
    model = get_model(model_name, num_classes=num_classes).to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    print(f"\n{'='*70}")
    print(f"LOSS FUNCTION: {loss_name.upper()}")
    print(f"{'='*70}")
    criterion = get_loss(loss_name, num_classes=num_classes)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(
        output_path, 
        f'{model_name}_{loss_name}_{dataset}_nc{num_classes}_s{seed}.pth'
    )
    
    best_val_loss = float('inf')
    
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Image size: {size}")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', ncols=100)
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{train_loss/(batch_idx+1):.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]  ', ncols=100)
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{val_loss/(batch_idx+1):.4f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'\nEpoch {epoch+1}/{epochs} Summary:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss:   {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'num_classes': num_classes,
                'dataset': dataset
            }, save_path)
            print(f'  ✓ Best model saved! (Val Loss: {best_val_loss:.4f})')
        
        print(f"{'-'*70}\n")
    
    print(f"\n{'='*70}")
    print("TESTING ON BEST MODEL")
    print(f"{'='*70}")
    print(f"Loading best model from: {save_path}")
    
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing', ncols=100)
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            
            if num_classes == 1:
                preds = torch.sigmoid(outputs)
            else:
                preds = torch.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(masks.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{test_loss/(batch_idx+1):.4f}'
            })
    
    avg_test_loss = test_loss / len(test_loader)
    
    print("\nCalculating mIOU...")
    miou = calculate_miou(all_preds, all_labels, num_classes)
    
    print(f"\n{'='*70}")
    print("FINAL TEST RESULTS")
    print(f"{'='*70}")
    print(f"Dataset:       {dataset}")
    print(f"Model:         {model_name}")
    print(f"Loss:          {loss_name}")
    print(f"Image Size:    {size}")
    print(f"Num Classes:   {num_classes}")
    print(f"Test Loss:     {avg_test_loss:.4f}")
    print(f"mIOU:          {miou:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f} (from epoch {checkpoint['epoch']+1})")
    print(f"{'='*70}")
    print(f"Model saved at: {save_path}")
    print(f"{'='*70}\n")
    
    return {
        'test_loss': avg_test_loss,
        'miou': miou,
        'best_val_loss': best_val_loss,
        'model_path': save_path
    }