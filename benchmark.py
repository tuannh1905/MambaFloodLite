import os
import sys
import torch
import random
import numpy as np
import argparse
import gdown
import shutil
import zipfile

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DATASETS = {
    'floodvn': {'id': '1GM77HHYx45XnabUsk9g-f150rScGXoC8', 'dir': 'floodvn'},
    'floodkaggle': {'id': '1hue0azsiVoCrFRQ7FIn6xpKViw5ZEKlO', 'dir': 'floodkaggle'},
    'floodnet': {'id': '1IbbI5iomI7elrvERrGlgGB0V32KgOdIj', 'dir': '.'}
}

def download_dataset(name):
    folder = name if name != 'floodnet' else 'floodnet'
    if os.path.exists(folder):
        print(f"{name.capitalize()} exists. Skipping.")
        return
    
    cfg = DATASETS[name]
    url = f'https://drive.google.com/uc?id={cfg["id"]}'
    output = f'{name}.zip'
    
    print(f"Downloading {name.capitalize()}...")
    gdown.download(url, output, quiet=False)
    
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(cfg['dir'])
    
    os.remove(output)
    print(f"{name.capitalize()} ready.")

def main():
    parser = argparse.ArgumentParser(description='Flood Detection Training Benchmark')
    
    parser.add_argument('--mode', type=str, required=True, choices=['classification', 'segmentation'])
    parser.add_argument('--dataset', type=str, default='floodvn', 
                        choices=['floodvn', 'floodkaggle', 'floodnet'])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--size', type=int, default=None)
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_path', type=str, default='outputs')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    
    if args.download:
        download_dataset(args.dataset)
    
    num_classes = 1 if args.dataset in ['floodvn', 'floodkaggle'] else 10
    
    if args.size is None:
        args.size = 512 if args.mode == 'segmentation' else 320
    
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Mode:          {args.mode}")
    print(f"Dataset:       {args.dataset}")
    print(f"Model:         {args.model}")
    print(f"Size:          {args.size}")
    print(f"Loss:          {args.loss}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Seed:          {args.seed}")
    print(f"Num Classes:   {num_classes}")
    print(f"Output Path:   {args.output_path}")
    print("="*70)
    
    if args.mode == 'segmentation':
        from segmentation.utils.trainer import train_segmentation
        train_segmentation(
            model_name=args.model,
            loss_name=args.loss,
            size=args.size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dataset=args.dataset,
            output_path=args.output_path,
            seed=args.seed,
            num_classes=num_classes
        )

if __name__ == '__main__':
    main()