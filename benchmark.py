import os
import sys
import torch
import random
import numpy as np
import argparse
import gdown
import shutil
import zipfile
import json

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.use_deterministic_algorithms(True)
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, 'cuda'):
        if hasattr(torch.backends.cuda, 'matmul'):
            if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = False
    
DATASETS = {
    'floodvn': {'id': '1tQYUVtSdYJ3cGn1oftmb9MeWrmu4ez7P', 'dir': 'floodvn'},
    'floodkaggle': {'id': '1tg3N5DW27LWgJ9cTvNeIUz5xoBqSrmEs', 'dir': 'floodkaggle'},
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


def verify_reproducibility(args, num_runs=2):
    print(f"\n{'#'*70}")
    print("REPRODUCIBILITY TEST")
    print(f"{'#'*70}\n")
    
    results = []
    for run in range(num_runs):
        print(f"\nRUN {run+1}/{num_runs} with seed={args.seed}")
        set_seed(args.seed)
        
        from utils.trainer import train_segmentation
        result = train_segmentation(
            model_name=args.model,
            loss_name=args.loss,
            size=args.size,
            epochs=min(args.epochs, 5),
            batch_size=args.batch_size,
            lr=args.lr,
            dataset=args.dataset,
            output_path=os.path.join(args.output_path, f'repro_run{run}'),
            seed=args.seed,
            num_classes=1 if args.dataset in ['floodvn', 'floodkaggle'] else 10
        )
        results.append(result)
    
    print(f"\n{'='*70}")
    print("VERIFICATION RESULTS")
    print(f"{'='*70}")
    passed = True
    for i in range(1, num_runs):
        loss_diff = abs(results[0]['test_loss'] - results[i]['test_loss'])
        miou_diff = abs(results[0]['miou'] - results[i]['miou'])
        print(f"Run 1 vs {i+1}:")
        print(f"  Loss diff: {loss_diff:.10f}")
        print(f"  mIOU diff: {miou_diff:.10f}")
        if loss_diff > 1e-6 or miou_diff > 1e-6:
            print("  ❌ FAIL")
            passed = False
        else:
            print("  ✅ PASS")
    
    print(f"{'='*70}")
    if passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ REPRODUCIBILITY FAILED")
    print(f"{'='*70}\n")
    return passed


def run_multiseed_experiments(args, seeds):
    print(f"\n{'#'*70}")
    print(f"MULTI-SEED EXPERIMENT")
    print(f"Seeds: {seeds}")
    print(f"{'#'*70}\n")
    
    results = []
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED: {seed}")
        print(f"{'='*70}\n")
        
        set_seed(seed)
        from utils.trainer import train_segmentation
        result = train_segmentation(
            model_name=args.model,
            loss_name=args.loss,
            size=args.size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dataset=args.dataset,
            output_path=os.path.join(args.output_path, f'seed_{seed}'),
            seed=seed,
            num_classes=1 if args.dataset in ['floodvn', 'floodkaggle'] else 10
        )
        results.append({
            'seed': seed,
            'test_loss': result['test_loss'],
            'miou': result['miou'],
            'best_val_loss': result['best_val_loss']
        })
    
    losses = [r['test_loss'] for r in results]
    mious = [r['miou'] for r in results]
    val_losses = [r['best_val_loss'] for r in results]
    
    print(f"\n{'='*70}")
    print("STATISTICS FOR PAPER")
    print(f"{'='*70}")
    print(f"Test Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f"mIOU:      {np.mean(mious):.4f} ± {np.std(mious):.4f}")
    print(f"Val Loss:  {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"{'='*70}\n")
    
    result_file = os.path.join(args.output_path, f'{args.model}_{args.dataset}_multiseed.json')
    with open(result_file, 'w') as f:
        json.dump({
            'config': {
                'model': args.model,
                'dataset': args.dataset,
                'loss': args.loss,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'size': args.size
            },
            'seeds': seeds,
            'results': results,
            'statistics': {
                'test_loss': {'mean': float(np.mean(losses)), 'std': float(np.std(losses))},
                'miou': {'mean': float(np.mean(mious)), 'std': float(np.std(mious))},
                'val_loss': {'mean': float(np.mean(val_losses)), 'std': float(np.std(val_losses))}
            }
        }, f, indent=2)
    
    print(f"📊 Results saved to: {result_file}")
    print(f"\nLaTeX format:")
    print(f"Test Loss: ${np.mean(losses):.4f} \\pm {np.std(losses):.4f}$")
    print(f"mIOU: ${np.mean(mious):.4f} \\pm {np.std(mious):.4f}$\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Flood Detection Training Benchmark')
    
    parser.add_argument('--dataset', type=str, default='floodvn', 
                        choices=['floodvn', 'floodkaggle', 'floodnet'])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--size', type=int, default=None)
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_path', type=str, default='outputs')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--verify_repro', action='store_true', help='Verify reproducibility')
    parser.add_argument('--multiseed', action='store_true', help='Run multi-seed experiments')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 2024])
    
    args = parser.parse_args()

    set_seed(args.seed)
    
    if args.download:
        download_dataset(args.dataset)
    
    num_classes = 1 if args.dataset in ['floodvn', 'floodkaggle'] else 10
    
    if args.size is None:
        args.size = 256
    
    # NEW: Check for special modes
    if args.verify_repro:
        verify_reproducibility(args, num_runs=2)
        return
    
    if args.multiseed:
        run_multiseed_experiments(args, seeds=args.seeds)
        return
    
    # Normal training
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
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
    
    from utils.trainer import train_segmentation
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