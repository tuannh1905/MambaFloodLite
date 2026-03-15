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
import warnings

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
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
        
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    torch.use_deterministic_algorithms(False)
    
    warnings.filterwarnings('ignore', category=UserWarning)
    
    print(f"✓ Seed set to {seed} (STRICT MODE)")

    
DATASETS = {
    'floodvn': {'id': '1tQYUVtSdYJ3cGn1oftmb9MeWrmu4ez7P', 'dir': 'floodvn'},
    'floodkaggle': {'id': '1tg3N5DW27LWgJ9cTvNeIUz5xoBqSrmEs', 'dir': 'floodkaggle'}
}

def download_dataset(name):
    folder = name
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
    print(f"✓ {name.capitalize()} ready.")


def verify_reproducibility(args, num_runs=2):
    print(f"\n{'#'*70}")
    print("REPRODUCIBILITY TEST")
    print(f"{'#'*70}\n")
    
    results = []
    for run in range(num_runs):
        print(f"\nRUN {run+1}/{num_runs} with seed={args.seed}")
        set_seed(args.seed)
        
        from utils.trainer import train_segmentation
        
        dataset_path = args.dataset
        
        result = train_segmentation(
            model_name=args.model,
            loss_name=args.loss,
            size=args.size,
            epochs=min(args.epochs, 5),
            batch_size=args.batch_size,
            lr=args.lr,
            dataset=dataset_path,
            output_path=os.path.join(args.output_path, f'repro_run{run}'),
            seed=args.seed,
            num_classes=1,
            dataset_type=args.dataset
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
        
        dataset_path = args.dataset
        
        result = train_segmentation(
            model_name=args.model,
            loss_name=args.loss,
            size=args.size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dataset=dataset_path,
            output_path=os.path.join(args.output_path, f'seed_{seed}'),
            seed=seed,
            num_classes=1,
            dataset_type=args.dataset
        )
        results.append({
            'seed': seed,
            'test_loss': result['test_loss'],
            'miou': result['miou'],
            'dice': result['dice'],
            'pixel_accuracy': result['pixel_accuracy'],
            'best_val_loss': result['best_val_loss'],
            'total_params': result['complexity']['total_params'],
            'model_size_mb': result['complexity']['model_size_mb'],
            'memory_mb': result['complexity']['memory_mb'],
            'gflops': result['complexity']['gflops'],
            'fps': result['inference_stats']['fps'],
            'latency_ms': result['inference_stats']['latency_ms'],
            'avg_inference_ms': result['inference_stats']['avg_time_s'] * 1000
        })
    
    losses = [r['test_loss'] for r in results]
    mious = [r['miou'] for r in results]
    dices = [r['dice'] for r in results]
    pixel_accs = [r['pixel_accuracy'] for r in results]
    val_losses = [r['best_val_loss'] for r in results]
    fps_list = [r['fps'] for r in results]
    latencies = [r['latency_ms'] for r in results]
    
    print(f"\n{'='*70}")
    print("STATISTICS FOR PAPER")
    print(f"{'='*70}")
    print(f"\nACCURACY METRICS:")
    print(f"  Test Loss:       {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f"  mIOU:            {np.mean(mious):.4f} ± {np.std(mious):.4f}")
    print(f"  Dice Score:      {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"  Pixel Accuracy:  {np.mean(pixel_accs):.4f} ± {np.std(pixel_accs):.4f}")
    print(f"  Val Loss:        {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    
    print(f"\nMODEL COMPLEXITY:")
    print(f"  Parameters:      {results[0]['total_params']:,}")
    print(f"  Model Size:      {results[0]['model_size_mb']:.2f} MB")
    print(f"  Memory Usage:    {results[0]['memory_mb']:.2f} MB")
    print(f"  GFLOPs:          {results[0]['gflops']:.4f}")
    
    print(f"\nINFERENCE PERFORMANCE:")
    print(f"  FPS:             {np.mean(fps_list):.2f} ± {np.std(fps_list):.2f}")
    print(f"  Latency:         {np.mean(latencies):.4f} ± {np.std(latencies):.4f} ms")
    print(f"  Avg Inference:   {np.mean([r['avg_inference_ms'] for r in results]):.4f} ms")
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
                'dice': {'mean': float(np.mean(dices)), 'std': float(np.std(dices))},
                'pixel_accuracy': {'mean': float(np.mean(pixel_accs)), 'std': float(np.std(pixel_accs))},
                'val_loss': {'mean': float(np.mean(val_losses)), 'std': float(np.std(val_losses))},
                'fps': {'mean': float(np.mean(fps_list)), 'std': float(np.std(fps_list))},
                'latency_ms': {'mean': float(np.mean(latencies)), 'std': float(np.std(latencies))}
            },
            'complexity': {
                'total_params': results[0]['total_params'],
                'model_size_mb': results[0]['model_size_mb'],
                'memory_mb': results[0]['memory_mb'],
                'gflops': results[0]['gflops']
            }
        }, f, indent=2)
    
    print(f"📊 Results saved to: {result_file}")
    print(f"\nLaTeX format:")
    print(f"Test Loss: ${np.mean(losses):.4f} \\pm {np.std(losses):.4f}$")
    print(f"mIOU: ${np.mean(mious):.4f} \\pm {np.std(mious):.4f}$")
    print(f"Dice: ${np.mean(dices):.4f} \\pm {np.std(dices):.4f}$")
    print(f"FPS: ${np.mean(fps_list):.2f} \\pm {np.std(fps_list):.2f}$")
    print(f"Latency: ${np.mean(latencies):.4f} \\pm {np.std(latencies):.4f}$ ms\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Flood Detection Training Benchmark')
    
    parser.add_argument('--dataset', type=str, default='floodvn', 
                        choices=['floodvn', 'floodkaggle'])
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
    
    num_classes = 1
    
    if args.size is None:
        args.size = 256
    
    dataset_path = args.dataset
    
    if args.verify_repro:
        verify_reproducibility(args, num_runs=2)
        return
    
    if args.multiseed:
        run_multiseed_experiments(args, seeds=args.seeds)
        return
    
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Dataset:       {args.dataset}")
    print(f"Dataset Path:  {dataset_path}")
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
        dataset=dataset_path,
        output_path=args.output_path,
        seed=args.seed,
        num_classes=num_classes,
        dataset_type=args.dataset
    )

if __name__ == '__main__':
    main()