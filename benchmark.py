import os
import sys
import torch
import random
import numpy as np
import argparse
import gdown
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
    'floodkaggle': {
        'id': '1uMgjTSxseBuxH65VhLdN_4_rlMihamHa',
        #'id': '1LwWCPk3gsPDiRAnB_RUVffCky-j8yNO0',
        'dir': 'floodkaggle'
    },
    'floodscene': {
        'id': '12f0UmV46uwzjMyYPpfc_S0fHLRe4LKIf',
        'dir': 'floodscene'
    }
}


def download_dataset(name):
    cfg    = DATASETS[name]
    folder = cfg['dir']
    if os.path.exists(folder):
        print(f"{name} exists. Skipping.")
        return
    url    = f'https://drive.google.com/uc?id={cfg["id"]}'
    output = f'{name}.zip'
    print(f"Downloading {name}...")
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile(output, 'r') as z:
        z.extractall(folder)
    os.remove(output)
    print(f"✓ {name} ready.")


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
            model_name=args.model, loss_name=args.loss, size=args.size,
            epochs=min(args.epochs, 5), batch_size=args.batch_size, lr=args.lr,
            dataset=args.dataset, output_path=os.path.join(args.output_path, f'repro_run{run}'),
            seed=args.seed, num_classes=1, dataset_type=args.dataset
        )
        results.append(result)

    print(f"\n{'='*70}")
    print("VERIFICATION RESULTS")
    print(f"{'='*70}")
    passed = True
    for i in range(1, num_runs):
        loss_diff = abs(results[0]['test_loss'] - results[i]['test_loss'])
        miou_diff = abs(results[0]['miou'] - results[i]['miou'])
        print(f"Run 1 vs {i+1}: Loss diff={loss_diff:.10f} | mIOU diff={miou_diff:.10f}")
        if loss_diff > 1e-6 or miou_diff > 1e-6:
            print("  ❌ FAIL"); passed = False
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
        from utils.trainer import train_segmentation
        result = train_segmentation(
            model_name=args.model, loss_name=args.loss, size=args.size,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            dataset=args.dataset, output_path=os.path.join(args.output_path, f'seed_{seed}'),
            seed=seed, num_classes=1, dataset_type=args.dataset
        )
        results.append({
            'seed': seed,
            'test_loss': result['test_loss'], 'miou': result['miou'],
            'dice': result['dice'], 'pixel_accuracy': result['pixel_accuracy'],
            'best_val_loss': result['best_val_loss'],
            'total_params': result['complexity']['total_params'],
            'model_size_mb': result['complexity']['model_size_mb'],
            'memory_mb': result['complexity']['memory_mb'],
            'gflops': result['complexity']['gflops'],
            'fps': result['inference_stats']['fps'],
            'latency_ms': result['inference_stats']['latency_ms'],
            'avg_inference_ms': result['inference_stats']['avg_time_s'] * 1000
        })

    losses    = [r['test_loss'] for r in results]
    mious     = [r['miou'] for r in results]
    dices     = [r['dice'] for r in results]
    pixel_acc = [r['pixel_accuracy'] for r in results]
    val_loss  = [r['best_val_loss'] for r in results]
    fps_list  = [r['fps'] for r in results]
    latencies = [r['latency_ms'] for r in results]

    print(f"\n{'='*70}\nSTATISTICS FOR PAPER\n{'='*70}")
    print(f"Test Loss:      {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f"mIOU:           {np.mean(mious):.4f} ± {np.std(mious):.4f}")
    print(f"Dice:           {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Pixel Accuracy: {np.mean(pixel_acc):.4f} ± {np.std(pixel_acc):.4f}")
    print(f"FPS:            {np.mean(fps_list):.2f} ± {np.std(fps_list):.2f}")
    print(f"Latency:        {np.mean(latencies):.4f} ± {np.std(latencies):.4f} ms")
    print(f"\nLaTeX:")
    print(f"mIOU: ${np.mean(mious):.4f} \\pm {np.std(mious):.4f}$")
    print(f"Dice: ${np.mean(dices):.4f} \\pm {np.std(dices):.4f}$")
    print(f"FPS:  ${np.mean(fps_list):.2f} \\pm {np.std(fps_list):.2f}$")
    print(f"{'='*70}\n")

    result_file = os.path.join(args.output_path, f'{args.model}_{args.dataset}_multiseed.json')
    with open(result_file, 'w') as f:
        json.dump({
            'config': {
                'model': args.model, 'dataset': args.dataset, 'loss': args.loss,
                'epochs': args.epochs, 'batch_size': args.batch_size,
                'lr': args.lr, 'size': args.size
            },
            'seeds': seeds, 'results': results,
            'statistics': {
                'test_loss':      {'mean': float(np.mean(losses)),    'std': float(np.std(losses))},
                'miou':           {'mean': float(np.mean(mious)),     'std': float(np.std(mious))},
                'dice':           {'mean': float(np.mean(dices)),     'std': float(np.std(dices))},
                'pixel_accuracy': {'mean': float(np.mean(pixel_acc)), 'std': float(np.std(pixel_acc))},
                'val_loss':       {'mean': float(np.mean(val_loss)),  'std': float(np.std(val_loss))},
                'fps':            {'mean': float(np.mean(fps_list)),  'std': float(np.std(fps_list))},
                'latency_ms':     {'mean': float(np.mean(latencies)), 'std': float(np.std(latencies))}
            },
            'complexity': {
                'total_params': results[0]['total_params'],
                'model_size_mb': results[0]['model_size_mb'],
                'memory_mb': results[0]['memory_mb'],
                'gflops': results[0]['gflops']
            }
        }, f, indent=2)
    print(f"Results saved: {result_file}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      type=str,   default='floodkaggle',
                        choices=['floodkaggle', 'floodscene'])
    parser.add_argument('--model',        type=str,   required=True)
    parser.add_argument('--size',         type=int,   default=None)
    parser.add_argument('--loss',         type=str,   default='bce')
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=4)
    parser.add_argument('--lr',           type=float, default=0.001)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--output_path',  type=str,   default='outputs')
    parser.add_argument('--download',     action='store_true')
    parser.add_argument('--verify_repro', action='store_true')
    parser.add_argument('--multiseed',    action='store_true')
    parser.add_argument('--seeds',        type=int, nargs='+', default=[42, 123, 456, 789, 2024])
    args = parser.parse_args()

    set_seed(args.seed)

    if args.download:
        download_dataset(args.dataset)

    if args.size is None:
        args.size = 256

    if args.verify_repro:
        verify_reproducibility(args, num_runs=2)
        return

    if args.multiseed:
        run_multiseed_experiments(args, seeds=args.seeds)
        return

    print("="*70)
    print(f"Dataset: {args.dataset} | Model: {args.model} | Size: {args.size}")
    print(f"Loss: {args.loss} | Epochs: {args.epochs} | BS: {args.batch_size} | LR: {args.lr}")
    print("="*70)

    from utils.trainer import train_segmentation
    train_segmentation(
        model_name=args.model, loss_name=args.loss, size=args.size,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        dataset=args.dataset, output_path=args.output_path,
        seed=args.seed, num_classes=1, dataset_type=args.dataset
    )


if __name__ == '__main__':
    main()