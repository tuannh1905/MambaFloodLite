# 🌊 Flood Segmentation Training Framework

Complete training pipeline for flood detection with **strict reproducibility** using PyTorch.

---

## 📋 Table of Contents
- [Datasets](#-datasets)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training Commands](#-training-commands)
- [Multi-Seed Experiments](#-multi-seed-experiments)
- [Project Structure](#-project-structure)

---

## 🗂️ Datasets

### 1. **FloodVN** (Binary Segmentation)
- **Classes**: 1 (flooded vs non-flooded)
- **Loss**: `bce` (Binary Cross Entropy)
- **Auto-download**: ✅ Yes
- **Structure**:
  ```
  floodvn/
  ├── train/
  │   ├── images/
  │   └── labels/  (JSON format)
  ├── val/
  └── test/
  ```

### 2. **FloodKaggle** (Binary Segmentation)
- **Classes**: 1 (flooded vs non-flooded)
- **Loss**: `bce`
- **Auto-download**: ✅ Yes
- **Structure**: Same as FloodVN

### 3. **FloodNet** (Multi-class Segmentation)
- **Classes**: 10 (background, building-flooded, building-non-flooded, road-flooded, etc.)
- **Loss**: `ce` (Cross Entropy)
- **Auto-download**: ❌ Manual setup required
- **Location**: `/content/drive/MyDrive/FloodNet/`
- **Structure**:
  ```
  FloodNet/
  ├── train/
  │   ├── image/
  │   └── mask/
  ├── val/
  │   ├── image/
  │   └── mask/
  └── test/
      ├── image/
      └── mask/
  ```

---

## 🛠️ Installation

```bash
# Clone repository
git clone <your-repo>
cd flood-segmentation

# Install dependencies
pip install torch torchvision albumentations opencv-python tqdm gdown numpy pillow

# For Google Colab
!pip install -q gdown albumentations
```

---

## 🚀 Quick Start

### **FloodVN** (Easiest - Auto Download)

```bash
# Download dataset
python main.py --dataset floodvn --download

# Train
python main.py \
  --dataset floodvn \
  --model unet \
  --loss bce \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.001 \
  --seed 42
```

### **FloodKaggle**

```bash
# Download dataset
python main.py --dataset floodkaggle --download

# Train
python main.py \
  --dataset floodkaggle \
  --model unet \
  --loss bce \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.001 \
  --seed 42
```

### **FloodNet** (Multi-class)

```bash
# NO auto-download - Manual setup required
# 1. Upload FloodNet to: /content/drive/MyDrive/FloodNet/
# 2. Ensure structure: train/image, train/mask, val/, test/

# Train
python main.py \
  --dataset floodnet \
  --model unet \
  --loss ce \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.001 \
  --seed 42
```

---

## 📝 Training Commands

### Basic Training

```bash
# FloodVN (binary)
python main.py --dataset floodvn --model unet --epochs 50

# FloodNet (multi-class) 
python main.py --dataset floodnet --model unet --loss ce --epochs 50
```

### Advanced Options

```bash
python main.py \
  --dataset floodvn \           # Dataset: floodvn, floodkaggle, floodnet
  --model unet \                # Model architecture
  --size 512 \                  # Input size (default: 256)
  --loss bce \                  # Loss: bce (binary) or ce (multi-class)
  --epochs 100 \                # Number of epochs
  --batch_size 8 \              # Batch size
  --lr 0.0001 \                 # Learning rate
  --seed 42 \                   # Random seed
  --output_path outputs/exp1    # Output directory
```

---

## 🔬 Multi-Seed Experiments (For Paper)

Run experiments with **multiple seeds** to get mean ± std statistics:

```bash
# Default seeds: 42, 123, 456, 789, 2024
python main.py \
  --dataset floodvn \
  --model unet \
  --multiseed \
  --epochs 50
```

**Custom seeds:**
```bash
python main.py \
  --dataset floodnet \
  --model unet \
  --loss ce \
  --multiseed \
  --seeds 42 100 200 300 400 \
  --epochs 50
```

**Output:**
```
STATISTICS FOR PAPER
════════════════════════════════════════
Test Loss: 0.1234 ± 0.0056
mIOU:      0.8765 ± 0.0123
Val Loss:  0.1456 ± 0.0078
════════════════════════════════════════

LaTeX format:
Test Loss: $0.1234 \pm 0.0056$
mIOU: $0.8765 \pm 0.0123$

📊 Results saved to: outputs/unet_floodvn_multiseed.json
```

---

## 📊 Dataset-Specific Best Practices

### FloodVN & FloodKaggle

| Parameter | Recommended Value | Note |
|-----------|------------------|------|
| `--loss` | `bce` | Binary segmentation |
| `--size` | `256` or `512` | Higher = slower but better |
| `--batch_size` | `4` or `8` | Depends on GPU |
| `--lr` | `0.001` | Standard for Adam |
| `--epochs` | `50-100` | Monitor val loss |

**Example:**
```bash
python main.py --dataset floodvn --model unet --size 512 --epochs 100 --batch_size 8
```

### FloodNet (Multi-class)

| Parameter | Recommended Value | Note |
|-----------|------------------|------|
| `--loss` | `ce` | **MUST use CrossEntropy** |
| `--size` | `256` or `512` | 10 classes = more memory |
| `--batch_size` | `4` | Lower due to 10 classes |
| `--lr` | `0.0001` | Lower for stability |
| `--epochs` | `50-100` | Multi-class needs more |

**Example:**
```bash
python main.py --dataset floodnet --model unet --loss ce --size 512 --epochs 100 --batch_size 4 --lr 0.0001
```

---

## 📁 Project Structure

```
flood-segmentation/
├── main.py                    # Main entry point
├── models/
│   ├── __init__.py           # get_model()
│   ├── unet.py               # U-Net architecture
│   └── ...                   # Other models
├── losses/
│   ├── __init__.py           # get_loss()
│   └── ...                   # Loss functions
├── utils/
│   ├── dataloader.py         # Dataset & DataLoader
│   ├── trainer.py            # Training loop
│   └── metrics.py            # mIOU, Dice, Accuracy
├── outputs/                  # Saved models & results
│   ├── unet_bce_floodvn_s42.pth
│   └── unet_floodvn_multiseed.json
├── floodvn/                  # Auto-downloaded
├── floodkaggle/              # Auto-downloaded
└── README.md                 # This file
```

---

## 🎯 Complete Training Pipeline

### 1. Train on FloodVN
```bash
# Download dataset
python main.py --dataset floodvn --download

# Single seed training
python main.py --dataset floodvn --model unet --epochs 50 --seed 42

# Multi-seed for paper
python main.py --dataset floodvn --model unet --multiseed --epochs 50
```

### 2. Train on FloodKaggle
```bash
python main.py --dataset floodkaggle --download
python main.py --dataset floodkaggle --model unet --multiseed --epochs 50
```

### 3. Train on FloodNet
```bash
# Manual setup: Upload to /content/drive/MyDrive/FloodNet/

# Single seed
python main.py --dataset floodnet --model unet --loss ce --epochs 50 --seed 42

# Multi-seed
python main.py --dataset floodnet --model unet --loss ce --multiseed --epochs 50
```

---

## 🔧 Troubleshooting

### FloodNet not found
```
FileNotFoundError: FloodNet not found at /content/drive/MyDrive/FloodNet
```
**Solution:** Manually upload FloodNet folder to Google Drive at exactly this path.

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size: `--batch_size 2`

### Wrong loss function
```
ValueError: num_classes=10 but using bce loss
```
**Solution:** Use `--loss ce` for FloodNet (multi-class)

---

## 📈 Results Format

**Single training output:**
```
════════════════════════════════════════
FINAL RESULTS
════════════════════════════════════════
Test Loss:     0.1234567890
mIOU:          0.8765432100
Best Val Loss: 0.1456789000
Saved:         outputs/unet_bce_floodvn_s42.pth
════════════════════════════════════════
```

**Multi-seed JSON output:**
```json
{
  "config": {
    "model": "unet",
    "dataset": "floodvn",
    "loss": "bce",
    "epochs": 50
  },
  "seeds": [42, 123, 456, 789, 2024],
  "results": [
    {"seed": 42, "test_loss": 0.123, "miou": 0.876},
    ...
  ],
  "statistics": {
    "test_loss": {"mean": 0.1234, "std": 0.0056},
    "miou": {"mean": 0.8765, "std": 0.0123}
  }
}
```

---

## ✅ Reproducibility Features

This framework ensures **strict reproducibility**:

- ✅ Fixed seeds for all RNGs (Python, NumPy, PyTorch, CUDA)
- ✅ Deterministic CUDA operations (`torch.use_deterministic_algorithms(True)`)
- ✅ Disabled cuDNN benchmark and TF32
- ✅ Per-sample deterministic augmentation
- ✅ Fixed DataLoader worker seeds
- ✅ Saved RNG states in checkpoints

**Note:** `num_workers=4` for speed. Use `num_workers=0` for 100% reproducibility if needed.

---

## 📚 Citation

If you use this framework, please cite:

```bibtex
@misc{flood-segmentation,
  title={Flood Segmentation Training Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourrepo}
}
```

---

## 📧 Contact

For issues or questions, open an issue on GitHub or contact: your.email@example.com