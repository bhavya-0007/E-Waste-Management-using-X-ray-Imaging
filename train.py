"""
train.py
========
Full training pipeline for E-Waste Detection using YOLOv8.

Features:
- Configurable model size (nano/small/medium/large)
- 80/10/10 dataset split
- Early stopping, LR scheduling, augmentation
- Automatic checkpointing
- Training plots saved automatically

Usage:
    python train.py                        # Default (yolov8s, 50 epochs)
    python train.py --model yolov8n.pt    # Nano (fastest)
    python train.py --model yolov8m.pt    # Medium (more accurate)
    python train.py --epochs 100          # More epochs
    python train.py --batch 8             # Smaller batch (less RAM)
    python train.py --device cpu          # Force CPU training
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CLASSES = [
    "circuit_board", "mobile_phone", "battery", "cable",
    "charger", "keyboard", "monitor", "mouse",
    "hard_drive", "ram", "cpu", "capacitor"
]

DEFAULT_CONFIG = {
    "model": "yolov8s.pt",       # yolov8n (nano), yolov8s (small), yolov8m (medium)
    "epochs": 50,                 # 50 epochs is usually enough for convergence
    "batch": 16,                  # Reduce to 8 if OOM
    "imgsz": 640,                 # Input image size
    "patience": 15,               # Early stopping patience
    "lr0": 0.01,                  # Initial learning rate
    "lrf": 0.01,                  # Final LR factor
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "device": "",                 # "" = auto (GPU if available, else CPU)
    "workers": 4,
    "project": "runs/detect",
    "name": "ewaste_yolov8",
    "save_period": 10,            # Save checkpoint every N epochs
    "val": True,
    "plots": True,
    "verbose": True,
    # Augmentation
    "hsv_h": 0.015,               # Hue augmentation
    "hsv_s": 0.7,                 # Saturation augmentation
    "hsv_v": 0.4,                 # Value (brightness) augmentation
    "degrees": 10.0,              # Rotation
    "translate": 0.1,             # Translation
    "scale": 0.5,                 # Scale
    "shear": 2.0,                 # Shear
    "perspective": 0.001,
    "flipud": 0.0,                # Vertical flip (rare in real-world)
    "fliplr": 0.5,                # Horizontal flip
    "mosaic": 1.0,                # Mosaic augmentation
    "mixup": 0.1,                 # Mixup augmentation
    "copy_paste": 0.0,
}

# ─── FUNCTIONS ───────────────────────────────────────────────────────────────

def check_dependencies():
    """Check and install required packages."""
    try:
        import ultralytics
        print(f"✅ Ultralytics {ultralytics.__version__} found.")
    except ImportError:
        print("Installing ultralytics...")
        os.system("pip install ultralytics --quiet")
        import ultralytics
        print(f"✅ Ultralytics {ultralytics.__version__} installed.")

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, training on CPU (slower).")
        return cuda_available
    except ImportError:
        print("⚠️  PyTorch not installed.")
        return False


def validate_dataset():
    """Validate that the dataset exists and has the right structure."""
    required = [
        Path("data/images/train"),
        Path("data/images/val"),
        Path("data/images/test"),
        Path("data/labels/train"),
        Path("data/labels/val"),
        Path("data/labels/test"),
        Path("dataset.yaml"),
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("❌ Missing required files/directories:")
        for m in missing:
            print(f"   - {m}")
        print("\n👉 Run: python download_dataset.py")
        sys.exit(1)

    # Count images
    splits = {}
    for split in ["train", "val", "test"]:
        img_dir = Path(f"data/images/{split}")
        count = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        splits[split] = count

    print(f"\n📊 Dataset validated:")
    for split, count in splits.items():
        print(f"   {split:8s}: {count} images")

    if splits["train"] == 0:
        print("❌ No training images found!")
        print("👉 Run: python download_dataset.py")
        sys.exit(1)

    # Validate YAML
    with open("dataset.yaml") as f:
        cfg = yaml.safe_load(f)
    print(f"   Classes : {cfg.get('nc', '?')} ({', '.join(cfg.get('names', [])[:3])}...)")
    print("✅ Dataset structure valid.\n")

    return splits


def update_yaml_path():
    """Ensure dataset.yaml has absolute paths."""
    yaml_path = Path("dataset.yaml")
    if not yaml_path.exists():
        print("❌ dataset.yaml not found! Run download_dataset.py first.")
        sys.exit(1)

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = str(Path("data").resolve())
    cfg["train"] = "images/train"
    cfg["val"] = "images/val"
    cfg["test"] = "images/test"
    cfg["nc"] = len(CLASSES)
    cfg["names"] = CLASSES

    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("✅ dataset.yaml paths updated.")
    return str(yaml_path.resolve())


def train(config):
    """Run YOLOv8 training."""
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("  E-WASTE DETECTION MODEL TRAINING")
    print("  Architecture: YOLOv8 | Split: 80/10/10")
    print("=" * 60)

    yaml_path = update_yaml_path()

    print(f"\n📋 Training Configuration:")
    print(f"   Model:   {config['model']}")
    print(f"   Epochs:  {config['epochs']}")
    print(f"   Batch:   {config['batch']}")
    print(f"   Img size:{config['imgsz']}x{config['imgsz']}")
    print(f"   Classes: {len(CLASSES)}")
    print(f"   Device:  {'auto (CUDA if available)' if not config['device'] else config['device']}")
    print()

    # Load model (downloads pretrained weights automatically)
    print(f"📥 Loading model: {config['model']}")
    model = YOLO(config["model"])

    # Start training
    print(f"\n🚀 Starting training...\n")
    start_time = time.time()

    results = model.train(
        data=yaml_path,
        epochs=config["epochs"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        patience=config["patience"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        warmup_epochs=config["warmup_epochs"],
        device=config["device"],
        workers=config["workers"],
        project=config["project"],
        name=config["name"],
        save_period=config["save_period"],
        val=config["val"],
        plots=config["plots"],
        verbose=config["verbose"],
        # Augmentation params
        hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"],
        hsv_v=config["hsv_v"],
        degrees=config["degrees"],
        translate=config["translate"],
        scale=config["scale"],
        shear=config["shear"],
        perspective=config["perspective"],
        flipud=config["flipud"],
        fliplr=config["fliplr"],
        mosaic=config["mosaic"],
        mixup=config["mixup"],
        copy_paste=config["copy_paste"],
    )

    elapsed = time.time() - start_time
    print(f"\n✅ Training complete! Time: {elapsed/60:.1f} minutes")

    # Save path info
    weights_dir = Path(config["project"]) / config["name"] / "weights"
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"

    print(f"\n📁 Output Files:")
    print(f"   Best weights : {best_weights}")
    print(f"   Last weights : {last_weights}")
    print(f"   Results plot : {Path(config['project']) / config['name'] / 'results.png'}")

    if best_weights.exists():
        print(f"\n✅ Model saved successfully!")
    else:
        print(f"\n⚠️  Weights not found at expected path. Check {config['project']}/{config['name']}/")

    # Print final metrics
    print("\n📊 Final Training Metrics:")
    try:
        metrics = results.results_dict
        for k, v in metrics.items():
            if any(m in k for m in ["precision", "recall", "map", "fitness"]):
                print(f"   {k:30s}: {v:.4f}")
    except Exception:
        print("   (Check results.png for plots)")

    print(f"\n📌 NEXT STEP: Evaluate model:")
    print(f"   python evaluate.py")

    return results


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train E-Waste YOLOv8 Model")
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"],
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                                 "yolov8l.pt", "yolov8x.pt"],
                        help="Model variant (n=nano fastest, x=xlarge most accurate)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch", type=int, default=DEFAULT_CONFIG["batch"])
    parser.add_argument("--imgsz", type=int, default=DEFAULT_CONFIG["imgsz"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"],
                        help="cuda:0 or cpu")
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--name", default=DEFAULT_CONFIG["name"])
    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    # Validate dataset
    validate_dataset()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["model"] = args.model
    config["epochs"] = args.epochs
    config["batch"] = args.batch
    config["imgsz"] = args.imgsz
    config["device"] = args.device
    config["patience"] = args.patience
    config["name"] = args.name

    # Train
    train(config)


if __name__ == "__main__":
    main()
