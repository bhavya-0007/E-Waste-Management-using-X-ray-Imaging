"""
quicktest.py
============
Runs a quick end-to-end test of the entire pipeline.
Tests without needing a trained model.

Usage:
    python quicktest.py
"""

import sys
import os

print("=" * 60)
print("  E-WASTE DETECTION - PIPELINE QUICKTEST")
print("=" * 60)

# Test 1: Import check
print("\n[1/5] Checking dependencies...")
deps = {
    "yaml":        "pyyaml",
    "numpy":       "numpy",
    "PIL":         "Pillow",
    "json":        "built-in",
    "pathlib":     "built-in",
}

all_ok = True
for module, package in deps.items():
    try:
        __import__(module)
        print(f"   ✅ {module}")
    except ImportError:
        print(f"   ❌ {module} (install: pip install {package})")
        all_ok = False

try:
    from ultralytics import YOLO
    import ultralytics
    print(f"   ✅ ultralytics {ultralytics.__version__}")
except ImportError:
    print(f"   ⚠️  ultralytics not installed (run: pip install ultralytics)")

try:
    import torch
    print(f"   ✅ torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
except ImportError:
    print(f"   ⚠️  torch not installed (will be installed with ultralytics)")

# Test 2: Dataset YAML
print("\n[2/5] Checking dataset.yaml...")
try:
    import yaml
    with open("dataset.yaml") as f:
        cfg = yaml.safe_load(f)
    print(f"   ✅ dataset.yaml found")
    print(f"   Classes: {cfg['nc']} | Names: {cfg['names'][:3]}...")
except FileNotFoundError:
    print("   ❌ dataset.yaml not found. Run: python download_dataset.py")

# Test 3: Dataset structure
print("\n[3/5] Checking data directories...")
from pathlib import Path
splits = ["train", "val", "test"]
for split in splits:
    img_dir = Path(f"data/images/{split}")
    lbl_dir = Path(f"data/labels/{split}")
    if img_dir.exists():
        n = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        print(f"   ✅ {split}: {n} images")
    else:
        print(f"   ⚠️  data/images/{split} missing → run download_dataset.py")

# Test 4: Material classifier
print("\n[4/5] Testing material classifier...")
try:
    sys.path.insert(0, ".")
    from material_classifier import estimate_value, classify_batch

    test_result = estimate_value("circuit_board")
    assert test_result["total_value_usd"] > 0
    print(f"   ✅ circuit_board value: ${test_result['total_value_usd']:.4f}")

    test_result2 = estimate_value("battery")
    print(f"   ✅ battery hazardous: {test_result2['is_hazardous']}")
    print(f"   ✅ battery priority: {test_result2['recovery_priority']}")

    batch = classify_batch([
        {"class": "cpu", "confidence": 0.9},
        {"class": "battery", "confidence": 0.85},
    ])
    print(f"   ✅ batch classify: {batch['summary']['total_value_usd']:.4f} USD total")
    print(f"   ✅ hazardous items: {batch['summary']['hazardous_count']}")
except Exception as e:
    print(f"   ❌ material classifier error: {e}")

# Test 5: Trained model check
print("\n[5/5] Checking for trained model...")
candidates = [
    "runs/detect/ewaste_yolov8/weights/best.pt",
]
runs_path = Path("runs/detect")
if runs_path.exists():
    for run in runs_path.iterdir():
        best = run / "weights" / "best.pt"
        if best.exists():
            candidates.insert(0, str(best))

model_found = False
for c in candidates:
    if Path(c).exists():
        print(f"   ✅ Trained model found: {c}")
        model_found = True
        break

if not model_found:
    print("   ⚠️  No trained model yet → run: python train.py")

# Summary
print("\n" + "=" * 60)
print("  PIPELINE STATUS")
print("=" * 60)
print(f"  Dependencies : {'✅ OK' if all_ok else '⚠️ Some missing'}")

data_ok = Path("data/images/train").exists() and \
          len(list(Path("data/images/train").glob("*.jpg")) +
              list(Path("data/images/train").glob("*.png"))) > 0
print(f"  Dataset      : {'✅ Ready' if data_ok else '⚠️ Run download_dataset.py'}")
print(f"  Model        : {'✅ Trained' if model_found else '⚠️ Run train.py'}")

print("\n📌 Run Order:")
print("   1. pip install -r requirements.txt")
print("   2. python download_dataset.py")
print("   3. python train.py")
print("   4. python evaluate.py")
print("   5. python predict.py --source data/images/test/")
print("   6. python material_classifier.py --all-classes")
print("=" * 60)
