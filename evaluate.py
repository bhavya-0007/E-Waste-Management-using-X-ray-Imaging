"""
evaluate.py
===========
Comprehensive model evaluation on test set.
Produces:
- Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95
- Per-class breakdown
- Confusion matrix
- PR curves

Usage:
    python evaluate.py
    python evaluate.py --weights runs/detect/ewaste_yolov8/weights/best.pt
    python evaluate.py --weights best.pt --split test
"""

import os
import sys
import argparse
import json
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CLASSES = [
    "circuit_board", "mobile_phone", "battery", "cable",
    "charger", "keyboard", "monitor", "mouse",
    "hard_drive", "ram", "cpu", "capacitor"
]

DEFAULT_WEIGHTS = "runs/detect/ewaste_yolov8/weights/best.pt"

# ─── FUNCTIONS ───────────────────────────────────────────────────────────────

def find_best_weights():
    """Find the best available weights file."""
    candidates = [
        "runs/detect/ewaste_yolov8/weights/best.pt",
        "runs/detect/ewaste_yolov8/weights/last.pt",
    ]
    # Also search runs/ directory
    runs = Path("runs/detect")
    if runs.exists():
        for run in runs.iterdir():
            if run.is_dir():
                best = run / "weights" / "best.pt"
                if best.exists():
                    candidates.insert(0, str(best))

    for c in candidates:
        if Path(c).exists():
            return c

    return None


def evaluate_model(weights_path, split="test", conf=0.25, iou=0.5):
    """Run model evaluation."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("  E-WASTE MODEL EVALUATION")
    print(f"  Weights: {weights_path}")
    print(f"  Split:   {split}")
    print("=" * 60)

    model = YOLO(weights_path)

    # Update YAML to use test split
    import yaml
    yaml_path = Path("dataset.yaml")
    if not yaml_path.exists():
        print("❌ dataset.yaml not found. Run download_dataset.py first.")
        sys.exit(1)

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Temporarily set val to test split for evaluation
    original_val = cfg.get("val")
    cfg["val"] = f"images/{split}"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\n🔍 Running evaluation on {split} set...\n")

    results = model.val(
        data=str(yaml_path.resolve()),
        split=split,
        conf=conf,
        iou=iou,
        plots=True,
        save_json=True,
        verbose=True,
    )

    # Restore original val
    cfg["val"] = original_val or "images/val"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return results


def print_metrics_report(results):
    """Print a formatted metrics report."""
    print("\n" + "=" * 60)
    print("  📊 EVALUATION RESULTS")
    print("=" * 60)

    try:
        # Overall metrics
        metrics = results.results_dict
        box = results.box

        print("\n🎯 Overall Detection Metrics:")
        print(f"   {'Metric':<25} {'Value':>10}")
        print("   " + "-" * 37)

        metric_names = {
            "metrics/precision(B)": "Precision",
            "metrics/recall(B)": "Recall",
            "metrics/mAP50(B)": "mAP@0.5",
            "metrics/mAP50-95(B)": "mAP@0.5:0.95",
        }

        for key, name in metric_names.items():
            val = metrics.get(key, None)
            if val is not None:
                bar = "█" * int(val * 20)
                print(f"   {name:<25} {val:>8.4f}  {bar}")

        # F1 Score (calculated from precision and recall)
        p = metrics.get("metrics/precision(B)", 0)
        r = metrics.get("metrics/recall(B)", 0)
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
            bar = "█" * int(f1 * 20)
            print(f"   {'F1 Score':<25} {f1:>8.4f}  {bar}")

        # Per-class metrics
        print("\n📋 Per-Class Metrics:")
        print(f"   {'Class':<20} {'AP@0.5':>8} {'Images':>8}")
        print("   " + "-" * 38)

        try:
            for i, cls_name in enumerate(CLASSES):
                if i < len(box.ap50):
                    ap = box.ap50[i]
                    print(f"   {cls_name:<20} {ap:>8.4f}")
        except Exception:
            print("   (Per-class metrics not available)")

        print("\n" + "=" * 60)

        # Summary table for report
        print("\n📄 Summary Table (copy to report):")
        print("   ┌─────────────────────┬───────────┐")
        print("   │ Metric              │   Value   │")
        print("   ├─────────────────────┼───────────┤")
        for key, name in metric_names.items():
            val = metrics.get(key, 0)
            print(f"   │ {name:<19}  │  {val:>7.4f}  │")
        if p + r > 0:
            print(f"   │ {'F1 Score':<19}  │  {f1:>7.4f}  │")
        print("   └─────────────────────┴───────────┘")

        # Save results to JSON
        save_results(metrics, p, r, f1 if p + r > 0 else 0)

    except Exception as e:
        print(f"⚠️  Could not parse detailed metrics: {e}")
        print("   Check results.png in your runs/ directory.")


def save_results(metrics, precision, recall, f1):
    """Save evaluation results to JSON."""
    output = {
        "model": "YOLOv8",
        "dataset_split": "80/10/10",
        "classes": CLASSES,
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mAP_50": round(metrics.get("metrics/mAP50(B)", 0), 4),
            "mAP_50_95": round(metrics.get("metrics/mAP50-95(B)", 0), 4),
        },
        "raw_metrics": {k: round(float(v), 4) for k, v in metrics.items()
                        if isinstance(v, (int, float))}
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to: evaluation_results.json")


def demo_evaluation():
    """
    Demo evaluation when no trained model exists.
    Shows expected output format.
    """
    print("\n" + "=" * 60)
    print("  📊 DEMO EVALUATION OUTPUT (No model trained yet)")
    print("  Run train.py first for real results")
    print("=" * 60)

    demo_metrics = {
        "Precision":   0.8821,
        "Recall":      0.8497,
        "F1 Score":    0.8656,
        "mAP@0.5":     0.8634,
        "mAP@0.5:0.95":0.6921,
    }

    demo_per_class = {
        "circuit_board": 0.921,
        "mobile_phone":  0.903,
        "battery":       0.876,
        "cable":         0.812,
        "charger":       0.845,
        "keyboard":      0.891,
        "monitor":       0.934,
        "mouse":         0.867,
        "hard_drive":    0.889,
        "ram":           0.856,
        "cpu":           0.912,
        "capacitor":     0.798,
    }

    print("\n🎯 Overall Metrics (DEMO):")
    print(f"   {'Metric':<25} {'Value':>10}  {'Bar':}")
    print("   " + "-" * 55)
    for name, val in demo_metrics.items():
        bar = "█" * int(val * 20)
        print(f"   {name:<25} {val:>8.4f}  {bar}")

    print("\n📋 Per-Class AP@0.5 (DEMO):")
    print(f"   {'Class':<20} {'AP@0.5':>8}")
    print("   " + "-" * 30)
    for cls, ap in demo_per_class.items():
        print(f"   {cls:<20} {ap:>8.4f}")

    print("\n⚠️  These are DEMO values. Train the model for real metrics.")
    print("   python train.py")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate E-Waste Detection Model")
    parser.add_argument("--weights", default=None,
                        help="Path to model weights (.pt file)")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for NMS")
    parser.add_argument("--demo", action="store_true",
                        help="Show demo output without model")
    args = parser.parse_args()

    if args.demo:
        demo_evaluation()
        return

    # Find weights
    weights = args.weights or find_best_weights()

    if not weights or not Path(weights).exists():
        print(f"⚠️  No trained model found.")
        print("   Run: python train.py  (to train first)")
        print("   Or:  python evaluate.py --demo  (for demo output)\n")
        demo_evaluation()
        return

    print(f"✅ Using weights: {weights}")

    # Run evaluation
    results = evaluate_model(weights, args.split, args.conf, args.iou)

    # Print report
    print_metrics_report(results)

    print(f"\n📌 NEXT STEP: Run predictions:")
    print(f"   python predict.py --source data/images/test/")


if __name__ == "__main__":
    main()
