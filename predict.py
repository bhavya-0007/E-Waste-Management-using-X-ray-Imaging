"""
predict.py
==========
Run inference on images, video, or webcam feed.
Also integrates material classifier and hazard detection.

Usage:
    python predict.py --source data/images/test/           # Folder
    python predict.py --source image.jpg                   # Single image
    python predict.py --source 0                           # Webcam
    python predict.py --source video.mp4                   # Video
    python predict.py --source data/images/test/ --save    # Save results
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

# ─── MATERIAL + HAZARD CLASSIFIER ────────────────────────────────────────────

# Material composition profiles (from recycling literature)
MATERIAL_PROFILES = {
    "circuit_board": {
        "materials": {"copper": 15.0, "gold": 0.03, "silver": 0.15,
                      "palladium": 0.01, "lead": 2.0, "tin": 4.0},
        "hazardous": ["lead", "brominated_flame_retardants"],
        "recovery_priority": "HIGH",
        "color": (0, 255, 0),
    },
    "mobile_phone": {
        "materials": {"gold": 0.034, "silver": 0.35, "copper": 15.0,
                      "cobalt": 5.0, "lithium": 2.0},
        "hazardous": ["lithium", "cobalt"],
        "recovery_priority": "HIGH",
        "color": (0, 200, 255),
    },
    "battery": {
        "materials": {"lithium": 8.0, "cobalt": 15.0, "nickel": 10.0,
                      "manganese": 5.0, "lead": 20.0},
        "hazardous": ["lead", "lithium", "cadmium", "mercury"],
        "recovery_priority": "CRITICAL_HAZARD",
        "color": (0, 0, 255),
    },
    "cable": {
        "materials": {"copper": 60.0, "pvc_plastic": 30.0, "tin": 2.0},
        "hazardous": ["pvc_plasticizers"],
        "recovery_priority": "MEDIUM",
        "color": (255, 165, 0),
    },
    "charger": {
        "materials": {"copper": 25.0, "iron": 10.0, "plastic": 40.0},
        "hazardous": ["lead_solder"],
        "recovery_priority": "MEDIUM",
        "color": (255, 100, 0),
    },
    "keyboard": {
        "materials": {"plastic": 70.0, "copper": 8.0, "steel": 5.0},
        "hazardous": [],
        "recovery_priority": "LOW",
        "color": (180, 180, 180),
    },
    "monitor": {
        "materials": {"glass": 50.0, "plastic": 25.0, "copper": 5.0,
                      "rare_earth": 0.5},
        "hazardous": ["mercury_backlight", "lead_glass"],
        "recovery_priority": "HIGH",
        "color": (100, 100, 255),
    },
    "mouse": {
        "materials": {"plastic": 60.0, "copper": 5.0, "steel": 8.0},
        "hazardous": [],
        "recovery_priority": "LOW",
        "color": (180, 180, 0),
    },
    "hard_drive": {
        "materials": {"steel": 35.0, "aluminium": 20.0, "copper": 5.0,
                      "rare_earth_magnets": 2.0},
        "hazardous": [],
        "recovery_priority": "MEDIUM",
        "color": (100, 200, 100),
    },
    "ram": {
        "materials": {"gold": 0.05, "silver": 0.2, "copper": 20.0,
                      "silicon": 15.0},
        "hazardous": ["lead_solder"],
        "recovery_priority": "HIGH",
        "color": (0, 255, 200),
    },
    "cpu": {
        "materials": {"gold": 0.2, "silver": 0.1, "copper": 15.0,
                      "silicon": 20.0, "palladium": 0.01},
        "hazardous": ["lead_solder"],
        "recovery_priority": "HIGH",
        "color": (200, 0, 200),
    },
    "capacitor": {
        "materials": {"aluminium": 40.0, "paper": 20.0, "electrolyte": 15.0},
        "hazardous": ["electrolyte_chemicals"],
        "recovery_priority": "MEDIUM",
        "color": (255, 200, 0),
    },
}

# Economic value estimates (USD per kg of material, approximate 2024)
MATERIAL_VALUES_USD_PER_KG = {
    "gold": 60000,
    "silver": 800,
    "palladium": 40000,
    "platinum": 30000,
    "copper": 8,
    "aluminium": 2,
    "steel": 0.5,
    "cobalt": 35,
    "lithium": 25,
    "nickel": 15,
    "rare_earth": 100,
    "rare_earth_magnets": 50,
}

# Typical component weights in grams
COMPONENT_WEIGHTS_G = {
    "circuit_board": 200,
    "mobile_phone": 180,
    "battery": 50,
    "cable": 100,
    "charger": 150,
    "keyboard": 800,
    "monitor": 3000,
    "mouse": 120,
    "hard_drive": 500,
    "ram": 30,
    "cpu": 50,
    "capacitor": 10,
}


def classify_material(class_name):
    """Return material profile for a detected class."""
    return MATERIAL_PROFILES.get(class_name, {
        "materials": {},
        "hazardous": [],
        "recovery_priority": "UNKNOWN",
        "color": (128, 128, 128),
    })


def estimate_recovery_value(class_name):
    """Estimate economic recovery value in USD."""
    profile = classify_material(class_name)
    weight_g = COMPONENT_WEIGHTS_G.get(class_name, 100)
    weight_kg = weight_g / 1000.0

    total_value = 0.0
    breakdown = {}

    for material, pct in profile["materials"].items():
        material_weight_kg = weight_kg * (pct / 100.0)
        value_per_kg = MATERIAL_VALUES_USD_PER_KG.get(material, 0)
        value = material_weight_kg * value_per_kg
        if value > 0.001:
            breakdown[material] = round(value, 4)
        total_value += value

    return {
        "class": class_name,
        "total_value_usd": round(total_value, 4),
        "weight_g": weight_g,
        "breakdown": breakdown,
        "recovery_priority": profile["recovery_priority"],
        "hazardous_materials": profile["hazardous"],
        "is_hazardous": len(profile["hazardous"]) > 0,
    }


def print_recovery_report(detections):
    """Print a recovery/hazard report for all detections."""
    if not detections:
        return

    print("\n" + "=" * 65)
    print("  ♻️  RECOVERY & HAZARD ANALYSIS REPORT")
    print("=" * 65)

    total_value = 0.0
    hazardous_items = []
    high_value_items = []

    for det in detections:
        cls = det["class"]
        conf = det["confidence"]
        recovery = estimate_recovery_value(cls)
        total_value += recovery["total_value_usd"]

        if recovery["is_hazardous"]:
            hazardous_items.append((cls, recovery["hazardous_materials"]))
        if recovery["recovery_priority"] == "HIGH":
            high_value_items.append((cls, recovery["total_value_usd"]))

        print(f"\n  📦 {cls.upper()} (conf: {conf:.2f})")
        print(f"     Priority   : {recovery['recovery_priority']}")
        print(f"     Est. Value : ${recovery['total_value_usd']:.4f}")
        if recovery["hazardous_materials"]:
            print(f"     ⚠️  Hazardous: {', '.join(recovery['hazardous_materials'])}")
        if recovery["breakdown"]:
            top = sorted(recovery["breakdown"].items(), key=lambda x: -x[1])[:3]
            print(f"     Top metals : {', '.join(f'{m}=${v:.4f}' for m,v in top)}")

    print(f"\n  💰 TOTAL ESTIMATED RECOVERY VALUE: ${total_value:.3f}")
    print(f"  🔴 HAZARDOUS ITEMS: {len(hazardous_items)}")
    print(f"  🟢 HIGH-VALUE ITEMS: {len(high_value_items)}")

    if hazardous_items:
        print(f"\n  ⚠️  HAZARD ISOLATION REQUIRED:")
        for cls, hazards in hazardous_items:
            print(f"     → {cls}: {', '.join(hazards)}")

    if high_value_items:
        print(f"\n  🏆 PRIORITY DISMANTLING:")
        for cls, val in sorted(high_value_items, key=lambda x: -x[1]):
            print(f"     → {cls}: ${val:.4f}")

    print("=" * 65)


def find_best_weights():
    """Find trained weights."""
    candidates = [
        DEFAULT_WEIGHTS,
        "runs/detect/ewaste_yolov8/weights/best.pt",
    ]
    runs = Path("runs/detect")
    if runs.exists():
        for run in sorted(runs.iterdir(), key=os.path.getmtime, reverse=True):
            if run.is_dir():
                best = run / "weights" / "best.pt"
                if best.exists():
                    candidates.insert(0, str(best))
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def run_inference(weights, source, conf=0.25, iou=0.45, save=True, show=False):
    """Run YOLOv8 inference."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Run: pip install ultralytics")
        sys.exit(1)

    print(f"\n🔍 Running inference...")
    print(f"   Weights: {weights}")
    print(f"   Source:  {source}")
    print(f"   Conf:    {conf}")

    model = YOLO(weights)

    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        save=save,
        show=show,
        save_txt=True,
        save_conf=True,
        verbose=True,
    )

    # Collect all detections for report
    all_detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"
                all_detections.append({
                    "class": cls_name,
                    "class_id": cls_id,
                    "confidence": conf_score,
                    "bbox": box.xyxy[0].tolist(),
                })

    # Print recovery report
    if all_detections:
        print_recovery_report(all_detections)

        # Save detection results to JSON
        output_json = {
            "total_detections": len(all_detections),
            "detections": all_detections,
        }
        with open("predictions.json", "w") as f:
            json.dump(output_json, f, indent=2)
        print(f"\n💾 Detection results saved: predictions.json")
    else:
        print("\n⚠️  No objects detected. Try lowering --conf threshold.")

    return results, all_detections


def demo_predict():
    """Show demo prediction output without model."""
    print("\n" + "=" * 60)
    print("  🔍 DEMO PREDICTION OUTPUT")
    print("  (Run train.py first for real inference)")
    print("=" * 60)

    demo_detections = [
        {"class": "circuit_board", "confidence": 0.921, "bbox": [10, 20, 300, 280]},
        {"class": "battery",       "confidence": 0.887, "bbox": [320, 10, 500, 200]},
        {"class": "cpu",           "confidence": 0.843, "bbox": [50, 300, 200, 450]},
        {"class": "ram",           "confidence": 0.798, "bbox": [220, 310, 420, 440]},
    ]

    print("\nDetected Objects:")
    for d in demo_detections:
        print(f"  ✅ {d['class']:<20} confidence: {d['confidence']:.3f}")

    print_recovery_report(demo_detections)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run E-Waste Detection Inference")
    parser.add_argument("--weights", default=None, help="Path to .pt weights")
    parser.add_argument("--source", default="data/images/test/",
                        help="Image/folder/video path, or 0 for webcam")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save annotated images")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Show results window")
    parser.add_argument("--demo", action="store_true",
                        help="Demo output without model")
    args = parser.parse_args()

    if args.demo:
        demo_predict()
        return

    weights = args.weights or find_best_weights()

    if not weights or not Path(weights).exists():
        print(f"⚠️  No trained model found. Showing demo output.\n")
        print("   Run: python train.py  to train first.\n")
        demo_predict()
        return

    source = args.source
    if not Path(source).exists() and source != "0":
        print(f"⚠️  Source not found: {source}")
        print("   Run download_dataset.py first to get test images.")
        demo_predict()
        return

    run_inference(weights, source, args.conf, args.iou, args.save, args.show)


if __name__ == "__main__":
    main()
