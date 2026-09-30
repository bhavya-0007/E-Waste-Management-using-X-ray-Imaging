# 🔬 E-Waste Detection & Classification System
## AI-Based Object Detection using YOLOv8 | 80/10/10 Split

---

## ⚡ QUICK START (Run in ~2 Hours)

```bash
# 1. Install dependencies
pip install ultralytics opencv-python matplotlib numpy pandas seaborn pyyaml requests tqdm pillow

# 2. Download dataset (run this first - takes ~5 minutes)
python download_dataset.py

# 3. Train the model (~60-90 min on GPU, ~4-6 hrs CPU)
python train.py

# 4. Evaluate on test set
python evaluate.py

# 5. Run inference on images
python predict.py --source data/images/test/
```

---

## 📁 Project Structure

```
ewaste_project/
├── README.md
├── download_dataset.py       ← Auto-downloads dataset
├── train.py                  ← Full training pipeline
├── evaluate.py               ← Metrics + confusion matrix
├── predict.py                ← Inference on images/video
├── material_classifier.py    ← Rule-based material + hazard inference
├── recovery_estimator.py     ← Economic value estimation
├── dataset.yaml              ← Auto-generated dataset config
└── data/                     ← Auto-created by download_dataset.py
    ├── images/train/
    ├── images/val/
    ├── images/test/
    ├── labels/train/
    ├── labels/val/
    └── labels/test/
```

---

## 🗂️ Dataset Info

**Primary Source**: Roboflow Universe (free, no login needed for some)
**Backup**: Script auto-generates synthetic data if download fails

**Classes (12)**:
| ID | Class |
|----|-------|
| 0  | circuit_board |
| 1  | mobile_phone |
| 2  | battery |
| 3  | cable |
| 4  | charger |
| 5  | keyboard |
| 6  | monitor |
| 7  | mouse |
| 8  | hard_drive |
| 9  | ram |
| 10 | cpu |
| 11 | capacitor |

---

## 🏗️ Architecture (Improved vs Original)

| Feature | Original (70/20/10) | This Project (80/10/10) |
|---------|-------------------|------------------------|
| Model | YOLOv8 only | YOLOv8 + material inference |
| Output | BBox + class | BBox + class + hazard flag + value |
| Hazard detection | ❌ | ✅ Toxic component flagging |
| Recovery value | ❌ | ✅ Economic value estimator |
| Dataset split | 70/20/10 | **80/10/10** |

---

## 📊 Expected Results

| Metric | Expected |
|--------|---------|
| Precision | ~85-90% |
| Recall | ~82-87% |
| mAP@0.5 | ~0.84-0.88 |
| F1 Score | ~0.85 |

---

## Requirements
- Python 3.8+, 8GB RAM, GPU optional
