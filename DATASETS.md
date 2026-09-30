# 📥 DATASET SOURCES - E-Waste Detection
# All free, no payment required

## ─── OPTION 1: Roboflow Universe (RECOMMENDED, Easiest) ───────────────────

### Step-by-step:
1. Go to: https://universe.roboflow.com/
2. Search: "e-waste detection" or "electronic components"
3. Click on a dataset → Export → YOLOv8 format → Download ZIP
4. Extract ZIP contents into your `data/` folder

### Best datasets on Roboflow (search these):
- "E-waste Detection" by recycling-fmovz
- "PCB Components" (for circuit boards)
- "Electronic Waste Classification"
- "E-waste YOLOv8" (multiple versions available)

### With free API key (auto-download):
```python
# After getting free API key from roboflow.com
python download_dataset.py --source roboflow --api-key YOUR_FREE_KEY
```


## ─── OPTION 2: Kaggle (Free, ~5 min setup) ────────────────────────────────

### Datasets to download:
1. https://www.kaggle.com/datasets/akshaychavan123/electronic-waste-e-waste
2. https://www.kaggle.com/datasets/ravisane1/e-waste-dataset-v2
3. https://www.kaggle.com/datasets/inertiablocks/e-waste-dataset

### Setup Kaggle CLI:
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/ (download from kaggle.com → Account → API)
kaggle datasets download akshaychavan123/electronic-waste-e-waste
unzip electronic-waste-e-waste.zip -d data/raw/
```

### After downloading, convert to YOLO format:
Images should be in: data/images/train/, data/images/val/, data/images/test/
Labels should be in: data/labels/train/, data/labels/val/, data/labels/test/
Label format: class_id cx cy w h (all normalized 0-1)


## ─── OPTION 3: Google Open Images (Large, High Quality) ──────────────────

Classes available in Open Images:
- Mobile phone
- Computer keyboard
- Computer mouse
- Hard drive
- Battery charger
- Power cables

Download tool:
```bash
pip install openimages
oi_download_dataset --base_dir data/openimages \
  --labels "Mobile phone" "Computer keyboard" "Battery charger" \
  --format darknet  # YOLO format
```


## ─── OPTION 4: Synthetic (No Internet, Pipeline Testing) ─────────────────

```bash
python download_dataset.py --source synthetic --synthetic-size 2000
```
Creates 1600 train + 200 val + 200 test synthetic images.
Sufficient to test the entire pipeline works.
Not suitable for production model accuracy.


## ─── LABEL FORMAT (YOLO) ─────────────────────────────────────────────────

Each image needs a corresponding .txt file with same name:
- image:  data/images/train/photo001.jpg
- labels: data/labels/train/photo001.txt

Each line in the .txt = one object:
```
class_id center_x center_y width height
```
All values normalized 0-1:
```
0 0.512 0.437 0.324 0.412    # circuit_board at center
2 0.782 0.251 0.156 0.203    # battery top-right
```

Class IDs:
| ID | Class        |
|----|-------------|
| 0  | circuit_board|
| 1  | mobile_phone |
| 2  | battery      |
| 3  | cable        |
| 4  | charger      |
| 5  | keyboard     |
| 6  | monitor      |
| 7  | mouse        |
| 8  | hard_drive   |
| 9  | ram          |
| 10 | cpu          |
| 11 | capacitor    |


## ─── DATASET SPLIT: 80/10/10 ─────────────────────────────────────────────

As per paper requirements:
- Training:   80% of images → data/images/train/
- Validation: 10% of images → data/images/val/
- Testing:    10% of images → data/images/test/

The download_dataset.py script handles this split automatically.


## ─── MINIMUM DATASET SIZE ────────────────────────────────────────────────

For acceptable accuracy:
- Minimum: 500 total images (400 train, 50 val, 50 test)
- Recommended: 2000+ total images
- Ideal: 5000+ images with balanced classes

For quick testing: use --synthetic-size 500
