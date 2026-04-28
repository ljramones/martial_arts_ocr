# Image Layout Model – Data Preparation, Training & Integration

## Folder Layout

```
image_layout_model/
├── dataset/
│   ├── data.yaml              # YOLO dataset config
│   ├── images/
│   │   ├── train/             # full-page training images
│   │   ├── val/               # validation pages
│   │   ├── test/              # test pages (mixed)
│   │   └── train_tiles/       # micro-crops and augmented variants
│   ├── labels/
│   │   ├── train/             # YOLO .txt labels for full pages
│   │   ├── val/
│   │   ├── test/
│   │   └── train_tiles/       # matching labels for crops (usually “0 0.5 0.5 1 1”)
│   └── lists/
│       ├── figures_list.txt
│       ├── train_pos.txt
│       ├── train_neg.txt
│       ├── train_tiles.txt
│       ├── train.txt
│       └── val.txt
├── split_dataset.py           # splits figures_list into train/val/test
├── make_micro_crops.py        # generates focused crops around missed figures
├── tools/
│   └── blur_variants.py       # adds blurred / contrast-boosted variants for robustness
├── yolov8n.pt                 # base YOLOv8n weights (downloaded automatically)
└── Overview.md                # this guide
```

---

## 1. Objective

Detect **figures** (line drawings, grayscale images, photos) in scanned historical or technical documents.

Single-class YOLOv8 detector:

```yaml
names: [figure]
```

---

## 2. Data Preparation Workflow

### 2.1 Source Setup

Place all original pages (e.g., 112 total) in a staging folder.
List the pages known to contain figures in:

```
dataset/lists/figures_list.txt
```

### 2.2 Split Dataset

From the root (`image_layout_model/`):

```bash
python split_dataset.py
```

This creates `dataset/images/{train,val,test}` and corresponding `labels/` folders.

### 2.3 Annotate with CVAT

Annotate all figure-containing pages with tight bounding boxes around each figure:

* **Label:** `figure`
* **Export:** YOLO 1.1 format

Put labels under:

```
dataset/labels/train/
dataset/labels/val/
dataset/labels/test/
```

---

## 3. Dataset Enhancement

### 3.1 Micro-Crops for Small or Missed Figures

After an initial YOLO pass, identify missed or low-confidence figures and create tight **micro-crops** with:

```bash
python make_micro_crops.py
```

This script writes cropped figures into:

```
dataset/images/train_tiles/
dataset/labels/train_tiles/
```

Each label file typically contains:

```
0 0.5 0.5 1 1
```

(one full bounding box covering the crop).

### 3.2 Blur / Contrast Variants

To make the model robust to poor scan quality:

```bash
python tools/blur_variants.py
```

This creates `_g.jpg` and `_c.jpg` copies (blurred / contrast-boosted) for every micro-crop.

---

## 4. Dataset Lists

Rebuild the training lists before retraining:

```bash
awk -v p="$(pwd)/dataset" '{
  gsub(/\.txt$/,"",$0); print p"/images/train/"$0".jpg"
}' <(ls dataset/labels/train/*.txt | xargs -n1 basename) > dataset/lists/train_pos.txt

ls -1 "$(pwd)"/dataset/images/train_tiles/*.jpg > dataset/lists/train_tiles.txt

python - <<'PY'
from pathlib import Path
import random
root=Path("dataset")
imgs={p.stem for p in (root/"images/train").glob("*.jpg")}
lbls={p.stem for p in (root/"labels/train").glob("*.txt")}
negs=sorted(imgs - lbls)
POS=len(lbls); NEG=min(len(negs), POS*2)
random.seed(42)
pick = negs if len(negs)<=NEG else random.sample(negs, NEG)
out=root/"lists"/"train_neg.txt"; out.parent.mkdir(parents=True, exist_ok=True)
with open(out,"w") as f:
    for s in pick:
        f.write(str((root/"images/train"/f"{s}.jpg").resolve())+"\n")
print(f"POS={POS}, NEG={len(pick)}")
PY

cat dataset/lists/train_pos.txt dataset/lists/train_tiles.txt dataset/lists/train_neg.txt > dataset/lists/train.txt
```

---

## 5. Training the YOLO Model

### 5.1 First Training Run

```bash
yolo detect train \
  model=yolov8n.pt \
  data=dataset/data.yaml \
  imgsz=1280 epochs=80 batch=8 \
  optimizer=adamw lr0=0.001 cos_lr=True \
  patience=20 mosaic=0 mixup=0 copy_paste=0 erasing=0 \
  degrees=1.5 translate=0.04 scale=0.15 fliplr=0 flipud=0 \
  workers=0
```

### 5.2 Fine-Tune (after micro-crops)

```bash
yolo detect train \
  model=runs/detect/train6/weights/best.pt \
  data=dataset/data.yaml \
  imgsz=1664 epochs=25 \
  optimizer=adamw lr0=0.00035 cos_lr=True \
  patience=10 mosaic=0 mixup=0 copy_paste=0 erasing=0 \
  degrees=1.0 translate=0.03 scale=0.12 fliplr=0 flipud=0 \
  workers=0
```

---

## 6. Validation and Testing

### Evaluate on Validation

```bash
yolo detect val \
  model=runs/detect/trainX/weights/best.pt \
  data=dataset/data.yaml
```

### Predict on Test Set

```bash
yolo detect predict \
  model=runs/detect/trainX/weights/best.pt \
  source=dataset/images/test \
  save=True conf=0.20 imgsz=1536
```

> Typical strong model performance:
> mAP50 ≈ 0.85, Recall ≈ 0.65–0.75 on validation pages with figures.

---

## 7. Integration into the Layout Analyzer

Create a new detector module:
`utils/image/layout/detectors/yolo_figure.py`

```python
from __future__ import annotations
from typing import List, Dict, Any
import cv2, numpy as np
from ultralytics import YOLO
from utils.image.regions.core_image import ImageRegion
from . import BaseDetector

class YOLOFigureDetector(BaseDetector):
    def __init__(self, cfg: Dict[str, Any]):
        self.model = YOLO(cfg["yolo_model_path"])
        self.conf  = float(cfg.get("yolo_conf", 0.22))
        self.iou   = float(cfg.get("yolo_iou", 0.60))
        self.imgsz = int(cfg.get("yolo_imgsz", 1536))
        self.tta   = bool(cfg.get("yolo_tta", False))

    def detect(self, image: np.ndarray) -> List[ImageRegion]:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        result = self.model.predict(source=bgr, imgsz=self.imgsz,
                                    conf=self.conf, iou=self.iou,
                                    augment=self.tta, verbose=False)[0]
        out = []
        if result.boxes is None: return out
        for (x1, y1, x2, y2), conf in zip(result.boxes.xyxy.cpu().numpy(),
                                          result.boxes.conf.cpu().numpy()):
            out.append(ImageRegion(
                x=int(x1), y=int(y1),
                width=int(x2-x1), height=int(y2-y1),
                region_type="figure", confidence=float(conf)
            ))
        return out
```

### Add to `LayoutAnalyzer`

In `utils/image/layout/analyzer.py`:

```python
from .detectors.yolo_figure import YOLOFigureDetector

if self.cfg.get("use_yolo_figure", False):
    self.figure = YOLOFigureDetector(self.cfg)
```

### Example Config Override

```python
{
  "use_yolo_figure": True,
  "yolo_model_path": "/Users/larrymitchell/ML/martial_arts_ocr/runs/detect/train6/weights/best.pt",
  "yolo_conf": 0.22,
  "yolo_iou": 0.60,
  "yolo_imgsz": 1536,
  "yolo_tta": False
}
```

---

## 8. Testing in the CLI Harness

```bash
python analyze_page.py \
  --input ./dataset/images/val \
  --out ./data/notebook_outputs/debug_output \
  --detectors figure \
  --no_mask_images
```

---

## 9. Key Points

* **Figures detected with high confidence** (0.8–0.95 on clear scans).
* **Recall improved by micro-crops** on missed or blurry examples.
* Use **conf=0.18, iou=0.55, imgsz=1792, augment=True** for high-recall QA runs.
* Keep negatives balanced to preserve precision.
* Integrate YOLO as the **primary image detector** replacing heuristic figure detection.

---

## 10) Quick Fine-Tuning Recipe (Resume + Tiles)

Use this when you add a few new labels or micro-crops and want a fast recall bump.

### A. Rebuild lists

```bash
# positives from full pages
awk -v p="$(pwd)/dataset" '{
  gsub(/\.txt$/,"",$0); print p"/images/train/"$0".jpg"
}' <(ls dataset/labels/train/*.txt | xargs -n1 basename) > dataset/lists/train_pos.txt

# ALL tiles (micro-crops + blur/contrast variants)
ls -1 "$(pwd)"/dataset/images/train_tiles/*.jpg > dataset/lists/train_tiles.txt

# balanced negatives (2:1)
python - <<'PY'
from pathlib import Path
import random
root=Path("dataset")
imgs={p.stem for p in (root/"images/train").glob("*.jpg")}
lbls={p.stem for p in (root/"labels/train").glob("*.txt")}
negs=sorted(imgs - lbls)
POS=len(lbls); NEG=min(len(negs), POS*2)
random.seed(42)
pick = negs if len(negs)<=NEG else random.sample(negs, NEG)
(root/"lists").mkdir(parents=True, exist_ok=True)
with open(root/"lists"/"train_neg.txt","w") as f:
    for s in pick:
        f.write(str((root/"images/train"/f"{s}.jpg").resolve())+"\n")
print(f"POS={POS}, NEG={len(pick)}")
PY

# merged training list
cat dataset/lists/train_pos.txt dataset/lists/train_tiles.txt dataset/lists/train_neg.txt > dataset/lists/train.txt

# keep val as-is for stable metrics
ls -1 "$(pwd)"/dataset/images/val/*.jpg > dataset/lists/val.txt
```

Ensure `dataset/data.yaml` uses the lists:

```yaml
path: dataset
train: lists/train.txt
val: lists/val.txt
test: images/test
names: [figure]
```

### B. Resume from your latest best weights

```bash
LATEST=$(ls -td /Users/larrymitchell/ML/martial_arts_ocr/runs/detect/train* | head -1)

yolo detect train \
  model="$LATEST/weights/best.pt" \
  data=dataset/data.yaml \
  imgsz=1664 epochs=25 \
  optimizer=adamw lr0=0.00035 cos_lr=True \
  patience=10 mosaic=0 mixup=0 copy_paste=0 erasing=0 \
  degrees=1.0 translate=0.03 scale=0.12 fliplr=0 flipud=0 \
  workers=0
```

### C. Validate & Visual QA

```bash
# metrics
yolo detect val \
  model=runs/detect/train*/weights/best.pt \
  data=dataset/data.yaml

# overlays on val (high-recall pass)
yolo detect predict \
  model=runs/detect/train*/weights/best.pt \
  source=dataset/images/val \
  save=True conf=0.18 iou=0.55 imgsz=1792 augment=True
```

**Expected:**

* mAP@50 ≥ 0.80 on `val`,
* Recall in the 0.60–0.75 range (or better) for small/blurry panels,
* No boxes on text-only pages at `conf≈0.22`.

### D. Integrate into pipeline

Set the YOLO detector config:

```python
{
  "use_yolo_figure": True,
  "yolo_model_path": "/Users/larrymitchell/ML/martial_arts_ocr/runs/detect/train6/weights/best.pt",
  "yolo_conf": 0.22,
  "yolo_iou": 0.60,
  "yolo_imgsz": 1536,
  "yolo_tta": False      # set True for QA or tough pages
}
```
