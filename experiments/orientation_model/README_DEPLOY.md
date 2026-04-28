Here’s a clean, copy-pasteable handoff you can drop in as `README_DEPLOY.md`.

---

# Orientation Model — Deploy Handoff

This doc packages everything you need to **ship** the page-orientation classifier in your OCR pipeline, with options for a **single best model** (ConvNeXt-Tiny) and a **conditional ensemble** (ConvNeXt-Tiny ⊕ EfficientNetV2-S) for borderline pages.

---

## 0) Environment & Repo Layout

**Python:** 3.11 (MPS supported on Apple Silicon)
**Key libs:** `torch`, `torchvision`, `Pillow`, `opencv-python`, `tqdm`

```
orientation_model/
├── data/                         # train/validation/test prepared via prepare_dataset.py
├── checkpoints/
│   ├── orient_convnext_tiny.pth  # ✅ best single-head model (ship this)
│   └── orient_effnetv2s.pth      # optional ensemble partner
├── src/
│   ├── train_convnext_tiny.py    # train ConvNeXt-Tiny (single head)
│   ├── train_effnetv2_s.py       # train EfficientNetV2-S (single head)
│   ├── predict_model.py          # arch-aware single-head loader + predictor
│   ├── predict_ensemble.py       # logits-average ensemble (ConvNeXt + EffNet)
│   ├── eval_test.py              # evaluates single model on data/test
│   └── eval_ensemble.py          # evaluates ensemble on data/test
└── prepare_dataset.py            # EXIF-safe JPEG dataset builder
```

---

## 1) What to Deploy (TL;DR)

* **Default** (fast & accurate):
  **ConvNeXt-Tiny** (`checkpoints/orient_convnext_tiny.pth`) via `src/predict_model.py`
  → **95.83%** test accuracy (0°:100%, others: 94.4%)

* **Conditional Ensemble** (optional, for low-margin pages):
  Run **ConvNeXt ⊕ EffNetV2-S** only when top-1 probability from ConvNeXt is **< 0.55**
  → **94.44%** overall when always on; as a fallback it recovers edge cases.

---

## 2) One-time Sanity (already done, commands preserved)

**Evaluate ConvNeXt-Tiny:**

```bash
python -m src.eval_test --ckpt ./checkpoints/orient_convnext_tiny.pth --test ./data/test
# -> Test accuracy: 69/72 = 95.83%
```

**Evaluate Ensemble:**

```bash
python -m src.eval_ensemble \
  --ckpt_a ./checkpoints/orient_convnext_tiny.pth \
  --ckpt_b ./checkpoints/orient_effnetv2s.pth \
  --test ./data/test
# -> Test accuracy: ~94.44%
```

---

## 3) Integrate in OCR Pipeline

### A) Single-model call (ConvNeXt-Tiny)

If you call from Python with **file paths**:

```python
from orientation_model.src.predict_model import load_model, predict_image

MODEL_CKPT = "orientation_model/checkpoints/orient_convnext_tiny.pth"

model, device, tfm, size, idx_map = load_model(MODEL_CKPT)  # idx_map is fixed {0:0,1:90,2:180,3:270}

def predict_degrees_path(img_path: str) -> tuple[int, dict[int, float]]:
    # bands=True does a simple center+border vote
    return predict_image(img_path, model, device, tfm, size, idx_map, bands=True)
```

If you have **NumPy images** in memory, either save to temp and reuse the function above, or adapt `predict_image` to accept PIL images directly (trivial: wrap the last few lines with `Image.fromarray`).

### B) Conditional ensemble for low-margin pages

```python
from orientation_model.src.predict_model import load_model as load_single, predict_image as predict_single
from orientation_model.src.predict_ensemble import predict_image_ensemble

CNX_CKPT = "orientation_model/checkpoints/orient_convnext_tiny.pth"
EFN_CKPT = "orientation_model/checkpoints/orient_effnetv2s.pth"

# load ConvNeXt once
cnx_model, cnx_dev, cnx_tfm, cnx_size, idx_map = load_single(CNX_CKPT)

def predict_with_fallback(img_path: str, margin: float = 0.55) -> tuple[int, dict[int, float], str]:
    deg, scores = predict_single(img_path, cnx_model, cnx_dev, cnx_tfm, cnx_size, idx_map, bands=True)
    if scores[max(scores, key=scores.get)] >= margin:
        return deg, scores, "convnext"
    # low-confidence → ensemble
    deg2, scores2 = predict_ensemble(img_path, CNX_CKPT, EFN_CKPT, use_bands=True)
    return deg2, scores2, "ensemble"
```

**Recommendation:** start with `margin=0.55`. Raise to `0.60` if you want more ensemble calls; lower if you want to minimize them.

---

## 4) Operational Notes

* **EXIF normalization**: we fixed dataset prep and two-head predictor to honor EXIF. If you feed **photo JPEGs** from scanners/phones, make sure any ad-hoc code uses `ImageOps.exif_transpose` before analysis, or just stick to the predictor as provided.
* **Label order at inference**: single-head predictors decode logits using a **fixed order** `[0, 90, 180, 270]` that matches training. We *do not* rely on `ImageFolder.class_to_idx` at inference (prevents cyclic errors).
* **Throughput**: ConvNeXt-Tiny at 384 with a center+bands vote is **fast** on M-series; ensemble doubles compute only when you need it.
* **Memory**: If you push to 448 input, reduce batch sizes in training; inference RAM impact is modest.

---

## 5) Monitoring & Logging

* Log **top-1 probability** per page. Keep a counter of pages where top-1 < threshold (e.g., 0.55).
  If this rate grows, investigate input drift (new document templates, heavier graphics, etc.).
* Store a small sample of **(image_path, predicted_deg, prob, model_used)** for audit.
* Periodically run `src/eval_test.py` / `src/eval_ensemble.py` against a stable hold-out set to track regressions.

---

## 6) Repro & Retrain (optional)

**Dataset (EXIF-safe, JPEG, native size):**

```bash
python prepare_dataset.py \
  --source_images ../all_DFD_Notes_Master_File \
  --master_key   ../master_key.txt \
  --output_dir   ./data \
  --split 0.7,0.15,0.15 \
  --ext .jpg
```

**Train ConvNeXt-Tiny (ship model):**

```bash
python -m src.train_convnext_tiny \
  --data ./data \
  --out ./checkpoints/orient_convnext_tiny.pth \
  --epochs 40 --img_size 384 --batch 16
```

**Train EfficientNetV2-S (ensemble partner):**

```bash
python -m src.train_effnetv2_s \
  --data ./data \
  --out ./checkpoints/orient_effnetv2s.pth \
  --epochs 40 --img_size 384 --batch 16
```

---

## 7) Known Good Numbers (current run)

* **ConvNeXt-Tiny (single)**: **95.83%** test (0°:100%, 90°:94.4%, 180°:94.4%, 270°:94.4%)
* **ConvNeXt ⊕ EffNetV2-S (always on)**: **94.44%** test (0°:100%, 90°:94.4%, 180°:94.4%, 270°:88.9%)

**Recommended deployment:** ConvNeXt-Tiny default + conditional ensemble for low margins.

---

## 8) Versioning & Backups

* Keep a `checkpoints/README.txt` listing: filename, date, arch, `img_size`, and validation/test scores.
* Consider exporting TorchScript/ONNX later if you want to ship a lighter runtime binary; accuracy will be the same with the current preprocessing.

---

## 9) Quick FAQ

**Q: Are bands necessary?**
A: They’re cheap and help polarity (0↔180, 90↔270) on mixed pages. Keep them on for the single-head predictor.

**Q: Should I just ensemble all the time?**
A: You can, but conditional usage is typically enough and keeps latency down.

**Q: What about two-head?**
A: Now that ConvNeXt single-head performs at 95.8%, two-head isn’t required. If you add new templates with heavier side artifacts, we can revisit two-head family/polarity logic later.

---

**That’s it.** You can deploy with ConvNeXt-Tiny today, flip on the conditional ensemble for lower margins, and you’re set.
