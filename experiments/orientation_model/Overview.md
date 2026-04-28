# Orientation Model Overview

## Purpose

The **Orientation Model** predicts one of four rotations — **0°, 90°, 180°, 270°** — for scanned or photographed pages so they can be auto-uprighted before OCR.

We started with heuristics (projection peaks, contour ratios, footer polarity) and moved to **CNNs trained on EXIF-normalized JPEGs**. The result is a robust, fast, and accurate pre-OCR step.

---

## Directory Structure

```
orientation_model/
├── data/
│   ├── train/{0,90,180,270}/
│   ├── validation/{0,90,180,270}/
│   └── test/{0,90,180,270}/
├── checkpoints/
│   ├── orient_mnv3s.pth             # MobileNetV3-Small (single-head)
│   ├── orient_twohead_mnv3s.pth     # MobileNetV3-Small (two-head)
│   ├── orient_convnext_tiny.pth     # ✅ ConvNeXt-Tiny (best single)
│   └── orient_effnetv2s.pth         # EfficientNetV2-S (ensemble partner)
├── src/
│   ├── train_model.py               # MobileNetV3-Small (single-head)
│   ├── train_twohead.py             # MobileNetV3-Small (two-head family+polarity)
│   ├── train_convnext_tiny.py       # ConvNeXt-Tiny (single-head)
│   ├── train_effnetv2_s.py          # EfficientNetV2-S (single-head)
│   ├── predict_model.py             # Arch-aware single-head predictor
│   ├── predict_twohead.py           # Two-head predictor (two-stage voting)
│   ├── predict_ensemble.py          # ConvNeXt + EffNetV2-S ensemble predictor
│   ├── eval_test.py                 # Single-model evaluator
│   └── eval_ensemble.py             # Ensemble evaluator
├── prepare_dataset.py               # EXIF-safe JPEG dataset builder
└── overview.md
```

---

## Dataset Preparation (`prepare_dataset.py`)

* **EXIF-aware**: loads with Pillow + `ImageOps.exif_transpose()` so pixels match what you saw when labeling.
* **Canonicalization**: rotate by **negative** of master angle to make an **upright** page; then synthesize the 4 rotations.
* **JPEG by default** (quality 95, 4:4:4), native resolution and color preserved.
* Deterministic split into **train/validation/test**; optional `--audit N` writes small mosaics for spot-checks.

**Example**

```bash
python prepare_dataset.py \
  --source_images ../all_DFD_Notes_Master_File \
  --master_key   ../master_key.txt \
  --output_dir   ./data \
  --split 0.7,0.15,0.15 \
  --ext .jpg \
  --audit 3
```

---

## Models Tried (and how they fared)

| Model / Recipe                      | Notes                                                                     |                                       Validation |                                      Test (overall) | Typical Errors                     |
| ----------------------------------- | ------------------------------------------------------------------------- | -----------------------------------------------: | --------------------------------------------------: | ---------------------------------- |
| **MobileNetV3-Small (single-head)** | 4-class classifier, label-safe augs, fixed label order                    |                                             ~90% |                                          **76–85%** | 90↔270 sometimes                   |
| **MobileNetV3-Small (two-head)**    | Family (portrait/landscape) + Polarity (up/down) with 576→1024 projection | peaked **~90.6%** then unstable until eval fixed |      varied; eval logic sensitive to margin/weights | family drift to “landscape” on 0°  |
| **ConvNeXt-Tiny (single-head)**     | 384 input, augs + AdamW; arch-aware loader                                |                                         **high** |             **95.83%** (0°:100%, 90/180/270: 94.4%) | very few 1-off flips               |
| **EfficientNetV2-S (single-head)**  | 384 input; initially polarity-heavy errors                                |                          **100%** (val, one run) | **68.06%** out-of-box; improved as ensemble partner | 0°/270°→180° without center-anchor |
| **Ensemble (ConvNeXt + EffNet)**    | Average logits (center + simple bands)                                    |                                              n/a |     **94.44%** (0°:100%, 90/180: 94.4%, 270: 88.9%) | a few 270° still borderline        |

### Key lessons

* **EXIF normalization** was essential—without it, labels and pixels disagree by ±90°.
* Heuristics helped us reason about **family vs polarity**, and those insights still matter at inference (center-anchored family decision; bands refine polarity).
* **ConvNeXt-Tiny** is the best single model on this corpus; **EffNetV2-S** is a strong ensemble complement.

---

## Training Pipelines

### MobileNetV3-Small (single-head) — `src/train_model.py`

* Label-safe augs (shear, padding, light blur, jitter).
* Head-only warm-up → unfreeze and fine-tune with AdamW + cosine LR.
* Saves `arch="mobilenet_v3_small"` and `img_size` into the checkpoint.

### MobileNetV3-Small (two-head) — `src/train_twohead.py`

* Family and polarity heads train jointly (projection 576→1024 + Hardswish + Dropout).
* Combined **log-prob** 4-way output in fixed order `[0, 90, 180, 270]`.
* Two-stage inference recommended (center decides family; bands refine polarity with masking).

### ConvNeXt-Tiny — `src/train_convnext_tiny.py`  ✅ *Recommended*

* Same augs; FP32 fine-tune is stable on M-series.
* Best single-model generalization on this dataset.

### EfficientNetV2-S — `src/train_effnetv2_s.py`

* Strong backbone; benefitted from center-anchored inference or ensemble use.

---

## Inference & Evaluation

### Single-head (`src/predict_model.py`)

* **Arch-aware** loader (MobileNetV3 / ConvNeXt / EffNetV2-S).
* Uses **fixed label order** `[0, 90, 180, 270]` for decoding (matches training).
* Simple **center + bands** voting.

```bash
python -m src.eval_test \
  --ckpt ./checkpoints/orient_convnext_tiny.pth \
  --test ./data/test
# Test accuracy: 95.83%
```

### Two-head (`src/predict_twohead.py`)

* **Two-stage voting**: center-only decides **family**; family-aware bands refine **polarity**; final masking prevents flips.
* Multi-scale (384 + 448) and flips supported.

### Ensemble (`src/predict_ensemble.py`, `src/eval_ensemble.py`)

* Logit average of ConvNeXt + EffNet across center and bands.
* Delivers nearly the same score as ConvNeXt alone, and helps on a few edge cases.

```bash
python -m src.eval_ensemble \
  --ckpt_a ./checkpoints/orient_convnext_tiny.pth \
  --ckpt_b ./checkpoints/orient_effnetv2s.pth \
  --test ./data/test
# Test accuracy: 94.44%
```

---

## Deployment Recommendation

* **Default**: **ConvNeXt-Tiny** (`checkpoints/orient_convnext_tiny.pth`) via `predict_model.py`.
* **Conditional ensemble (optional)**: if ConvNeXt top-1 probability **< 0.55**, re-score with the ensemble and use that result. This recovers borderline pages with a small compute cost.

Pseudo:

```python
deg, scores = predict_single(img_path, convnext_ckpt)
top = max(scores, key=scores.get)
if scores[top] < 0.55:
    deg, scores = predict_ensemble(img_path, convnext_ckpt, effnet_ckpt)
```

---

## Tuning & Extensions

| Goal              | Change                                                                       |
| ----------------- | ---------------------------------------------------------------------------- |
| Reduce 90↔270     | Increase side-band width/weight; keep **center anchor high**.                |
| Reduce 0↔180      | Emphasize top/bottom bands; optional light polarity prior by footer density. |
| Higher fidelity   | Train at **448** (drop batch a bit); often +1–2%.                            |
| Faster runtime    | Export TorchScript/ONNX later; current Python is plenty fast on M-series.    |
| Confidence gating | Add temperature scaling on validation; gate ensemble on low-margin pages.    |

---

## Summary

* **Best single model:** ConvNeXt-Tiny at **95.83%** test accuracy.
* **Ensemble option:** ConvNeXt ⊕ EffNetV2-S at **94.44%** (good fallback on low-margin pages).
* **Data:** EXIF-normalized JPEGs, native resolution.
* **Inference:** Center-anchored voting with label order fixed to `[0, 90, 180, 270]`.
* **Status:** Ready to ship as a pre-OCR orientation step.

*(Last updated: 2025-10-09 — multi-model results + deployment guidance)*
