# Orientation Subsystem Inventory

## Status

The repository already contains a neural-network-based page-orientation subsystem for classifying scanned page rotation into:

```text
0, 90, 180, 270 degrees
```

This work should be preserved. The best documented deployment path is ConvNeXt-Tiny with optional EfficientNetV2-S fallback/ensemble. No workbench integration was implemented in this inventory pass.

## Existing Files Found

| Path | Purpose |
| --- | --- |
| `experiments/orientation_model/Overview.md` | Main summary of the orientation model history, architecture choices, dataset layout, test results, and deployment recommendation. |
| `experiments/orientation_model/README_DEPLOY.md` | Deployment handoff for the orientation classifier, including commands and integration examples. |
| `experiments/orientation_model/prepare_dataset.py` | EXIF-safe dataset builder. Reads a master key of image rotations, canonicalizes pages upright, and synthesizes `0/90/180/270` training examples. |
| `experiments/orientation_model/src/train_convnext_tiny.py` | ConvNeXt-Tiny 4-class training pipeline. This is the recommended single model. |
| `experiments/orientation_model/src/train_effnetv2_s.py` | EfficientNetV2-S 4-class training pipeline, intended as an optional ensemble partner. |
| `experiments/orientation_model/src/train_model.py` | MobileNetV3-Small single-head training pipeline. Older baseline. |
| `experiments/orientation_model/src/train_twohead.py` | MobileNetV3-Small two-head training pipeline for family/polarity. Older experimental path. |
| `experiments/orientation_model/src/model_twohead.py` | Two-head MobileNet model definition: portrait/landscape family plus up/down polarity. |
| `experiments/orientation_model/src/predict_model.py` | Arch-aware single-head inference loader/predictor for MobileNetV3, ConvNeXt-Tiny, and EfficientNetV2-S checkpoints. |
| `experiments/orientation_model/src/predict_twohead.py` | EXIF-safe two-head inference with center/band voting. |
| `experiments/orientation_model/src/predict_ensemble.py` | ConvNeXt + EffNet logits-average ensemble predictor. |
| `experiments/orientation_model/src/eval_test.py` | Single-model evaluator for `data/test`. |
| `experiments/orientation_model/src/eval_ensemble.py` | Ensemble evaluator for `data/test`. |
| `utils/image/preprocessing/orientation_cnn.py` | Runtime-facing wrapper around the experiment model loader. Exposes model initialization and NumPy-image prediction. |
| `utils/image/preprocessing/orientation.py` | Heuristic orientation scorer using contours, projections, footer polarity, and optional OCR tie-breaks. This is separate from the NN subsystem. |
| `utils/image/preprocessing/ocr_osd.py` | Tesseract OSD helpers and 0-vs-180 tie-break utilities used by heuristic orientation logic. |
| `data/training/orientation/.gitkeep` | Placeholder for future organized orientation training data location. |
| `docs/dataset-inventory.md` | Existing note classifying `experiments/orientation_model/data/` as training/derived and `experiments/orientation_model/checkpoints/` as model output. |

Local generated/training artifacts were also found:

| Path | Status |
| --- | --- |
| `experiments/orientation_model/data/train/{0,90,180,270}/` | Local generated training images. Not tracked. |
| `experiments/orientation_model/data/validation/{0,90,180,270}/` | Local generated validation images. Not tracked. |
| `experiments/orientation_model/data/test/{0,90,180,270}/` | Local generated test images. Not tracked. |
| `experiments/orientation_model/checkpoints/orient_convnext_tiny.pth` | Local ignored checkpoint, about 111 MB. Recommended single model. |
| `experiments/orientation_model/checkpoints/orient_effnetv2s.pth` | Local ignored checkpoint, about 82 MB. Optional ensemble partner. |
| `experiments/orientation_model/checkpoints/orient_twohead_mnv3s.pth` | Local ignored checkpoint, older two-head model. |
| `experiments/orientation_model/checkpoints/orient_twohead_mnv3s.last.pth` | Local ignored checkpoint, last two-head checkpoint. |

## Training Pipeline

Training is centered under `experiments/orientation_model/`.

Dataset preparation:

```bash
python prepare_dataset.py \
  --source_images ../../data/corpora/donn_draeger/dfd_notes_master/original \
  --master_key ../master_key.txt \
  --output_dir ./data \
  --split 0.7,0.15,0.15 \
  --ext .jpg
```

The dataset builder:

- reads a filename-to-angle master key;
- uses Pillow `ImageOps.exif_transpose()` so pixels match visible orientation;
- rotates each labeled source page to canonical upright by negating the master-key angle;
- synthesizes class folders `0`, `90`, `180`, and `270`;
- writes train/validation/test splits;
- uses JPEG output by default.

Primary training command from the deploy handoff:

```bash
python -m src.train_convnext_tiny \
  --data ./data \
  --out ./checkpoints/orient_convnext_tiny.pth \
  --epochs 40 --img_size 384 --batch 16
```

Optional ensemble partner:

```bash
python -m src.train_effnetv2_s \
  --data ./data \
  --out ./checkpoints/orient_effnetv2s.pth \
  --epochs 40 --img_size 384 --batch 16
```

Other training pipelines exist for MobileNetV3 single-head and two-head variants. They should remain available for comparison, but the documented recommendation is ConvNeXt-Tiny as the default.

## Inference Pipeline

Single-head inference:

```python
from experiments.orientation_model.src.predict_model import load_model, predict_image

model, device, tfm, size, idx_map = load_model("experiments/orientation_model/checkpoints/orient_convnext_tiny.pth")
deg, scores = predict_image(image_path_or_pil, model, device, tfm, size, idx_map, bands=True)
```

Output:

- `deg`: one of `0`, `90`, `180`, `270`;
- `scores`: probability map keyed by degree, for example `{0: 0.98, 90: 0.01, 180: 0.01, 270: 0.00}`.

The class order is fixed:

```text
idx 0 -> 0
idx 1 -> 90
idx 2 -> 180
idx 3 -> 270
```

The deploy handoff describes the predicted degree as the page rotation class. It should be treated as the rotation/orientation detected for the input image. A future workbench wrapper must make the correction convention explicit before applying any rotation:

```text
detected rotation class vs. rotation needed to upright
```

Runtime-facing wrapper:

```python
from utils.image.preprocessing.orientation_cnn import init_orientation_model, predict_degrees

init_orientation_model(
    "experiments/orientation_model/checkpoints/orient_convnext_tiny.pth",
    "experiments/orientation_model/checkpoints/orient_effnetv2s.pth",
)
deg, scores_by_deg, top1_prob, model_used = predict_degrees(np_img)
```

`predict_degrees` accepts a NumPy image and returns:

- predicted degree;
- score/probability map by degree;
- top-1 probability;
- model used: `"convnext"` or `"ensemble"`.

The wrapper initializes ConvNeXt once and conditionally invokes the ensemble when top-1 probability is below the margin and the EffNet checkpoint/path is available.

Note: `predict_ensemble.py` imports `from src.predict_model import ...`, so direct package import may depend on running from the orientation model directory or setting `PYTHONPATH`. The current `orientation_cnn.py` catches import failure and treats ensemble as optional. A future integration wrapper should normalize this import path rather than rewriting model logic.

## Model Artifacts

Model artifacts exist locally under:

```text
experiments/orientation_model/checkpoints/
```

Observed local files:

```text
orient_convnext_tiny.pth
orient_effnetv2s.pth
orient_twohead_mnv3s.pth
orient_twohead_mnv3s.last.pth
```

`.gitignore` ignores:

```text
experiments/orientation_model/checkpoints/
```

`git ls-files` showed no tracked checkpoint files. Do not commit these model artifacts unless the user explicitly chooses a model artifact policy.

## Data / Dataset Layout

Current local generated dataset layout:

```text
experiments/orientation_model/data/
  train/
    0/
    90/
    180/
    270/
  validation/
    0/
    90/
    180/
    270/
  test/
    0/
    90/
    180/
    270/
```

This is generated/derived training data. `docs/dataset-inventory.md` already records that a future organized home is:

```text
data/training/orientation/
```

Do not move the current dataset during workbench integration. Existing scripts assume the experiment-relative layout.

## Input / Output Contract

Existing single-head inference accepts:

- image path string; or
- `PIL.Image.Image`.

`utils/image/preprocessing/orientation_cnn.py` accepts:

- NumPy image array, grayscale or RGB-like.

Important input notes:

- `prepare_dataset.py` and `predict_twohead.py` are explicitly EXIF-safe.
- `predict_model.py` opens file paths with `Image.open(...).convert("RGB")`; EXIF handling is not explicit there.
- `orientation_cnn._np_to_pil()` assumes raw pixels for NumPy arrays and does not perform EXIF handling.

Existing output:

```python
(deg, scores_by_deg)
```

or from `orientation_cnn.predict_degrees`:

```python
(deg, scores_by_deg, top1_prob, model_used)
```

Future workbench-facing contract should wrap this as:

```python
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class OrientationResult:
    rotation_degrees: int
    confidence: float | None
    source: str
    scores_by_degree: dict[int, float]
    metadata: dict[str, Any]
```

The wrapper must document whether `rotation_degrees` means detected page orientation or corrective rotation to apply.

## Dependencies

The NN subsystem uses:

- `torch`;
- `torchvision`;
- `Pillow`;
- `opencv-python` / `cv2`;
- `numpy`;
- `tqdm` for dataset preparation/training.

The heuristic orientation subsystem uses:

- `cv2`;
- `numpy`;
- optional Tesseract through `utils/image/preprocessing/ocr_osd.py`.

No new dependency should be added for the inventory or initial workbench wrapper.

## Current Usability

Current usability appears good but not yet workbench-integrated.

Evidence from existing docs:

- ConvNeXt-Tiny single model: `95.83%` test accuracy.
- ConvNeXt + EffNet ensemble: `94.44%` when always on.
- Recommended deployment: ConvNeXt-Tiny default, optional ensemble for low top-1 probability.

Existing wrapper:

- `utils/image/preprocessing/orientation_cnn.py` already exposes `init_orientation_model(...)` and `predict_degrees(...)`.
- This is the likely seam to preserve for future app/workbench integration.

Current caveats:

- checkpoint files are local ignored artifacts;
- import path for ensemble may need a wrapper fix;
- correction direction convention must be made explicit before rotating any workbench image;
- no current workbench UI/API integration has been done.

## Tests / Validation Evidence

Existing validation evidence is documented in:

- `experiments/orientation_model/Overview.md`;
- `experiments/orientation_model/README_DEPLOY.md`.

Known documented numbers:

```text
ConvNeXt-Tiny:
  69/72 test accuracy = 95.83%
  0°: 100%
  90°: 94.4%
  180°: 94.4%
  270°: 94.4%

ConvNeXt + EffNet ensemble:
  94.44% overall when always enabled
```

Evaluation commands:

```bash
python -m src.eval_test --ckpt ./checkpoints/orient_convnext_tiny.pth --test ./data/test

python -m src.eval_ensemble \
  --ckpt_a ./checkpoints/orient_convnext_tiny.pth \
  --ckpt_b ./checkpoints/orient_effnetv2s.pth \
  --test ./data/test
```

No new tests were run for this docs-only inventory.

## Risks

- Accidentally replacing the NN detector with a heuristic would discard validated work.
- Moving experiment data/checkpoints would break existing scripts and local paths.
- Importing model checkpoints into git would add large binary artifacts without a policy decision.
- Applying rotation in the workbench without clarifying convention could rotate pages in the wrong direction.
- EXIF handling differs across parts of the subsystem and needs a narrow integration decision.
- Ensemble import path is fragile if used outside the experiment directory.
- Workbench orientation should remain advisory/reviewable; it should not silently mutate original image files.

## Preservation Rules

- Do not delete `experiments/orientation_model/`.
- Do not remove training scripts, prediction scripts, local dataset layout, or checkpoint directory references.
- Do not replace the NN detector with a heuristic without explicit user approval.
- Do not retrain unless explicitly requested.
- Do not move model artifacts without preserving paths and updating docs.
- Do not commit `.pth` checkpoint files by default.
- Do not silently rotate source images.
- Workbench integration should call the existing detector through a wrapper.
- Orientation results should be inspectable and overrideable in the workbench.

## Recommended Integration Seam

Add a small wrapper around the existing NN predictor rather than rewriting it.

Recommended future file:

```text
src/martial_arts_ocr/review/orientation_service.py
```

Reason:

- orientation is needed first by the local review workbench;
- the service can remain review/workbench-scoped before any pipeline default changes;
- it can call `utils.image.preprocessing.orientation_cnn` and preserve the existing experiment model code;
- it can normalize path configuration, checkpoint availability, EXIF assumptions, confidence metadata, and correction-direction naming.

Suggested future interface:

```python
@dataclass(frozen=True)
class OrientationResult:
    rotation_degrees: int
    confidence: float | None
    source: str
    scores_by_degree: dict[int, float]
    metadata: dict[str, Any]


class OrientationService:
    def predict_page_orientation(self, image_path: Path) -> OrientationResult:
        ...
```

Suggested workbench behavior:

```text
Run Orientation
  -> call existing NN via wrapper
  -> show predicted degree, confidence, scores, model used
  -> allow reviewer override
  -> save orientation review state
  -> do not rotate source image silently
```

## Recommended Next Pass

Implement an orientation service wrapper for the workbench that calls the existing NN detector.

The next pass should:

- use `utils/image/preprocessing/orientation_cnn.py`;
- configure checkpoint paths explicitly;
- report missing checkpoints cleanly;
- expose prediction, confidence, score map, and model-used metadata;
- add tests with a fake model/predictor;
- avoid changing OCR, extraction, or runtime defaults;
- avoid rotating images automatically;
- add a workbench UI control only after the wrapper contract is tested.
