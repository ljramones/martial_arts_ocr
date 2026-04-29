# Paddle Fusion Design

## Purpose

This pass adds optional Paddle PPStructureV3 layout evidence to review-mode
image extraction without replacing the classical/OCR-aware detector.

Image extraction remains disabled by default. Paddle fusion is also disabled by
default and must be explicitly enabled for review experiments.

## Configuration

User-facing options:

- `ENABLE_PADDLE_LAYOUT_FUSION=false`
- `PADDLE_LAYOUT_MODEL_DIR=`

Internal V1 constants:

- `PADDLE_INSIDE_CLASSICAL_MIN=0.75`
- `PADDLE_AREA_TIGHTNESS_MAX=0.80`
- `PADDLE_AREA_TIGHTNESS_MIN=0.05`

The internal constants are not public config knobs in this pass.

## Fusion Rule

Fusion runs only on classical candidates already marked as `mixed_region` or
`needs_review`.

For each Paddle/classical pair, the fusion layer records:

- `intersection_area`
- `iou`
- `paddle_inside_classical_ratio`
- `classical_covered_by_paddle_ratio`
- `area_tightness_ratio`

IoU is diagnostic only. It is not the primary match gate because a tight Paddle
figure can sit inside a broad classical crop and still have a low IoU.

A Paddle visual region qualifies only when:

- the Paddle region is mostly inside the classical parent
- the Paddle region is meaningfully tighter than the classical parent
- the Paddle region is not a tiny accidental micro-crop
- the Paddle label is visual: `figure`, `image`, `photo`, or `diagram`

## V1 Limitations

If multiple Paddle visual regions qualify inside one classical mixed parent, V1
emits only the best single region. Metadata records
`multiple_paddle_candidates` and `fusion_limitation` so review tooling can see
where multi-region splitting may be needed later.

Tables are not converted to `ImageRegion` in V1. Paddle table output remains
diagnostic layout evidence only.

## Runtime Behavior

When Paddle fusion is enabled and Paddle is available:

1. `ExtractionService` runs the existing classical/OCR-aware image extraction.
2. It runs the optional Paddle layout strategy.
3. It fuses Paddle visual boxes only into mixed/needs-review classical regions.
4. It records fusion status and events in extraction metadata.

If Paddle is unavailable or fails, extraction continues with classical-only
regions and records the skip/failure metadata. Normal tests do not import or
require PaddleOCR.

## What Remains Before Runtime Use

The validation gate must show that containment fusion improves the broad/mixed
Corpus 2 cases without regressing DFD hard pages or known-good pages. If this
gate fails, the next step is not more config knobs; it is either a richer layout
fusion design, multi-region splitting, or deeper document-layout backend
evaluation.
