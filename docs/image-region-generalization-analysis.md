# Image Region Generalization Analysis

Date: 2026-04-28

## Problem

The DFD detector tuning reduced the original dominant failure: normal title/body/vertical text being accepted as image regions. Corpus 2 is visually similar scanned martial-arts/research material, but the same detector settings regressed:

- body text blocks were accepted as figures
- heading/sidebar bands were accepted
- large mixed text/image crops appeared
- photo-grid pages were sometimes missed

Because the corpora are similar, this should not be solved by per-corpus settings. The detector needed stronger candidate scoring.

## DFD vs Corpus 2

DFD pages worked better because many true image regions have strong visual evidence: large dark shapes, line-art structure, sparse arrows/symbol clusters, or isolated diagrams. Corpus 2 has more article-style pages where scanned text blocks can look like figures after binarization: text has gray compression noise, regular row/column projections, and large merged contours.

Corpus 2 false positives were mostly:

- body-text columns
- title/header bands
- caption/sidebar text
- broad mixed text/image regions
- manuscript-plus-translation blocks

Corpus 2 false negatives were mostly:

- photo grids where individual panels did not satisfy prior isolation thresholds
- cartoon/photo areas adjacent to text
- manuscript strips that look like vertical text

## Brittle Heuristics

The previous filter used hard binary checks such as component density, aspect ratio, and text-like connected components. These helped DFD pages but missed cases where:

- text was compressed into a few large connected components
- photo-like regions had many tiny components
- figure candidates from the heuristic detector were exempted from filtering
- broad crops contained some real visual content but too much text

## Generalization Changes

The detector now scores each candidate with diagnostic features:

- area ratio and aspect ratio
- fill density
- connected component count and component size distribution
- row/column occupancy
- regular horizontal/vertical projection patterns
- edge density and Hough-line evidence
- dark component evidence
- photo-like grayscale/texture evidence
- crop broadness penalty

The classifier exposes:

- `text_like_score`
- `figure_like_score`
- `photo_like_score`
- `sparse_symbol_score`
- `crop_quality_score`

Acceptance is based on combined evidence:

- reject high text score when visual evidence is weak
- reject broad text-like crops unless visual evidence is strong
- preserve labeled diagrams, line art, sparse arrows, and large dark figures
- filter heuristic `figure` candidates instead of blindly exempting them

## Remaining Risks

- Photo-grid recall improved but is still partial.
- Some broad text/image crops remain when photo evidence is strong.
- Vertical manuscript strips remain ambiguous.
- The detector remains review-mode only and disabled by default.

## Recommendation

Use the single stronger review-mode detector for the next broader evaluation pass. Do not add per-corpus presets yet. If failures remain consistent after more pages, improve the scoring features rather than branching by corpus.
