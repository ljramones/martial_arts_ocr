# Extraction Quality Rubric

Use this rubric when reviewing real pages in `notebooks/05_real_page_extraction_review.ipynb`.

## Image And Diagram Regions

- Diagrams, photos, figures, tables, and illustrations are detected as image-like regions.
- Normal body text blocks are not misclassified as images.
- Tiny dust, scan marks, punctuation, and isolated blobs are ignored.
- Bounding boxes are close enough for useful crops without cutting off content.
- Crops are written under `data/notebook_outputs/real_page_review/<sample_id>/`.
- Source page dimensions are preserved in metadata or review notes.
- Region IDs, bbox values, crop paths, confidence, and reading order are stable enough to map into `ImageRegion`.

## Text Cleanup

- Meaningful line breaks are preserved for headings, body text, captions, and lists.
- Cleanup does not destructively collapse headings, captions, or paragraphs.
- Japanese characters, kana, and kanji are preserved.
- Romanized Japanese macrons survive cleanup: `koryū`, `budō`, `Daitō-ryū`, `ō`, `ū`.
- Martial arts punctuation survives cleanup: `—`, `・`, `「」`.
- Cleanup avoids ASCII-only normalization.
- Martial arts terms are not damaged by generic OCR correction rules.

## Layout And Reading Order

- Text regions are roughly ordered top-to-bottom and left-to-right for horizontal pages.
- Vertical or unusual Japanese layout is flagged when order is uncertain.
- Captions remain near associated diagrams where possible.
- Diagrams do not badly interrupt text order.
- Multi-column pages are flagged when the current ordering is ambiguous.

## Decision Criteria

- **Safe to integrate:** representative pages pass image, text, and ordering checks with only minor notes.
- **Needs threshold tuning:** useful regions are found, but false positives or loose bboxes recur.
- **Needs algorithm replacement:** important regions are missed or text/image confusion is common.
- **Should remain experimental:** behavior depends on optional models, page type, or fragile settings.
