# utils/image/regions/layout.py
"""
Layout utilities for page regions (no heavy deps).

Responsibilities
- Split a set of regions into columns using simple, robust gap heuristics
- Sort regions in a Western reading order (columns L->R, rows T->B)
- Keep behavior deterministic and parameterized

Key APIs
- split_regions_into_columns(...)
- sort_regions_reading_order(...)

Design notes
- Works best on "line regions" (e.g., after merge_regions_into_lines + lines_to_regions),
  but also tolerates word/paragraph boxes.
- Column discovery is based on the largest gaps between region x-centers; no clustering libs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

try:
    from utils.image.regions.core_types import ImageRegion, BBox
    from utils.image.regions.geometry import bbox_union
except Exception:  # pragma: no cover (legacy import paths)
    from core_image import ImageRegion, BBox  # type: ignore
    from geometry import bbox_union  # type: ignore

__all__ = [
    "split_regions_into_columns",
    "sort_regions_reading_order",
]


# ------------------------------ small helpers ---------------------------------

def _tl_key(r: ImageRegion) -> Tuple[int, int]:
    return (r.y1, r.x1)

def _sort_tl(regs: Sequence[ImageRegion]) -> List[ImageRegion]:
    return sorted(regs, key=_tl_key)

def _x_center(r: ImageRegion) -> float:
    return (r.x1 + r.x2) * 0.5

def _union(regs: Sequence[ImageRegion]) -> BBox:
    if not regs:
        return (0, 0, 0, 0)
    bb = regs[0].to_tuple()
    for r in regs[1:]:
        bb = bbox_union(bb, r.to_tuple())
    return bb


# -------------------------- column split (gap-based) --------------------------

def split_regions_into_columns(
    regions: Sequence[ImageRegion],
    *,
    num_columns: Optional[int] = None,
    min_gutter: int = 32,
    max_columns: int = 4,
) -> List[List[ImageRegion]]:
    """
    Split regions into columns using largest x-center gaps.

    Algorithm (gap heuristic):
      1) Sort regions by x-center.
      2) Compute consecutive gaps between x-centers.
      3) Pick the (k-1) largest gaps as column separators.
         - If num_columns is None, choose k = 1..max_columns such that
           k-1 gaps >= min_gutter exist (favor larger k).
      4) Partition by those separators and return columns in left-to-right order.

    Args:
      num_columns: if provided, force that many columns (>=1).
      min_gutter: minimum pixel gap between adjacent columns' centers.
      max_columns: upper bound when auto-selecting.

    Returns:
      List of columns, each a list[ImageRegion] sorted top-left inside the column.
      If no valid split is found, returns a single column with all regions.
    """
    regs = list(regions)
    if not regs:
        return []

    # Sort by x-center; keep original regions
    regs.sort(key=_x_center)
    centers = [_x_center(r) for r in regs]

    # Compute gaps between consecutive centers
    gaps: List[Tuple[float, int]] = []  # (gap_size, index_between_i_and_i+1)
    for i in range(len(centers) - 1):
        gaps.append((centers[i + 1] - centers[i], i))

    # Candidate separators: large enough gaps
    big = [(g, idx) for (g, idx) in gaps if g >= float(min_gutter)]
    big.sort(reverse=True)  # largest gaps first

    # Decide how many columns
    if num_columns is None:
        # we can create at most len(big)+1 columns, but cap with max_columns
        k = min(max_columns, len(big) + 1)
        if k <= 1:
            # no big gaps: single column
            return [_sort_tl(regs)]
        # use top (k-1) separators
        use = big[: k - 1]
    else:
        k = max(1, int(num_columns))
        if k == 1:
            return [_sort_tl(regs)]
        # ensure we have enough big gaps to justify k columns
        if len(big) < (k - 1):
            # not enough separation; degrade gracefully
            k = min(len(big) + 1, k)
            if k <= 1:
                return [_sort_tl(regs)]
        use = big[: k - 1]

    # Build cut indices: indices after which a split occurs
    cut_after = sorted(idx for (_g, idx) in use)  # ascending by position

    # Partition regs according to cuts
    cols: List[List[ImageRegion]] = []
    start = 0
    for cut in cut_after:
        cols.append(_sort_tl(regs[start : cut + 1]))
        start = cut + 1
    cols.append(_sort_tl(regs[start :]))  # tail

    # Drop empties defensively (shouldn't happen)
    cols = [c for c in cols if c]

    return cols


# ---------------------------- reading-order sort ------------------------------

def sort_regions_reading_order(
    regions: Sequence[ImageRegion],
    *,
    y_tolerance: int = 8,
    known_columns: Optional[List[List[ImageRegion]]] = None,
    min_gutter: int = 32,
    max_columns: int = 4,
) -> List[ImageRegion]:
    """
    Sort regions in a Western reading order:
      - Columns left-to-right
      - Within each column: top-to-bottom, then left-to-right with a small y banding

    Strategy:
      - If `known_columns` provided, use them (assumed L->R).
      - Else, split into columns with `split_regions_into_columns` (gap heuristic).
      - For each column, sort by bands of y (rows) so that minor jitter doesn't shuffle order.

    Args:
      y_tolerance: vertical pixel tolerance for line banding.
      known_columns: optional pre-split columns (overrides auto split).
      min_gutter/max_columns: passed to auto splitting when known_columns is None.

    Returns:
      Flattened list of regions in reading order.
    """
    if not regions:
        return []

    if known_columns is None:
        columns = split_regions_into_columns(
            regions, num_columns=None, min_gutter=min_gutter, max_columns=max_columns
        )
    else:
        # trust caller-provided columns; ensure stable order inside
        columns = [ _sort_tl(col) for col in known_columns if col ]

    ordered: List[ImageRegion] = []

    # Row-banding sorter: bucket by y1/y_tolerance, then x1
    def band_key(r: ImageRegion) -> Tuple[int, int]:
        band = r.y1 // max(1, y_tolerance)
        return (band, r.x1)

    for col in columns:
        col_sorted = sorted(col, key=band_key)
        ordered.extend(col_sorted)

    return ordered
