# utils/image/regions/grouping.py
"""
Region grouping utilities (no heavy deps).

Responsibilities
- Merge small sibling regions (e.g., word/char boxes) into text lines
- Cluster regions by proximity with simple, predictable rules
- Keep behavior parameterized (tune per model/page) and deterministic

Return shapes
- merge_regions_into_lines -> List[List[ImageRegion]]   (each inner list is a line, left-to-right)
- lines_to_regions         -> List[ImageRegion]         (one bbox per line, optional padding)
- group_regions_by_proximity -> List[List[ImageRegion]] (each inner list is a cluster)

Notes
- Uses bbox math from geometry.py to avoid code duplication.
- Assumes rectangular ImageRegion.bbox (polygon points are ignored for grouping).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Optional, Sequence, Tuple

# Core types & math
try:
    from utils.image.regions.core_types import ImageRegion, BBox
    from utils.image.regions.geometry import (
        bbox_union, bbox_width, bbox_height, y_overlap_ratio,
        horizontal_overlap, translate_region, expand_region,
    )
except Exception:  # pragma: no cover - legacy import path fallback
    from core_image import ImageRegion, BBox  # type: ignore
    from geometry import (  # type: ignore
        bbox_union, bbox_width, bbox_height, y_overlap_ratio,
        horizontal_overlap, translate_region, expand_region,
    )

__all__ = [
    "merge_regions_into_lines",
    "lines_to_regions",
    "group_regions_by_proximity",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _line_bbox(regs: Sequence[ImageRegion]) -> BBox:
    """Union bbox of all regions in the line (empty -> 0 box)."""
    if not regs:
        return (0, 0, 0, 0)
    bb = regs[0].to_tuple()
    for r in regs[1:]:
        bb = bbox_union(bb, r.to_tuple())
    return bb

def _sort_tl(regs: Sequence[ImageRegion]) -> List[ImageRegion]:
    return sorted(regs, key=lambda r: (r.y1, r.x1))

def _sort_lr(regs: Sequence[ImageRegion]) -> List[ImageRegion]:
    return sorted(regs, key=lambda r: r.x1)

def _x_gap(a: ImageRegion, b: ImageRegion) -> int:
    """Positive pixels between boxes when disjoint on X; negative/zero if touching/overlapping."""
    return b.x1 - a.x2

def _can_merge_to_line(
    r: ImageRegion,
    line_bbox: BBox,
    *,
    y_overlap_min: float,
    max_x_gap: int,
    require_progressive_x: bool,
) -> bool:
    """
    Decide if region r belongs to an existing line defined by line_bbox.
    Conditions (tunable):
      - sufficient vertical overlap with the line box (y_overlap_min)
      - not too far to the right (max_x_gap)
      - (optional) x must not go backwards (progressive left-to-right growth)
    """
    # vertical compatibility
    if y_overlap_ratio(r.to_tuple(), line_bbox) < y_overlap_min:
        return False

    # horizontal adjacency: allow touching or small gaps
    gap = r.x1 - line_bbox[2]  # r starts to the right of the current line box
    if gap > max_x_gap:
        return False

    # monotonic x (avoid "pulling" earlier words that sit to the left)
    if require_progressive_x and r.x2 <= line_bbox[2]:
        return False

    return True

# ---------------------------------------------------------------------------
# Merge small regions into text lines
# ---------------------------------------------------------------------------

def merge_regions_into_lines(
    regions: Sequence[ImageRegion],
    *,
    y_overlap_min: float = 0.3,
    max_x_gap: int = 48,
    require_progressive_x: bool = True,
    sort_within_lines: bool = True,
) -> List[List[ImageRegion]]:
    """
    Merge small sibling regions (characters/words) into line lists.

    Strategy:
      1) Sort all regions top-to-bottom, then left-to-right.
      2) Greedy pass: for each region r, try to append to the most recent line
         whose union bbox passes the vertical overlap & x-gap checks.
      3) If none qualify, start a new line.

    Args:
      y_overlap_min: min vertical overlap ratio (relative to min height of r vs line box).
      max_x_gap: max positive horizontal gap (px) allowed between line box and r.x1.
      require_progressive_x: avoid attaching boxes that jump backwards in x.
      sort_within_lines: ensure final per-line order is left-to-right.

    Returns:
      List of lines; each line is a list[ImageRegion] ordered left-to-right.
    """
    if not regions:
        return []

    lines: List[List[ImageRegion]] = []
    for r in _sort_tl(regions):
        attached = False

        # Try recent lines first (heuristic: locality)
        for line in reversed(lines):
            lb = _line_bbox(line)
            if _can_merge_to_line(
                r, lb,
                y_overlap_min=y_overlap_min,
                max_x_gap=max_x_gap,
                require_progressive_x=require_progressive_x,
            ):
                line.append(r)
                attached = True
                break

        if not attached:
            lines.append([r])

    if sort_within_lines:
        lines = [_sort_lr(line) for line in lines]

    return lines


def lines_to_regions(
    lines: Sequence[Sequence[ImageRegion]],
    *,
    pad: int = 0,
    region_type: Optional[str] = "line",
) -> List[ImageRegion]:
    """
    Convert lines (lists of regions) into single merged regions with union bboxes.
    Optionally expand each line bbox by 'pad' pixels.
    """
    out: List[ImageRegion] = []
    for line in lines:
        if not line:
            continue
        # Union bbox
        bb = _line_bbox(line)
        # Create a representative region (metadata of the first, bbox from union)
        base = line[0]
        merged = replace(base, bbox=bb, region_type=region_type)
        if pad > 0:
            merged = expand_region(merged, pad)
        out.append(merged)
    return out

# ---------------------------------------------------------------------------
# Generic proximity clustering
# ---------------------------------------------------------------------------

def _is_close_to_group(
    r: ImageRegion,
    group_bbox: BBox,
    *,
    max_dx: int,
    max_dy: int,
    y_overlap_min: float,
) -> bool:
    """
    Heuristic: r belongs to a group if it's not too far from its bbox,
    and shares enough vertical support (for text-like grouping).
    """
    # horizontal/vertical allowances
    # r is to the right of group
    dx_right = r.x1 - group_bbox[2]
    # r is to the left of group
    dx_left = group_bbox[0] - r.x2
    # r is above group
    dy_above = group_bbox[1] - r.y2
    # r is below group
    dy_below = r.y1 - group_bbox[3]

    close_horiz = (dx_right <= max_dx) or (dx_left <= max_dx)
    close_vert = (dy_above <= max_dy) or (dy_below <= max_dy)

    if not (close_horiz or close_vert):
        return False

    # Require some vertical compatibility to avoid merging across lines
    if y_overlap_ratio(r.to_tuple(), group_bbox) < y_overlap_min:
        return False

    return True


def group_regions_by_proximity(
    regions: Sequence[ImageRegion],
    *,
    max_dx: int = 64,
    max_dy: int = 24,
    y_overlap_min: float = 0.2,
    sort_members_lr: bool = True,
) -> List[List[ImageRegion]]:
    """
    Cluster regions into groups if they are within (max_dx, max_dy) of the group's bbox
    and share enough vertical overlap (y_overlap_min). Useful for bundling tokens into
    words/lines/paragraph fragments before downstream processing.

    Args:
      max_dx: horizontal tolerance in pixels.
      max_dy: vertical tolerance in pixels.
      y_overlap_min: min vertical overlap ratio vs group bbox.
      sort_members_lr: sort each group's members left-to-right.

    Returns:
      List of groups, each group a list[ImageRegion].
    """
    if not regions:
        return []

    groups: List[List[ImageRegion]] = []
    for r in _sort_tl(regions):
        placed = False
        # Try to attach to an existing group (check most recent first)
        for g in reversed(groups):
            gb = _line_bbox(g)
            if _is_close_to_group(r, gb, max_dx=max_dx, max_dy=max_dy, y_overlap_min=y_overlap_min):
                g.append(r)
                placed = True
                break
        if not placed:
            groups.append([r])

    if sort_members_lr:
        groups = [_sort_lr(g) for g in groups]

    return groups
