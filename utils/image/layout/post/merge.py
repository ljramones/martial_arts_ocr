from __future__ import annotations
from typing import Any, Dict, List, Tuple
from utils.image.regions.core_image import ImageRegion
from utils.image.layout.options import RegionDetectionOptions


def remove_overlaps(regions: List[ImageRegion], iou_threshold: float = 0.3) -> List[ImageRegion]:
    """
    Keep largest boxes first and drop others with IoU > threshold.
    """
    if len(regions) <= 1:
        return regions
    regions = sorted(regions, key=lambda r: r.area, reverse=True)
    kept: List[ImageRegion] = []
    for r in regions:
        if not any(r.iou(k) > iou_threshold for k in kept):
            kept.append(r)
    return kept


def merge_overlapping(regions: List[ImageRegion],
                      iou_threshold: float = 0.1,
                      gap: int = 12) -> List[ImageRegion]:
    """
    Merge boxes that overlap (IoU >= threshold) or are within `gap` pixels of each other.
    Produces new ImageRegion boxes spanning merged groups.

    Args:
        regions: list of ImageRegion objects
        iou_threshold: merge if IoU exceeds this value
        gap: allowed pixel gap between boxes to consider them part of same group
    """
    if not regions:
        return []

    boxes = [[r.x, r.y, r.x + r.width, r.y + r.height, r] for r in regions]
    merged = True
    while merged:
        merged = False
        out = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue
            x0, y0, x1, y1, r0 = boxes[i]

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                a0, b0, a1, b1, rj = boxes[j]

                # expanded-touch test (gap px)
                if not (x1 + gap < a0 or a1 + gap < x0 or y1 + gap < b0 or b1 + gap < y0):
                    # compute IoU
                    inter_x0 = max(x0, a0)
                    inter_y0 = max(y0, b0)
                    inter_x1 = min(x1, a1)
                    inter_y1 = min(y1, b1)
                    inter = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
                    area_i = (x1 - x0) * (y1 - y0)
                    area_j = (a1 - a0) * (b1 - b0)
                    iou = inter / float(area_i + area_j - inter + 1e-6)

                    if iou >= iou_threshold or inter > 0:
                        # merge box j into i
                        x0 = min(x0, a0)
                        y0 = min(y0, b0)
                        x1 = max(x1, a1)
                        y1 = max(y1, b1)
                        used[j] = True
                        merged = True

            used[i] = True
            out.append([x0, y0, x1, y1, r0])

        boxes = out

    merged_regions: List[ImageRegion] = []
    for x0, y0, x1, y1, ref in boxes:
        merged_regions.append(ImageRegion(
            x=int(x0),
            y=int(y0),
            width=int(x1 - x0),
            height=int(y1 - y0),
            region_type=ref.region_type,
            confidence=ref.confidence
        ))

    return merged_regions


def consolidate_regions(
    regions: List[ImageRegion],
    options: RegionDetectionOptions,
) -> Tuple[List[ImageRegion], List[Dict[str, Any]]]:
    """Consolidate accepted image regions after text-like filtering.

    This pass deliberately runs after semantic text rejection so rejected text
    candidates cannot suppress valid diagrams. It merges overlapping/adjacent
    accepted candidates and suppresses contained children when the parent crop is
    not excessively broad.
    """
    if len(regions) <= 1:
        return regions, []

    events: List[Dict[str, Any]] = []
    current = list(regions)
    changed = True
    while changed:
        changed = False
        used = [False] * len(current)
        next_regions: List[ImageRegion] = []

        for i, first in enumerate(current):
            if used[i]:
                continue
            merged = first
            used[i] = True

            for j in range(i + 1, len(current)):
                if used[j]:
                    continue
                second = current[j]
                action = _consolidation_action(merged, second, options)
                if action == "contained_suppression":
                    parent, child = _larger_smaller(merged, second)
                    if parent is not merged:
                        merged = parent
                    used[j] = True
                    changed = True
                    events.append({
                        "reason": action,
                        "kept": merged.to_dict(),
                        "suppressed": child.to_dict(),
                    })
                elif action in {"overlap_merge", "adjacent_merge"}:
                    merged = _merge_pair(merged, second, action)
                    used[j] = True
                    changed = True
                    events.append({
                        "reason": action,
                        "merged": merged.to_dict(),
                        "from": [first.to_dict(), second.to_dict()],
                    })

            next_regions.append(merged)

        current = next_regions

    return sorted(current, key=lambda r: (r.y, r.x, -r.area)), events


def _consolidation_action(
    first: ImageRegion,
    second: ImageRegion,
    options: RegionDetectionOptions,
) -> str | None:
    inter = _intersection_area(first, second)
    if inter <= 0 and not options.merge_adjacent_regions:
        return None

    smaller = min(first.area, second.area)
    larger = max(first.area, second.area)
    if smaller <= 0:
        return None

    contained_ratio = inter / float(smaller)
    if contained_ratio >= options.contained_region_suppression_threshold:
        if larger / float(smaller) <= options.contained_parent_max_area_ratio:
            return "contained_suppression"
        return None

    if options.merge_overlapping_regions and first.iou(second) >= options.overlap_merge_iou_threshold:
        return "overlap_merge"

    if options.merge_adjacent_regions and _should_merge_adjacent(first, second, options):
        return "adjacent_merge"

    return None


def _should_merge_adjacent(
    first: ImageRegion,
    second: ImageRegion,
    options: RegionDetectionOptions,
) -> bool:
    gap = _edge_gap(first, second)
    if gap > options.adjacent_merge_gap_px:
        return False

    union_area = _union_bbox_area(first, second)
    covered_area = first.area + second.area - _intersection_area(first, second)
    if covered_area <= 0:
        return False
    if union_area / float(covered_area) > options.adjacent_merge_max_area_growth_ratio:
        return False

    x_overlap = max(0, min(first.x2, second.x2) - max(first.x1, second.x1))
    y_overlap = max(0, min(first.y2, second.y2) - max(first.y1, second.y1))
    min_width = max(1, min(first.width, second.width))
    min_height = max(1, min(first.height, second.height))

    return (
        x_overlap / float(min_width) >= options.adjacent_merge_min_axis_overlap_ratio
        or y_overlap / float(min_height) >= options.adjacent_merge_min_axis_overlap_ratio
    )


def _merge_pair(first: ImageRegion, second: ImageRegion, reason: str) -> ImageRegion:
    x1 = min(first.x1, second.x1)
    y1 = min(first.y1, second.y1)
    x2 = max(first.x2, second.x2)
    y2 = max(first.y2, second.y2)
    confidence_values = [value for value in (first.confidence, second.confidence) if value is not None]
    confidence = max(confidence_values) if confidence_values else None
    metadata = dict(getattr(first, "metadata", {}) or {})
    metadata.update({
        "consolidation_reason": reason,
        "merged_from": [first.to_dict(), second.to_dict()],
    })
    return ImageRegion(
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        region_type=first.region_type or second.region_type,
        confidence=confidence,
        metadata=metadata,
    )


def _larger_smaller(first: ImageRegion, second: ImageRegion) -> Tuple[ImageRegion, ImageRegion]:
    if first.area >= second.area:
        return first, second
    return second, first


def _intersection_area(first: ImageRegion, second: ImageRegion) -> int:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def _union_bbox_area(first: ImageRegion, second: ImageRegion) -> int:
    x1 = min(first.x1, second.x1)
    y1 = min(first.y1, second.y1)
    x2 = max(first.x2, second.x2)
    y2 = max(first.y2, second.y2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def _edge_gap(first: ImageRegion, second: ImageRegion) -> int:
    horizontal_gap = max(0, max(first.x1, second.x1) - min(first.x2, second.x2))
    vertical_gap = max(0, max(first.y1, second.y1) - min(first.y2, second.y2))
    return max(horizontal_gap, vertical_gap)
