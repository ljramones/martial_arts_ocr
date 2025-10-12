from __future__ import annotations
from typing import List
from utils.image.regions.core_image import ImageRegion


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
