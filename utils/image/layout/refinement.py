"""Conservative OCR-mask refinement for mixed image/text candidates.

Bounding boxes in this module are page-coordinate ``(x, y, width, height)``.
The refiner is intentionally opt-in and is meant for review-mode diagnostics,
not authoritative layout segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from utils.image.layout.options import RegionDetectionOptions
from utils.image.regions.core_types import ImageRegion
from utils.text.geometry import (
    build_text_mask,
    compute_text_mask_overlap,
    normalize_bbox,
    normalize_text_boxes,
)


BBoxXYWH = tuple[int, int, int, int]


@dataclass(frozen=True)
class RefinementResult:
    bbox: BBoxXYWH
    original_bbox: BBoxXYWH
    refinement_applied: bool = False
    region_role: str = "image"
    needs_review: bool = False
    mixed_reason: str | None = None
    refinement_strategy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def refine_mixed_candidate(
    image: np.ndarray,
    candidate_region: ImageRegion,
    *,
    ocr_text_boxes: list[Any] | None = None,
    options: RegionDetectionOptions | None = None,
) -> RefinementResult:
    """Refine a mixed image/text candidate using subtractive OCR masks.

    Returns the original bbox unless OCR text boxes are available, the candidate
    has meaningful OCR overlap, and a coherent non-text visual mass can be
    isolated without separating nearby labels from a diagram.
    """
    opts = options or RegionDetectionOptions()
    original_bbox = _region_bbox_xywh(candidate_region)
    base = _base_result(original_bbox, candidate_region)
    boxes = normalize_text_boxes(ocr_text_boxes)
    if not boxes:
        return base

    gray = _to_gray(image)
    h_img, w_img = gray.shape[:2]
    x, y, width, height = _clamp_xywh(original_bbox, w_img, h_img)
    if width <= 0 or height <= 0:
        return base

    overlap = compute_text_mask_overlap((x, y, width, height), build_text_mask(gray.shape[:2], boxes, dilation_px=opts.ocr_text_mask_dilation_px))
    if overlap < opts.mixed_region_min_ocr_overlap:
        return base

    crop = gray[y:y + height, x:x + width]
    if crop.size == 0:
        return base

    local_boxes = _local_intersecting_boxes(boxes, (x, y, width, height))
    local_text_mask = build_text_mask((height, width), local_boxes, dilation_px=opts.ocr_text_mask_dilation_px)
    visual_mask = _visual_saliency_mask(crop)
    subtracted = visual_mask.copy()
    subtracted[local_text_mask > 0] = 0
    subtracted = _connect_visual_mask(subtracted)

    components = _component_bboxes(subtracted, min_area=max(40, int(width * height * 0.002)))
    if not components:
        return _needs_review(
            original_bbox,
            candidate_region,
            overlap=overlap,
            reason="visual_mass_removed_by_ocr_mask",
            fragmentation_count=0,
        )

    selected = _select_component_group(components, crop_area=width * height)
    if not selected:
        return _needs_review(
            original_bbox,
            candidate_region,
            overlap=overlap,
            reason="fragmented_visual_mass",
            fragmentation_count=len(components),
        )

    visual_bbox_local = _union_bboxes(selected)
    expanded_for_labels = _expand_xywh(visual_bbox_local, 20, width, height)
    label_density = _label_density(local_boxes, expanded_for_labels)
    interspersed_labels = len(local_boxes) >= 3 and label_density >= 0.45
    if interspersed_labels:
        return RefinementResult(
            bbox=original_bbox,
            original_bbox=original_bbox,
            refinement_applied=False,
            region_role="mixed_labeled",
            needs_review=True,
            mixed_reason="ocr_labels_interspersed_with_visual_mass",
            refinement_strategy="ocr_mask_subtractive_saliency",
            metadata=_metadata(overlap, len(components), label_density, visual_bbox_local, original_bbox),
        )

    refined_local = _expand_xywh(visual_bbox_local, 10, width, height)
    refined_bbox = (x + refined_local[0], y + refined_local[1], refined_local[2], refined_local[3])
    refined_area = refined_bbox[2] * refined_bbox[3]
    original_area = max(1, width * height)
    area_ratio = refined_area / original_area
    center_shift = _center_shift_ratio((0, 0, width, height), refined_local)

    if area_ratio <= 0.82 and center_shift >= 0.08 and refined_bbox[2] >= 20 and refined_bbox[3] >= 20:
        return RefinementResult(
            bbox=refined_bbox,
            original_bbox=original_bbox,
            refinement_applied=True,
            region_role="image",
            needs_review=False,
            mixed_reason=None,
            refinement_strategy="ocr_mask_subtractive_saliency",
            metadata=_metadata(overlap, len(components), label_density, visual_bbox_local, original_bbox, refined_bbox),
        )

    return _needs_review(
        original_bbox,
        candidate_region,
        overlap=overlap,
        reason="unsafe_or_insufficient_bbox_tightening",
        fragmentation_count=len(components),
        label_density=label_density,
        visual_bbox_local=visual_bbox_local,
    )


def _base_result(original_bbox: BBoxXYWH, region: ImageRegion) -> RefinementResult:
    metadata = dict(getattr(region, "metadata", {}) or {})
    return RefinementResult(
        bbox=original_bbox,
        original_bbox=original_bbox,
        region_role=str(metadata.get("region_role") or "image"),
        metadata={},
    )


def _needs_review(
    original_bbox: BBoxXYWH,
    region: ImageRegion,
    *,
    overlap: float,
    reason: str,
    fragmentation_count: int,
    label_density: float = 0.0,
    visual_bbox_local: BBoxXYWH | None = None,
) -> RefinementResult:
    return RefinementResult(
        bbox=original_bbox,
        original_bbox=original_bbox,
        refinement_applied=False,
        region_role=str((getattr(region, "metadata", {}) or {}).get("region_role") or "mixed"),
        needs_review=True,
        mixed_reason=reason,
        refinement_strategy="ocr_mask_subtractive_saliency",
        metadata=_metadata(overlap, fragmentation_count, label_density, visual_bbox_local, original_bbox),
    )


def _metadata(
    overlap: float,
    fragmentation_count: int,
    label_density: float,
    visual_bbox_local: BBoxXYWH | None,
    original_bbox: BBoxXYWH,
    refined_bbox: BBoxXYWH | None = None,
) -> dict[str, Any]:
    return {
        "mixed_region": True,
        "ocr_text_mask_overlap_ratio": round(float(overlap), 4),
        "fragmentation_count": int(fragmentation_count),
        "label_density": round(float(label_density), 4),
        "visual_bbox_local": _bbox_dict(visual_bbox_local) if visual_bbox_local else None,
        "original_bbox": _bbox_dict(original_bbox),
        "refined_bbox": _bbox_dict(refined_bbox or original_bbox),
    }


def _visual_saliency_mask(crop: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(crop, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 130)
    _, ink = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.inRange(blurred, 0, 150)
    return cv2.bitwise_or(cv2.bitwise_or(edges, ink), dark)


def _connect_visual_mask(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    kernel = np.ones((5, 5), dtype=np.uint8)
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    out = cv2.dilate(out, kernel, iterations=1)
    return out


def _component_bboxes(mask: np.ndarray, *, min_area: int) -> list[BBoxXYWH]:
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    boxes: list[BBoxXYWH] = []
    for index in range(1, num_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, width, height))
    return boxes


def _select_component_group(components: list[BBoxXYWH], *, crop_area: int) -> list[BBoxXYWH]:
    if not components:
        return []
    sorted_components = sorted(components, key=lambda box: box[2] * box[3], reverse=True)
    largest = sorted_components[0]
    largest_area = largest[2] * largest[3]
    selected = [largest]
    for box in sorted_components[1:]:
        area = box[2] * box[3]
        if area >= largest_area * 0.18 or _bbox_gap(largest, box) <= 24:
            selected.append(box)
    union = _union_bboxes(selected)
    if union[2] * union[3] < max(80, crop_area * 0.015):
        return []
    return selected


def _local_intersecting_boxes(boxes: list[Any], candidate_bbox: BBoxXYWH) -> list[dict[str, Any]]:
    cx, cy, cw, ch = candidate_bbox
    local: list[dict[str, Any]] = []
    for box in boxes:
        x, y, width, height = normalize_bbox(box)
        ix1 = max(cx, x)
        iy1 = max(cy, y)
        ix2 = min(cx + cw, x + width)
        iy2 = min(cy + ch, y + height)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        local.append(
            {
                "text": getattr(box, "text", ""),
                "bbox": {
                    "x": ix1 - cx,
                    "y": iy1 - cy,
                    "width": ix2 - ix1,
                    "height": iy2 - iy1,
                },
                "confidence": getattr(box, "confidence", None),
                "level": getattr(box, "level", "word"),
            }
        )
    return local


def _label_density(local_boxes: list[Any], expanded_visual_bbox: BBoxXYWH) -> float:
    if not local_boxes:
        return 0.0
    nearby = 0
    for box in local_boxes:
        x, y, width, height = normalize_bbox(box)
        center = (x + width / 2, y + height / 2)
        if _point_in_bbox(center, expanded_visual_bbox):
            nearby += 1
    return nearby / len(local_boxes)


def _point_in_bbox(point: tuple[float, float], bbox: BBoxXYWH) -> bool:
    x, y, width, height = bbox
    return x <= point[0] <= x + width and y <= point[1] <= y + height


def _union_bboxes(boxes: list[BBoxXYWH]) -> BBoxXYWH:
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return x1, y1, x2 - x1, y2 - y1


def _expand_xywh(bbox: BBoxXYWH, padding: int, max_width: int, max_height: int) -> BBoxXYWH:
    x, y, width, height = bbox
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(max_width, x + width + padding)
    y2 = min(max_height, y + height + padding)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def _bbox_gap(first: BBoxXYWH, second: BBoxXYWH) -> int:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    horizontal = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    vertical = max(0, max(ay, by) - min(ay + ah, by + bh))
    return max(horizontal, vertical)


def _center_shift_ratio(original: BBoxXYWH, refined: BBoxXYWH) -> float:
    ox, oy, ow, oh = original
    rx, ry, rw, rh = refined
    original_center = (ox + ow / 2, oy + oh / 2)
    refined_center = (rx + rw / 2, ry + rh / 2)
    distance = ((original_center[0] - refined_center[0]) ** 2 + (original_center[1] - refined_center[1]) ** 2) ** 0.5
    diagonal = max(1.0, (ow ** 2 + oh ** 2) ** 0.5)
    return distance / diagonal


def _region_bbox_xywh(region: ImageRegion) -> BBoxXYWH:
    return int(region.x), int(region.y), int(region.width), int(region.height)


def _clamp_xywh(bbox: BBoxXYWH, max_width: int, max_height: int) -> BBoxXYWH:
    x, y, width, height = bbox
    x1 = max(0, min(max_width, x))
    y1 = max(0, min(max_height, y))
    x2 = max(0, min(max_width, x + width))
    y2 = max(0, min(max_height, y + height))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def _bbox_dict(bbox: BBoxXYWH | None) -> dict[str, int] | None:
    if bbox is None:
        return None
    x, y, width, height = bbox
    return {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image
