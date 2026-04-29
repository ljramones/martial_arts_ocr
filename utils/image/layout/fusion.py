from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Iterable

from utils.image.regions.core_types import ImageRegion

PADDLE_INSIDE_CLASSICAL_MIN = 0.75
PADDLE_AREA_TIGHTNESS_MAX = 0.80
PADDLE_AREA_TIGHTNESS_MIN = 0.05
PADDLE_ADDITIVE_CONFIDENCE_MIN = 0.60
PADDLE_ADDITIVE_MICRO_AREA_RATIO_MIN = 0.05
PADDLE_ADDITIVE_ABSOLUTE_AREA_MIN = 1024
PADDLE_SHARED_SPAN_MIN = 0.50
PADDLE_PROXIMITY_GAP_PX = 48
PADDLE_DUPLICATE_INSIDE_MIN = 0.85
PADDLE_DUPLICATE_AREA_SIMILARITY_MIN = 0.65

VISUAL_LAYOUT_LABELS = {"figure", "image", "photo", "diagram"}


@dataclass(frozen=True)
class FusionResult:
    regions: list[ImageRegion]
    events: list[dict[str, Any]]


def fuse_paddle_layout_regions(
    classical_regions: Iterable[ImageRegion],
    paddle_regions: Iterable[ImageRegion],
    *,
    backend_name: str = "paddle_ppstructure_v3",
) -> FusionResult:
    """Use Paddle visual boxes to tighten classical mixed review candidates.

    Matching is containment-first, not IoU-first. A useful Paddle figure can be
    fully inside a broad classical parent and still have low IoU.
    """

    paddle_candidates = [
        region for region in paddle_regions if _is_visual_layout_region(region)
    ]
    source_regions = list(classical_regions)
    fused: list[ImageRegion] = []
    events: list[dict[str, Any]] = []
    additive_used_paddle_ids: set[int] = set()

    for classical in source_regions:
        if not _is_mixed_review_candidate(classical):
            fused.append(classical)
            continue

        scored = [
            (metrics, paddle)
            for paddle in paddle_candidates
            if (metrics := _match_metrics(classical, paddle))["qualifies"]
        ]

        if not scored:
            metadata = dict(classical.metadata or {})
            metadata.setdefault("layout_fusion_applied", False)
            metadata.setdefault("layout_fusion_reason", "no_qualifying_paddle_visual_region")
            fused_region = replace(classical, metadata=metadata)
            fused.append(fused_region)
            events.append(
                {
                    "fusion_applied": False,
                    "fusion_reason": "no_qualifying_paddle_visual_region",
                    "classical_region": classical.to_dict(),
                    "num_paddle_candidates": len(paddle_candidates),
                }
            )
            continue

        scored.sort(
            key=lambda item: (
                item[0]["paddle_inside_classical_ratio"],
                item[1].confidence or 0.0,
                -item[0]["area_tightness_ratio"],
            ),
            reverse=True,
        )
        metrics, paddle = scored[0]
        additive_used_paddle_ids.add(id(paddle))
        metadata = dict(classical.metadata or {})
        metadata.update(
            {
                "layout_fusion_applied": True,
                "layout_backend": backend_name,
                "classical_source_bbox": list(_xywh(classical)),
                "layout_source_bbox": list(_xywh(paddle)),
                "intersection_area": metrics["intersection_area"],
                "iou": metrics["iou"],
                "paddle_inside_classical_ratio": metrics["paddle_inside_classical_ratio"],
                "classical_covered_by_paddle_ratio": metrics["classical_covered_by_paddle_ratio"],
                "area_tightness_ratio": metrics["area_tightness_ratio"],
                "fusion_reason": "paddle_visual_inside_mixed_classical_region",
                "layout_label": _layout_label(paddle),
                "layout_confidence": paddle.confidence,
                "multiple_paddle_candidates": len(scored) > 1,
                "num_paddle_candidates": len(scored),
            }
        )
        if len(scored) > 1:
            metadata["fusion_limitation"] = "v1_best_single_region_only"

        fused_region = ImageRegion(
            bbox=paddle.bbox,
            region_type=classical.region_type or "figure",
            confidence=paddle.confidence if paddle.confidence is not None else classical.confidence,
            id=classical.id,
            page_index=classical.page_index,
            points=paddle.points,
            metadata=metadata,
        )
        fused.append(fused_region)
        events.append(
            {
                "fusion_applied": True,
                "fusion_reason": "paddle_visual_inside_mixed_classical_region",
                "classical_region": classical.to_dict(),
                "paddle_region": paddle.to_dict(),
                "result_region": fused_region.to_dict(),
                "metrics": metrics,
            }
        )

    additive_regions, additive_events = _add_paddle_visual_regions(
        source_regions,
        fused,
        paddle_candidates,
        additive_used_paddle_ids,
        backend_name,
    )
    fused.extend(additive_regions)
    events.extend(additive_events)

    return FusionResult(regions=fused, events=events)


def _is_mixed_review_candidate(region: ImageRegion) -> bool:
    metadata = region.metadata or {}
    return bool(metadata.get("mixed_region") or metadata.get("needs_review"))


def _is_visual_layout_region(region: ImageRegion) -> bool:
    return _layout_label(region) in VISUAL_LAYOUT_LABELS


def _layout_label(region: ImageRegion) -> str:
    metadata = region.metadata or {}
    label = metadata.get("layout_label") or metadata.get("raw_label") or metadata.get("label") or region.region_type
    label = str(label or "").lower()
    if label in {"figure", "image", "photo", "diagram"}:
        return label
    if "photo" in label or label == "pic":
        return "photo"
    if label == "vision":
        return "image"
    return label


def _match_metrics(classical: ImageRegion, paddle: ImageRegion) -> dict[str, Any]:
    inter = _intersection_area(classical, paddle)
    classical_area = max(1, classical.area)
    paddle_area = max(1, paddle.area)
    union = max(1, classical_area + paddle_area - inter)
    paddle_inside = inter / paddle_area
    classical_covered = inter / classical_area
    tightness = paddle_area / classical_area
    qualifies = (
        paddle_inside >= PADDLE_INSIDE_CLASSICAL_MIN
        and tightness <= PADDLE_AREA_TIGHTNESS_MAX
        and tightness >= PADDLE_AREA_TIGHTNESS_MIN
    )
    return {
        "intersection_area": inter,
        "iou": inter / union,
        "paddle_inside_classical_ratio": paddle_inside,
        "classical_covered_by_paddle_ratio": classical_covered,
        "area_tightness_ratio": tightness,
        "qualifies": qualifies,
    }


def _add_paddle_visual_regions(
    source_regions: list[ImageRegion],
    fused_regions: list[ImageRegion],
    paddle_candidates: list[ImageRegion],
    used_paddle_ids: set[int],
    backend_name: str,
) -> tuple[list[ImageRegion], list[dict[str, Any]]]:
    clean_classical_regions = [
        region for region in source_regions if not _is_mixed_review_candidate(region)
    ]
    mixed_regions = [
        region for region in source_regions if _is_mixed_review_candidate(region)
    ]
    added: list[ImageRegion] = []
    events: list[dict[str, Any]] = []

    for index, mixed in enumerate(mixed_regions):
        if _has_replacement_for_classical(fused_regions, mixed):
            continue
        scored: list[tuple[dict[str, Any], ImageRegion]] = []
        for paddle in paddle_candidates:
            if id(paddle) in used_paddle_ids:
                continue
            if _confidence(paddle) < PADDLE_ADDITIVE_CONFIDENCE_MIN:
                continue
            relation = _relation_metrics(mixed, paddle)
            if not relation["related"]:
                continue
            if _is_micro_paddle_candidate(mixed, paddle):
                continue
            if _duplicates_clean_classical(paddle, clean_classical_regions):
                continue
            scored.append((relation, paddle))

        if not scored:
            continue

        scored.sort(
            key=lambda item: (
                _relation_priority(item[0]["relation_reason"]),
                _confidence(item[1]),
                item[0]["horizontal_span_overlap_ratio"],
                item[0]["vertical_span_overlap_ratio"],
                item[0]["intersection_area"],
            ),
            reverse=True,
        )
        relation, paddle = scored[0]
        used_paddle_ids.add(id(paddle))
        needs_review = relation["area_tightness_ratio"] > 1.0 or relation["relation_reason"] != "partial_overlap"
        region_role = "paddle_added_mixed_or_uncertain" if needs_review else "paddle_added_visual"
        metadata = dict(paddle.metadata or {})
        metadata.update(
            {
                "layout_fusion_applied": True,
                "layout_backend": backend_name,
                "fusion_mode": "paddle_additive",
                "fusion_reason": "paddle_visual_near_unresolved_mixed_classical",
                "layout_source_bbox": list(_xywh(paddle)),
                "related_classical_bbox": list(_xywh(mixed)),
                "related_classical_region_role": (mixed.metadata or {}).get("region_role"),
                "paddle_confidence": paddle.confidence,
                "paddle_label": _layout_label(paddle),
                "needs_review": needs_review,
                "region_role": region_role,
                "horizontal_span_overlap_ratio": relation["horizontal_span_overlap_ratio"],
                "vertical_span_overlap_ratio": relation["vertical_span_overlap_ratio"],
                "relation_reason": relation["relation_reason"],
                "intersection_area": relation["intersection_area"],
                "paddle_inside_classical_ratio": relation["paddle_inside_classical_ratio"],
                "classical_covered_by_paddle_ratio": relation["classical_covered_by_paddle_ratio"],
                "area_tightness_ratio": relation["area_tightness_ratio"],
                "num_additive_paddle_candidates": len(scored),
            }
        )
        if len(scored) > 1:
            metadata["fusion_limitation"] = "v2_best_single_additive_region_per_mixed_parent"
        parent_id = str(mixed.id or f"{index:03d}")
        added_region = ImageRegion(
            bbox=paddle.bbox,
            region_type="figure",
            confidence=paddle.confidence,
            id=f"paddle_additive_{parent_id}_{len(added) + 1:03d}",
            page_index=paddle.page_index,
            points=paddle.points,
            metadata=metadata,
        )
        added.append(added_region)
        events.append(
            {
                "fusion_applied": True,
                "fusion_mode": "paddle_additive",
                "fusion_reason": "paddle_visual_near_unresolved_mixed_classical",
                "classical_region": mixed.to_dict(),
                "paddle_region": paddle.to_dict(),
                "result_region": added_region.to_dict(),
                "metrics": relation,
            }
        )
    return added, events


def _has_replacement_for_classical(fused_regions: list[ImageRegion], classical: ImageRegion) -> bool:
    classical_bbox = list(_xywh(classical))
    for region in fused_regions:
        metadata = region.metadata or {}
        if (
            metadata.get("layout_fusion_applied")
            and metadata.get("fusion_reason") == "paddle_visual_inside_mixed_classical_region"
            and metadata.get("classical_source_bbox") == classical_bbox
        ):
            return True
    return False


def _relation_metrics(classical: ImageRegion, paddle: ImageRegion) -> dict[str, Any]:
    match = _match_metrics(classical, paddle)
    horizontal_overlap = max(0, min(classical.x2, paddle.x2) - max(classical.x1, paddle.x1))
    vertical_overlap = max(0, min(classical.y2, paddle.y2) - max(classical.y1, paddle.y1))
    horizontal_span = horizontal_overlap / max(1, min(classical.width, paddle.width))
    vertical_span = vertical_overlap / max(1, min(classical.height, paddle.height))
    gap = _bbox_gap(classical, paddle)

    relation_reason = None
    if match["intersection_area"] > 0:
        relation_reason = "partial_overlap"
    elif horizontal_span >= PADDLE_SHARED_SPAN_MIN:
        relation_reason = "shared_horizontal_span"
    elif vertical_span >= PADDLE_SHARED_SPAN_MIN:
        relation_reason = "shared_vertical_span"
    elif gap <= PADDLE_PROXIMITY_GAP_PX:
        relation_reason = "proximity"

    return {
        **match,
        "horizontal_span_overlap_ratio": horizontal_span,
        "vertical_span_overlap_ratio": vertical_span,
        "gap_px": gap,
        "relation_reason": relation_reason,
        "related": relation_reason is not None,
    }


def _bbox_gap(first: ImageRegion, second: ImageRegion) -> int:
    horizontal_gap = max(0, max(first.x1, second.x1) - min(first.x2, second.x2))
    vertical_gap = max(0, max(first.y1, second.y1) - min(first.y2, second.y2))
    return max(horizontal_gap, vertical_gap)


def _is_micro_paddle_candidate(classical: ImageRegion, paddle: ImageRegion) -> bool:
    if paddle.area < PADDLE_ADDITIVE_ABSOLUTE_AREA_MIN:
        return True
    ratio = paddle.area / max(1, classical.area)
    return ratio < PADDLE_ADDITIVE_MICRO_AREA_RATIO_MIN


def _duplicates_clean_classical(paddle: ImageRegion, clean_regions: list[ImageRegion]) -> bool:
    for clean in clean_regions:
        metrics = _match_metrics(clean, paddle)
        area_similarity = min(clean.area, paddle.area) / max(1, max(clean.area, paddle.area))
        if metrics["paddle_inside_classical_ratio"] >= PADDLE_DUPLICATE_INSIDE_MIN:
            return True
        if (
            metrics["classical_covered_by_paddle_ratio"] >= PADDLE_DUPLICATE_INSIDE_MIN
            and area_similarity >= PADDLE_DUPLICATE_AREA_SIMILARITY_MIN
        ):
            return True
    return False


def _confidence(region: ImageRegion) -> float:
    return float(region.confidence or 0.0)


def _relation_priority(reason: str | None) -> int:
    return {
        "partial_overlap": 4,
        "shared_horizontal_span": 3,
        "shared_vertical_span": 3,
        "proximity": 1,
    }.get(reason, 0)


def _intersection_area(first: ImageRegion, second: ImageRegion) -> int:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    if x2 <= x1 or y2 <= y1:
        return 0
    return int((x2 - x1) * (y2 - y1))


def _xywh(region: ImageRegion) -> tuple[int, int, int, int]:
    return region.x, region.y, region.width, region.height
