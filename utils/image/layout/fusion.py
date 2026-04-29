from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Iterable

from utils.image.regions.core_types import ImageRegion

PADDLE_INSIDE_CLASSICAL_MIN = 0.75
PADDLE_AREA_TIGHTNESS_MAX = 0.80
PADDLE_AREA_TIGHTNESS_MIN = 0.05

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
    fused: list[ImageRegion] = []
    events: list[dict[str, Any]] = []

    for classical in classical_regions:
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
