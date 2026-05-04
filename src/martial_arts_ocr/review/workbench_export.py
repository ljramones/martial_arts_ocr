"""Reviewed page export artifacts for the local research workbench."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


TEXT_REGION_TYPES = {
    "english_text",
    "romanized_japanese_text",
    "modern_japanese_horizontal",
    "modern_japanese_vertical",
    "mixed_english_japanese",
    "caption_label",
}


def export_page_review(
    *,
    state: dict[str, Any],
    page: dict[str, Any],
    project_dir: Path,
    effective_image_path: Path,
    export_id: str | None = None,
) -> dict[str, Any]:
    """Write durable review artifacts for one workbench page.

    Export files are local research artifacts. They preserve raw OCR evidence
    and reviewer corrections separately; they do not mutate project state.
    """

    export_dir = _unique_export_dir(project_dir / "exports", export_id)
    crops_dir = export_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    created_at = _now()

    snapshot_path = export_dir / "project_state_snapshot.json"
    snapshot_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    attempts = list(page.get("ocr_attempts") or [])
    with Image.open(effective_image_path) as image:
        effective_image = image.convert("RGB")
        exported_regions = [
            _export_region(region, attempts, effective_image, crops_dir)
            for region in page.get("regions", [])
        ]

    review = {
        "project_id": state.get("project_id"),
        "page_id": page.get("page_id"),
        "source_path": page.get("source_path"),
        "filename": page.get("filename"),
        "orientation": page.get("orientation") or {},
        "regions": exported_regions,
        "export_metadata": {
            "created_at": created_at,
            "format_version": 1,
            "source_text_mutated": False,
        },
    }

    json_path = export_dir / f"page_{page.get('page_id')}_review.json"
    json_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown_path = export_dir / f"page_{page.get('page_id')}_review.md"
    markdown_path.write_text(_render_markdown(review), encoding="utf-8")

    text_path = export_dir / f"page_{page.get('page_id')}_text.txt"
    text_path.write_text(_render_text_export(exported_regions), encoding="utf-8")

    return {
        "export_dir": str(export_dir),
        "files": {
            "project_state_snapshot": str(snapshot_path),
            "page_review_json": str(json_path),
            "page_review_markdown": str(markdown_path),
            "page_text": str(text_path),
            "crops_dir": str(crops_dir),
        },
        "summary": {
            "page_id": page.get("page_id"),
            "region_count": len(exported_regions),
            "crop_count": sum(1 for region in exported_regions if region.get("crop_path")),
            "text_region_count": sum(
                1
                for region in exported_regions
                if region.get("effective_type") in TEXT_REGION_TYPES
                and not _is_ignored_region(region)
            ),
            "created_at": created_at,
        },
    }


def _export_region(
    region: dict[str, Any],
    attempts: list[dict[str, Any]],
    effective_image: Image.Image,
    crops_dir: Path,
) -> dict[str, Any]:
    attempt = _latest_attempt_for_region(region, attempts)
    crop_path = None
    if not _is_ignored_region(region) and region.get("effective_bbox"):
        crop_path = _write_crop(region, effective_image, crops_dir)

    raw_text = attempt.get("text") if attempt else None
    cleaned_text = attempt.get("cleaned_text") if attempt else None
    reviewed_text = attempt.get("reviewed_text") if attempt else None
    review_status = attempt.get("review_status") if attempt else None
    preferred_text = _preferred_text(attempt)

    return {
        "region_id": region.get("region_id"),
        "detected_type": region.get("detected_type"),
        "reviewed_type": region.get("reviewed_type"),
        "effective_type": region.get("effective_type"),
        "detected_bbox": region.get("detected_bbox"),
        "reviewed_bbox": region.get("reviewed_bbox"),
        "effective_bbox": region.get("effective_bbox"),
        "source": region.get("source"),
        "status": region.get("status"),
        "review_status": region.get("review_status"),
        "ignored": bool(region.get("ignored") or region.get("effective_type") == "ignore"),
        "notes": region.get("notes") or "",
        "metadata": region.get("metadata") or {},
        "training_feedback": region.get("training_feedback") or {},
        "ocr": {
            "latest_attempt_id": attempt.get("attempt_id") if attempt else None,
            "route": attempt.get("route") if attempt else None,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "reviewed_text": reviewed_text,
            "review_status": review_status,
            "preferred_text": preferred_text,
            "source_text_mutated": bool(attempt.get("source_text_mutated")) if attempt else False,
        },
        "crop_path": crop_path,
    }


def _latest_attempt_for_region(
    region: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    attempt_id = region.get("last_ocr_attempt_id")
    if attempt_id:
        for attempt in attempts:
            if attempt.get("attempt_id") == attempt_id:
                return attempt
    region_id = region.get("region_id")
    for attempt in reversed(attempts):
        if attempt.get("region_id") == region_id:
            return attempt
    return None


def _preferred_text(attempt: dict[str, Any] | None) -> str:
    if not attempt or attempt.get("review_status") == "rejected":
        return ""
    reviewed_text = attempt.get("reviewed_text")
    if reviewed_text:
        return str(reviewed_text)
    return str(attempt.get("cleaned_text") or attempt.get("text") or "")


def _write_crop(region: dict[str, Any], image: Image.Image, crops_dir: Path) -> str | None:
    bbox = region.get("effective_bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x, y, width, height = [int(value) for value in bbox]
    x = max(0, min(x, image.width - 1))
    y = max(0, min(y, image.height - 1))
    width = max(1, min(width, image.width - x))
    height = max(1, min(height, image.height - y))
    crop = image.crop((x, y, x + width, y + height))
    filename = f"region_{_safe_filename(region.get('region_id') or 'unknown')}.png"
    crop.save(crops_dir / filename)
    return f"crops/{filename}"


def _render_markdown(review: dict[str, Any]) -> str:
    lines = [
        f"# Page Review Export: {review.get('page_id')}",
        "",
        "## Source",
        "",
        f"- Project: `{review.get('project_id')}`",
        f"- Page: `{review.get('page_id')}`",
        f"- Source path: `{review.get('source_path')}`",
        "",
        "## Orientation",
        "",
    ]
    orientation = review.get("orientation") or {}
    lines.extend(
        [
            f"- Detected rotation: `{orientation.get('detected_rotation_degrees')}`",
            f"- Effective correction: `{orientation.get('effective_rotation_degrees')}`",
            f"- Status: `{orientation.get('status')}`",
            "",
            "## Regions",
            "",
        ]
    )
    for region in review.get("regions", []):
        ocr = region.get("ocr") or {}
        lines.extend(
            [
                f"### Region {region.get('region_id')} - {region.get('effective_type')}",
                "",
                f"- BBox: `{region.get('effective_bbox')}`",
                f"- Source: `{region.get('source')}`",
                f"- Status: `{region.get('status')}`",
                f"- Review status: `{region.get('review_status')}`",
                f"- Crop: `{region.get('crop_path') or ''}`",
                f"- OCR route: `{ocr.get('route') or ''}`",
                f"- OCR review status: `{ocr.get('review_status') or ''}`",
                "",
            ]
        )
        if ocr.get("reviewed_text"):
            lines.extend(["#### Reviewed Text", "", "```text", str(ocr.get("reviewed_text")), "```", ""])
        if ocr.get("raw_text") or ocr.get("cleaned_text"):
            lines.extend(["#### Raw OCR", "", "```text", str(ocr.get("cleaned_text") or ocr.get("raw_text") or ""), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _render_text_export(regions: list[dict[str, Any]]) -> str:
    text_regions = [
        region
        for region in regions
        if region.get("effective_type") in TEXT_REGION_TYPES
        and not _is_ignored_region(region)
    ]
    text_regions.sort(key=lambda region: _reading_order_key(region.get("effective_bbox")))
    chunks = [
        str((region.get("ocr") or {}).get("preferred_text") or "").strip()
        for region in text_regions
    ]
    return "\n\n".join(chunk for chunk in chunks if chunk) + ("\n" if any(chunks) else "")


def _reading_order_key(bbox: Any) -> tuple[int, int]:
    if isinstance(bbox, list) and len(bbox) == 4:
        return int(bbox[1]), int(bbox[0])
    return 0, 0


def _is_ignored_region(region: dict[str, Any]) -> bool:
    return bool(
        region.get("ignored")
        or region.get("effective_type") == "ignore"
        or region.get("status") == "ignored"
    )


def _unique_export_dir(root: Path, export_id: str | None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base_name = _safe_filename(export_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    candidate = root / base_name
    index = 2
    while candidate.exists():
        candidate = root / f"{base_name}_{index}"
        index += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value)).strip("._-") or "export"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
