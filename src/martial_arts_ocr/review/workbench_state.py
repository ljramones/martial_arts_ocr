"""File-backed state for the local research review workbench."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


REGION_TYPES = [
    "ignore",
    "image",
    "diagram",
    "photo",
    "english_text",
    "romanized_japanese_text",
    "modern_japanese_horizontal",
    "modern_japanese_vertical",
    "mixed_english_japanese",
    "caption_label",
    "unknown_needs_review",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class ReviewWorkbenchStore:
    """Manage local review project JSON without invoking OCR/extraction."""

    def __init__(self, project_root: Path, allowed_roots: list[Path] | None = None) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.allowed_roots = [
            Path(root).expanduser().resolve()
            for root in (allowed_roots or [])
        ]
        self.project_root.mkdir(parents=True, exist_ok=True)

    def create_project(self, source_folder: Path, project_id: str | None = None) -> dict[str, Any]:
        source_folder = Path(source_folder).expanduser().resolve()
        self._require_allowed(source_folder)
        if not source_folder.exists() or not source_folder.is_dir():
            raise ValueError(f"Source folder does not exist or is not a directory: {source_folder}")

        pages = self._page_records(source_folder)
        if not pages:
            raise ValueError(f"No supported image files found in: {source_folder}")

        now = _now()
        safe_project_id = _safe_id(project_id) if project_id else self._default_project_id(source_folder)
        state = {
            "project_id": safe_project_id,
            "source_folder": str(source_folder),
            "pages": pages,
            "metadata": {
                "schema_version": 1,
                "created_at": now,
                "updated_at": now,
                "local_only": True,
                "recognition_advisory": True,
            },
        }
        self.save_project(state)
        return state

    def load_project(self, project_id: str) -> dict[str, Any]:
        state_path = self.project_path(project_id)
        if not state_path.exists():
            raise FileNotFoundError(f"Review project not found: {project_id}")
        return json.loads(state_path.read_text(encoding="utf-8"))

    def save_project(self, state: dict[str, Any]) -> dict[str, Any]:
        state = dict(state)
        metadata = dict(state.get("metadata") or {})
        metadata["updated_at"] = _now()
        state["metadata"] = metadata
        project_dir = self.project_dir(state["project_id"])
        project_dir.mkdir(parents=True, exist_ok=True)
        self.project_path(state["project_id"]).write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return state

    def project_dir(self, project_id: str) -> Path:
        return self.project_root / _safe_id(project_id)

    def project_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "project_state.json"

    def get_page(self, state: dict[str, Any], page_id: str) -> dict[str, Any]:
        for page in state.get("pages", []):
            if page.get("page_id") == page_id:
                return page
        raise KeyError(f"Page not found: {page_id}")

    def get_region(self, page: dict[str, Any], region_id: str) -> dict[str, Any]:
        for region in page.get("regions", []):
            if region.get("region_id") == region_id:
                return region
        raise KeyError(f"Region not found: {region_id}")

    def add_region(
        self,
        state: dict[str, Any],
        page_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        page = self.get_page(state, page_id)
        regions = page.setdefault("regions", [])
        region_type = str(payload.get("reviewed_type") or payload.get("type") or "unknown_needs_review")
        if region_type not in REGION_TYPES:
            raise ValueError(f"Unsupported region type: {region_type}")
        bbox = _coerce_bbox(
            payload.get("reviewed_bbox") or payload.get("bbox") or _default_bbox(page),
            page,
        )
        region = _with_effective_fields(
            {
                "region_id": payload.get("region_id") or _next_region_id(regions),
                "detected_type": None,
                "reviewed_type": region_type,
                "detected_bbox": None,
                "reviewed_bbox": bbox,
                "status": str(payload.get("status") or "reviewed"),
                "source": "manual",
                "notes": str(payload.get("notes") or ""),
                "ignored": region_type == "ignore",
                "review_status": str(payload.get("review_status") or "manually_added"),
                "training_feedback": {
                    "label": "manually_added",
                    "target_type": region_type,
                },
            }
        )
        regions.append(region)
        self.save_project(state)
        return region

    def update_region(
        self,
        state: dict[str, Any],
        page_id: str,
        region_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        page = self.get_page(state, page_id)
        region = self.get_region(page, region_id)
        if "reviewed_type" in payload or "type" in payload:
            reviewed_type = str(payload.get("reviewed_type") or payload.get("type"))
            if reviewed_type not in REGION_TYPES:
                raise ValueError(f"Unsupported region type: {reviewed_type}")
            region["reviewed_type"] = reviewed_type
        if "reviewed_bbox" in payload or "bbox" in payload:
            region["reviewed_bbox"] = _coerce_bbox(
                payload.get("reviewed_bbox") or payload.get("bbox"),
                page,
            )
        if "notes" in payload:
            region["notes"] = str(payload.get("notes") or "")
        if "status" in payload:
            region["status"] = str(payload["status"])
        elif "reviewed_type" in payload or "type" in payload or "reviewed_bbox" in payload or "bbox" in payload:
            region["status"] = "reviewed"
        if payload.get("ignored") is not None:
            region["ignored"] = bool(payload["ignored"])
        if region.get("reviewed_type") == "ignore" or region.get("status") == "ignored":
            region["ignored"] = True
            region["status"] = "ignored"
        if region.get("source") not in {"manual", "reviewer_manual_duplicate"}:
            region["source"] = "reviewer_override"
        region.update(_effective_fields(region))
        _update_review_feedback(region)
        self.save_project(state)
        return region

    def duplicate_region(
        self,
        state: dict[str, Any],
        page_id: str,
        region_id: str,
        direction: str = "same",
    ) -> dict[str, Any]:
        page = self.get_page(state, page_id)
        regions = page.setdefault("regions", [])
        source_region = self.get_region(page, region_id)
        bbox = list(source_region.get("effective_bbox") or source_region.get("reviewed_bbox") or source_region.get("detected_bbox") or _default_bbox(page))
        new_bbox = _offset_bbox_for_duplicate(bbox, page, direction)
        effective_type = source_region.get("effective_type") or source_region.get("reviewed_type") or source_region.get("detected_type") or "unknown_needs_review"
        if effective_type not in REGION_TYPES:
            effective_type = "unknown_needs_review"
        metadata = dict(source_region.get("metadata") or {})
        duplicate_metadata = {
            "duplicated_from_region_id": region_id,
            "duplicated_from_source": source_region.get("source"),
            "duplicate_direction": direction,
        }
        if metadata.get("detector"):
            duplicate_metadata["duplicated_from_detector"] = metadata.get("detector")
        region = _with_effective_fields(
            {
                "region_id": _next_region_id(regions),
                "detected_type": None,
                "reviewed_type": effective_type,
                "detected_bbox": None,
                "reviewed_bbox": new_bbox,
                "status": "reviewed",
                "source": "reviewer_manual_duplicate",
                "notes": f"Duplicated from {region_id}",
                "ignored": effective_type == "ignore",
                "review_status": "manually_added",
                "metadata": duplicate_metadata,
                "training_feedback": {
                    "label": "manually_added",
                    "target_type": effective_type,
                    "related_machine_regions": [region_id] if source_region.get("source") == "machine_detection" else [],
                },
            }
        )
        regions.append(region)
        self.save_project(state)
        return region

    def delete_region(self, state: dict[str, Any], page_id: str, region_id: str) -> None:
        page = self.get_page(state, page_id)
        regions = page.setdefault("regions", [])
        page["regions"] = [
            region for region in regions
            if region.get("region_id") != region_id
        ]
        if len(page["regions"]) == len(regions):
            raise KeyError(f"Region not found: {region_id}")
        self.save_project(state)

    def add_region_ocr_attempt(
        self,
        state: dict[str, Any],
        page_id: str,
        region_id: str,
        attempt: dict[str, Any],
    ) -> dict[str, Any]:
        page = self.get_page(state, page_id)
        region = self.get_region(page, region_id)
        if not region.get("effective_bbox"):
            raise ValueError(f"Region has no effective bbox: {region_id}")
        attempts = page.setdefault("ocr_attempts", [])
        attempt_record = {
            **dict(attempt),
            "attempt_id": attempt.get("attempt_id") or _next_ocr_attempt_id(attempts),
            "region_id": region_id,
            "region_type": region.get("effective_type") or "unknown_needs_review",
            "bbox": list(region.get("effective_bbox")),
            "orientation_degrees": _effective_orientation_degrees(page),
            "created_at": _now(),
        }
        attempts.append(attempt_record)
        region_attempts = region.setdefault("ocr_attempt_ids", [])
        region_attempts.append(attempt_record["attempt_id"])
        region["last_ocr_attempt_id"] = attempt_record["attempt_id"]
        self.save_project(state)
        return attempt_record

    def import_detected_regions(
        self,
        state: dict[str, Any],
        page_id: str,
        detected_regions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Replace unreviewed machine detections while preserving reviewer work."""
        page = self.get_page(state, page_id)
        existing_regions = page.setdefault("regions", [])
        preserved_regions = [
            region for region in existing_regions
            if not _is_replaceable_machine_region(region)
        ]
        imported_regions = []
        for detected in detected_regions:
            bbox = _coerce_bbox(detected.get("bbox"), page)
            metadata = dict(detected.get("metadata") or {})
            confidence = detected.get("confidence")
            if confidence is not None:
                metadata.setdefault("confidence", confidence)
            detector = detected.get("detector") or metadata.get("detector") or "review_mode_extraction"
            metadata.setdefault("detector", detector)

            detected_type = _map_detected_region_type(
                detected.get("detected_type") or detected.get("region_type") or detected.get("type"),
                metadata,
            )
            region = _with_effective_fields(
                {
                    "region_id": _next_detected_region_id(preserved_regions + imported_regions),
                    "detected_type": detected_type,
                    "reviewed_type": None,
                    "detected_bbox": bbox,
                    "reviewed_bbox": None,
                    "status": "detected",
                    "source": "machine_detection",
                    "notes": str(detected.get("notes") or ""),
                    "ignored": False,
                    "review_status": "unreviewed",
                    "metadata": metadata,
                }
            )
            if confidence is not None:
                region["confidence"] = confidence
            if metadata.get("needs_review") is not None:
                region["needs_review"] = bool(metadata.get("needs_review"))
            imported_regions.append(region)

        page["regions"] = preserved_regions + imported_regions
        page["status"] = "regions_detected"
        page["recognition"] = {
            "last_run_at": _now(),
            "detected_count": len(imported_regions),
            "preserved_region_count": len(preserved_regions),
            "rerun_behavior": "replaced_unreviewed_machine_detection_regions",
            "orientation_degrees": _effective_orientation_degrees(page),
        }
        self.save_project(state)
        return page

    def update_page_orientation(
        self,
        state: dict[str, Any],
        page_id: str,
        *,
        detected_rotation_degrees: int | None = None,
        detected_confidence: float | None = None,
        reviewed_rotation_degrees: int | None = None,
        source: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update page orientation metadata without rotating the source image."""
        page = self.get_page(state, page_id)
        previous_rotation = _effective_orientation_degrees(page)
        orientation = dict(page.get("orientation") or _default_orientation())

        if detected_rotation_degrees is not None:
            orientation["detected_rotation_degrees"] = _coerce_rotation(detected_rotation_degrees)
        if detected_confidence is not None:
            orientation["detected_confidence"] = float(detected_confidence)
        if reviewed_rotation_degrees is not None:
            orientation["reviewed_rotation_degrees"] = _coerce_rotation(reviewed_rotation_degrees)
        if source is not None:
            orientation["source"] = str(source)
        if status is not None:
            orientation["status"] = str(status)
        if metadata is not None:
            existing_metadata = dict(orientation.get("metadata") or {})
            existing_metadata.update(metadata)
            orientation["metadata"] = existing_metadata

        orientation["effective_rotation_degrees"] = (
            orientation.get("reviewed_rotation_degrees")
            if orientation.get("reviewed_rotation_degrees") is not None
            else orientation.get("detected_rotation_degrees", 0)
        )
        orientation["updated_at"] = _now()
        page["orientation"] = orientation
        _update_effective_dimensions(page)

        new_rotation = _effective_orientation_degrees(page)
        if page.get("regions") and new_rotation != previous_rotation:
            _mark_regions_stale_for_orientation_change(page, previous_rotation, new_rotation)

        self.save_project(state)
        return page

    def image_path(self, state: dict[str, Any], page_id: str) -> Path:
        page = self.get_page(state, page_id)
        path = Path(page["source_path"]).expanduser().resolve()
        self._require_allowed(path)
        if not path.exists():
            raise FileNotFoundError(f"Page image not found: {path}")
        return path

    def _page_records(self, source_folder: Path) -> list[dict[str, Any]]:
        records = []
        for index, path in enumerate(_image_files(source_folder), start=1):
            width, height = _image_size(path)
            records.append(
                {
                    "page_id": f"page_{index:03d}",
                    "source_path": str(path),
                    "filename": path.name,
                    "width": width,
                    "height": height,
                    "effective_width": width,
                    "effective_height": height,
                    "orientation": _default_orientation(),
                    "regions": [],
                    "status": "new",
                    "notes": "",
                }
            )
        return records

    def _default_project_id(self, source_folder: Path) -> str:
        slug = _safe_id(source_folder.name or "review_project")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{slug}_{timestamp}"

    def _require_allowed(self, path: Path) -> None:
        if not self.allowed_roots:
            return
        resolved = path.expanduser().resolve()
        for root in self.allowed_roots:
            if resolved == root or root in resolved.parents:
                return
        raise PermissionError(f"Path is outside configured review roots: {path}")


def _image_files(source_folder: Path) -> list[Path]:
    return [
        path.resolve()
        for path in sorted(source_folder.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _default_bbox(page: dict[str, Any]) -> list[int]:
    width = int(page.get("effective_width") or page.get("width") or 200)
    height = int(page.get("effective_height") or page.get("height") or 200)
    return [20, 20, max(1, min(160, width - 20)), max(1, min(100, height - 20))]


def _coerce_bbox(raw_bbox: Any, page: dict[str, Any]) -> list[int]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError("bbox must be [x, y, width, height]")
    x, y, width, height = [int(round(float(value))) for value in raw_bbox]
    page_width = max(1, int(page.get("effective_width") or page.get("width") or x + width or 1))
    page_height = max(1, int(page.get("effective_height") or page.get("height") or y + height or 1))
    x = max(0, min(x, page_width - 1))
    y = max(0, min(y, page_height - 1))
    width = max(1, min(width, page_width - x))
    height = max(1, min(height, page_height - y))
    return [x, y, width, height]


def _offset_bbox_for_duplicate(
    bbox: list[int],
    page: dict[str, Any],
    direction: str,
) -> list[int]:
    x, y, width, height = [int(value) for value in bbox]
    direction = str(direction or "same").strip().lower()
    if direction == "left":
        x -= width
    elif direction == "right":
        x += width
    elif direction == "up":
        y -= height
    elif direction == "down":
        y += height
    return _coerce_bbox([x, y, width, height], page)


def _with_effective_fields(region: dict[str, Any]) -> dict[str, Any]:
    region.update(_effective_fields(region))
    return region


def _effective_fields(region: dict[str, Any]) -> dict[str, Any]:
    effective_type = region.get("reviewed_type") or region.get("detected_type") or "unknown_needs_review"
    effective_bbox = region.get("reviewed_bbox") or region.get("detected_bbox")
    return {
        "effective_type": effective_type,
        "effective_bbox": effective_bbox,
    }


def _update_review_feedback(region: dict[str, Any]) -> None:
    """Record review feedback without mutating preserved detector evidence."""
    effective_type = region.get("effective_type") or region.get("reviewed_type") or region.get("detected_type")
    source = region.get("source")
    detected_bbox = region.get("detected_bbox")
    reviewed_bbox = region.get("reviewed_bbox")
    detected_type = region.get("detected_type")
    reviewed_type = region.get("reviewed_type")

    if region.get("ignored") or region.get("status") == "ignored" or effective_type == "ignore":
        region["review_status"] = "ignored" if source in {"manual", "reviewer_manual_duplicate"} else "rejected"
        region["training_feedback"] = {
            "label": "ignored" if source in {"manual", "reviewer_manual_duplicate"} else "false_positive",
            "target_type": effective_type or "ignore",
            "reason": "reviewer_ignored",
        }
        return

    if source in {"manual", "reviewer_manual_duplicate"} or not detected_bbox:
        feedback = dict(region.get("training_feedback") or {})
        feedback.setdefault("label", "manually_added")
        feedback["target_type"] = effective_type or "unknown_needs_review"
        region["training_feedback"] = feedback
        region["review_status"] = str(region.get("review_status") or "manually_added")
        return

    feedback_label = "accepted_positive"
    review_status = "accepted"
    if reviewed_bbox and reviewed_bbox != detected_bbox:
        feedback_label = "resized_positive"
        review_status = "resized"
    elif reviewed_type and detected_type and reviewed_type != detected_type:
        feedback_label = "type_corrected"

    region["review_status"] = review_status
    region["training_feedback"] = {
        "label": feedback_label,
        "target_type": effective_type or "unknown_needs_review",
    }


def _next_region_id(regions: list[dict[str, Any]]) -> str:
    max_index = 0
    for region in regions:
        match = re.fullmatch(r"r_(\d+)", str(region.get("region_id", "")))
        if match:
            max_index = max(max_index, int(match.group(1)))
    return f"r_{max_index + 1:03d}"


def _next_detected_region_id(regions: list[dict[str, Any]]) -> str:
    max_index = 0
    for region in regions:
        match = re.fullmatch(r"det_(\d+)", str(region.get("region_id", "")))
        if match:
            max_index = max(max_index, int(match.group(1)))
    return f"det_{max_index + 1:03d}"


def _next_ocr_attempt_id(attempts: list[dict[str, Any]]) -> str:
    max_index = 0
    for attempt in attempts:
        match = re.fullmatch(r"ocr_(\d+)", str(attempt.get("attempt_id", "")))
        if match:
            max_index = max(max_index, int(match.group(1)))
    return f"ocr_{max_index + 1:03d}"


def _is_replaceable_machine_region(region: dict[str, Any]) -> bool:
    return (
        region.get("source") == "machine_detection"
        and region.get("status") == "detected"
        and not region.get("reviewed_type")
    )


def _map_detected_region_type(raw_type: Any, metadata: dict[str, Any]) -> str:
    if metadata.get("needs_review") or metadata.get("mixed_region"):
        return "unknown_needs_review"
    normalized = str(raw_type or "image").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "figure": "diagram",
        "picture": "image",
        "visual": "image",
        "text": "unknown_needs_review",
        "text_region": "unknown_needs_review",
        "caption": "caption_label",
        "label": "caption_label",
        "caption_or_label": "caption_label",
        "unknown": "unknown_needs_review",
    }
    mapped = aliases.get(normalized, normalized)
    return mapped if mapped in REGION_TYPES else "image"


def _default_orientation() -> dict[str, Any]:
    return {
        "detected_rotation_degrees": 0,
        "detected_confidence": None,
        "reviewed_rotation_degrees": None,
        "effective_rotation_degrees": 0,
        "status": "unreviewed",
        "source": "default",
        "metadata": {
            "rotation_convention": "clockwise_correction_to_apply",
            "model_output_convention": "current_orientation_degrees",
        },
    }


def _coerce_rotation(value: Any) -> int:
    rotation = int(value)
    if rotation not in {0, 90, 180, 270}:
        raise ValueError("rotation must be one of 0, 90, 180, 270")
    return rotation


def _effective_orientation_degrees(page: dict[str, Any]) -> int:
    orientation = dict(page.get("orientation") or _default_orientation())
    return _coerce_rotation(orientation.get("effective_rotation_degrees", 0))


def _update_effective_dimensions(page: dict[str, Any]) -> None:
    rotation = _effective_orientation_degrees(page)
    width = int(page.get("width") or 1)
    height = int(page.get("height") or 1)
    if rotation in {90, 270}:
        page["effective_width"] = height
        page["effective_height"] = width
    else:
        page["effective_width"] = width
        page["effective_height"] = height


def _mark_regions_stale_for_orientation_change(
    page: dict[str, Any],
    previous_rotation: int,
    new_rotation: int,
) -> None:
    timestamp = _now()
    for region in page.get("regions", []):
        region["stale"] = True
        metadata = dict(region.get("metadata") or {})
        metadata["stale_reason"] = "orientation_changed"
        metadata["previous_orientation_degrees"] = previous_rotation
        metadata["current_orientation_degrees"] = new_rotation
        metadata["stale_at"] = timestamp
        region["metadata"] = metadata
    page["regions_stale"] = True
    page["regions_stale_reason"] = "orientation_changed"


def _safe_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return slug or "review_project"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
