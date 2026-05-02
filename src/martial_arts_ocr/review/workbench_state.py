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
                "status": str(payload.get("status") or "manual"),
                "source": "manual",
                "notes": str(payload.get("notes") or ""),
                "ignored": region_type == "ignore",
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
        region["source"] = "reviewer_override" if region.get("source") != "manual" else "manual"
        region.update(_effective_fields(region))
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
    width = int(page.get("width") or 200)
    height = int(page.get("height") or 200)
    return [20, 20, max(1, min(160, width - 20)), max(1, min(100, height - 20))]


def _coerce_bbox(raw_bbox: Any, page: dict[str, Any]) -> list[int]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError("bbox must be [x, y, width, height]")
    x, y, width, height = [int(round(float(value))) for value in raw_bbox]
    page_width = max(1, int(page.get("width") or x + width or 1))
    page_height = max(1, int(page.get("height") or y + height or 1))
    x = max(0, min(x, page_width - 1))
    y = max(0, min(y, page_height - 1))
    width = max(1, min(width, page_width - x))
    height = max(1, min(height, page_height - y))
    return [x, y, width, height]


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


def _next_region_id(regions: list[dict[str, Any]]) -> str:
    max_index = 0
    for region in regions:
        match = re.fullmatch(r"r_(\d+)", str(region.get("region_id", "")))
        if match:
            max_index = max(max_index, int(match.group(1)))
    return f"r_{max_index + 1:03d}"


def _safe_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return slug or "review_project"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
