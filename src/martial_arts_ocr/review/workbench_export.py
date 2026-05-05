"""Reviewed page export artifacts for the local research workbench."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime
from html import escape
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

IMAGE_REGION_TYPES = {"image", "diagram", "photo"}
EXPORT_V2_FORMATS = {"review_bundle", "html", "docx"}


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


def export_project_review_v2(
    *,
    state: dict[str, Any],
    project_dir: Path,
    effective_image_paths: dict[str, Path],
    page_selection: dict[str, Any],
    formats: list[str],
    options: dict[str, Any] | None = None,
    export_id: str | None = None,
) -> dict[str, Any]:
    """Write Export v2 artifacts for one or more reviewed workbench pages."""

    normalized_formats = _normalize_formats(formats)
    normalized_options = _normalize_export_options(options)
    pages = resolve_page_selection(state, page_selection)
    created_at = _now()
    export_dir = _unique_export_dir(project_dir / "exports", export_id)
    snapshot_path = export_dir / "project_state_snapshot.json"
    snapshot_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    review_bundle_paths: dict[str, str] = {}
    html_paths: dict[str, str] = {}
    review_pages_dir = export_dir / "review_bundle" / "pages"
    review_crops_dir = export_dir / "review_bundle" / "crops"
    html_dir = export_dir / "html"
    html_assets_dir = html_dir / "assets"
    docx_dir = export_dir / "docx"
    docx_assets_dir = docx_dir / "assets"

    if "review_bundle" in normalized_formats:
        review_pages_dir.mkdir(parents=True, exist_ok=True)
        review_crops_dir.mkdir(parents=True, exist_ok=True)
        review_bundle_paths["root"] = str(export_dir / "review_bundle")
    if "html" in normalized_formats:
        html_assets_dir.mkdir(parents=True, exist_ok=True)
        html_paths["root"] = str(html_dir)
    docx_paths: dict[str, str] = {}
    if "docx" in normalized_formats:
        docx_assets_dir.mkdir(parents=True, exist_ok=True)
        docx_paths["root"] = str(docx_dir)

    page_models = []
    for page_index, page in enumerate(pages, start=1):
        page_id = str(page.get("page_id"))
        image_path = effective_image_paths[page_id]
        with Image.open(image_path) as image:
            effective_image = image.convert("RGB")
            review_regions = [
                _export_region(
                    region,
                    list(page.get("ocr_attempts") or []),
                    effective_image,
                    review_crops_dir,
                    crop_filename_prefix=f"{page_id}_",
                    crop_path_prefix="crops",
                    write_crop="review_bundle" in normalized_formats,
                )
                for region in page.get("regions", [])
            ]
            page_model = _build_page_model(
                state=state,
                page=page,
                page_index=page_index,
                exported_regions=review_regions,
                effective_image=effective_image,
                html_assets_dir=html_assets_dir,
                write_html_assets="html" in normalized_formats,
                docx_assets_dir=docx_assets_dir,
                write_docx_assets="docx" in normalized_formats,
            )
        page_models.append(page_model)

        if "review_bundle" in normalized_formats:
            page_review = _page_review_document(
                state=state,
                page=page,
                exported_regions=review_regions,
                created_at=created_at,
                format_version=2,
            )
            (review_pages_dir / f"{page_id}_review.json").write_text(
                json.dumps(page_review, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (review_pages_dir / f"{page_id}_review.md").write_text(
                _render_markdown(page_review),
                encoding="utf-8",
            )
            (review_pages_dir / f"{page_id}_text.txt").write_text(
                _render_text_export(review_regions),
                encoding="utf-8",
            )

    document_model = {
        "project_id": state.get("project_id"),
        "source_folder": state.get("source_folder"),
        "pages": page_models,
        "formats_requested": normalized_formats,
        "created_at": created_at,
        "export_version": 2,
        "source_text_mutated": False,
        "options": normalized_options,
        "page_selection": {
            "mode": page_selection.get("mode") or "current",
            "page_ids": [page.get("page_id") for page in pages],
        },
        "warnings": _document_warnings(page_models),
    }
    export_model_path = export_dir / "document_export_model.json"
    export_model_path.write_text(
        json.dumps(document_model, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if "html" in normalized_formats:
        html_path = html_dir / "document.html"
        html_path.write_text(_render_html_document(document_model), encoding="utf-8")
        html_paths["document"] = str(html_path)
        html_paths["assets_dir"] = str(html_assets_dir)
    if "docx" in normalized_formats:
        docx_path = docx_dir / "document.docx"
        _write_docx_document(
            document_model,
            docx_path,
            export_dir,
            options=normalized_options.get("docx") or {},
        )
        docx_paths["document"] = str(docx_path)
        docx_paths["assets_dir"] = str(docx_assets_dir)

    manifest = {
        "project_id": state.get("project_id"),
        "created_at": created_at,
        "page_selection": {
            "mode": page_selection.get("mode") or "current",
            "page_ids": [page.get("page_id") for page in pages],
        },
        "formats": normalized_formats,
        "options": normalized_options,
        "source_text_mutated": False,
        "export_version": 2,
        "artifact_paths": {
            "project_state_snapshot": str(snapshot_path),
            "document_export_model": str(export_model_path),
            "review_bundle": review_bundle_paths or None,
            "html": html_paths or None,
            "docx": docx_paths or None,
        },
    }
    manifest_path = export_dir / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "export_id": export_dir.name,
        "export_path": str(export_dir),
        "manifest_path": str(manifest_path),
        "formats": {
            "review_bundle": review_bundle_paths.get("root") if review_bundle_paths else None,
            "html": html_paths.get("document") if html_paths else None,
            "docx": docx_paths.get("document") if docx_paths else None,
        },
        "page_ids": [page.get("page_id") for page in pages],
        "summary": {
            "page_count": len(pages),
            "format_count": len(normalized_formats),
            "created_at": created_at,
            "source_text_mutated": False,
        },
    }


def resolve_page_selection(
    state: dict[str, Any],
    page_selection: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve Export v2 page selection to project-ordered page records."""

    pages = list(state.get("pages") or [])
    page_by_id = {str(page.get("page_id")): page for page in pages}
    mode = str(page_selection.get("mode") or "current").strip()

    if mode in {"current", "current_page"}:
        page_ids = [str(page_id) for page_id in page_selection.get("page_ids") or []]
        if not page_ids and page_selection.get("page_id"):
            page_ids = [str(page_selection["page_id"])]
        if len(page_ids) != 1:
            raise ValueError("current page export requires exactly one page_id")
    elif mode == "selected":
        page_ids = [str(page_id) for page_id in page_selection.get("page_ids") or []]
    elif mode == "range":
        page_range = page_selection.get("range") or {}
        start = str(page_range.get("start") or page_selection.get("start") or "")
        end = str(page_range.get("end") or page_selection.get("end") or "")
        page_ids = _page_range_ids(pages, start, end)
    elif mode == "all":
        page_ids = [str(page.get("page_id")) for page in pages]
    else:
        raise ValueError(f"Unsupported page selection mode: {mode}")

    if not page_ids:
        raise ValueError("export page selection is empty")
    missing = [page_id for page_id in page_ids if page_id not in page_by_id]
    if missing:
        raise ValueError(f"Unknown export page id(s): {', '.join(missing)}")

    requested = set(page_ids)
    return [page for page in pages if str(page.get("page_id")) in requested]


def _export_region(
    region: dict[str, Any],
    attempts: list[dict[str, Any]],
    effective_image: Image.Image,
    crops_dir: Path,
    *,
    crop_filename_prefix: str = "",
    crop_path_prefix: str = "crops",
    write_crop: bool = True,
) -> dict[str, Any]:
    attempt = _latest_attempt_for_region(region, attempts)
    crop_path = None
    if write_crop and not _is_ignored_region(region) and region.get("effective_bbox"):
        crop_path = _write_crop(
            region,
            effective_image,
            crops_dir,
            filename_prefix=crop_filename_prefix,
            path_prefix=crop_path_prefix,
        )

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


def _write_crop(
    region: dict[str, Any],
    image: Image.Image,
    crops_dir: Path,
    *,
    filename_prefix: str = "",
    path_prefix: str = "crops",
) -> str | None:
    bbox = region.get("effective_bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x, y, width, height = [int(value) for value in bbox]
    x = max(0, min(x, image.width - 1))
    y = max(0, min(y, image.height - 1))
    width = max(1, min(width, image.width - x))
    height = max(1, min(height, image.height - y))
    crop = image.crop((x, y, x + width, y + height))
    filename = (
        f"{_safe_filename_prefix(filename_prefix)}"
        f"region_{_safe_filename(region.get('region_id') or 'unknown')}.png"
    )
    crop.save(crops_dir / filename)
    return f"{path_prefix}/{filename}"


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


def _build_page_model(
    *,
    state: dict[str, Any],
    page: dict[str, Any],
    page_index: int,
    exported_regions: list[dict[str, Any]],
    effective_image: Image.Image,
    html_assets_dir: Path,
    write_html_assets: bool,
    docx_assets_dir: Path,
    write_docx_assets: bool,
) -> dict[str, Any]:
    blocks = []
    assets = []
    for region in sorted(exported_regions, key=lambda item: _reading_order_key(item.get("effective_bbox"))):
        block_type = _block_type(region)
        asset_path = None
        if write_html_assets and block_type == "image" and not _is_ignored_region(region):
            asset_path = _write_crop(
                region,
                effective_image,
                html_assets_dir,
                filename_prefix=f"{page.get('page_id')}_",
                path_prefix="assets",
            )
            if asset_path:
                assets.append(asset_path)
        docx_asset_path = None
        if write_docx_assets and block_type == "image" and not _is_ignored_region(region):
            docx_asset_path = _write_crop(
                region,
                effective_image,
                docx_assets_dir,
                filename_prefix=f"{page.get('page_id')}_",
                path_prefix="docx/assets",
            )
            if docx_asset_path:
                assets.append(docx_asset_path)
        ocr = region.get("ocr") or {}
        blocks.append(
            {
                "block_id": f"{page.get('page_id')}_{region.get('region_id')}",
                "type": block_type,
                "region_id": region.get("region_id"),
                "bbox": region.get("effective_bbox"),
                "text": ocr.get("preferred_text") or "",
                "raw_ocr": ocr.get("cleaned_text") or ocr.get("raw_text") or "",
                "ocr_route": ocr.get("route"),
                "review_status": ocr.get("review_status") or region.get("review_status"),
                "region_type": region.get("effective_type"),
                "source": region.get("source"),
                "status": region.get("status"),
                "asset_path": asset_path,
                "docx_asset_path": docx_asset_path,
                "notes": region.get("notes") or "",
                "needs_review": _block_needs_review(region, ocr),
                "source_text_mutated": bool(ocr.get("source_text_mutated")),
            }
        )
    return {
        "project_id": state.get("project_id"),
        "page_id": page.get("page_id"),
        "page_index": page_index,
        "source_path": page.get("source_path"),
        "filename": page.get("filename"),
        "orientation": page.get("orientation") or {},
        "review_status": _page_review_status(exported_regions),
        "blocks": blocks,
        "assets": assets,
        "warnings": _page_warnings(page, exported_regions, blocks),
    }


def _page_review_document(
    *,
    state: dict[str, Any],
    page: dict[str, Any],
    exported_regions: list[dict[str, Any]],
    created_at: str,
    format_version: int,
) -> dict[str, Any]:
    return {
        "project_id": state.get("project_id"),
        "page_id": page.get("page_id"),
        "source_path": page.get("source_path"),
        "filename": page.get("filename"),
        "orientation": page.get("orientation") or {},
        "regions": exported_regions,
        "export_metadata": {
            "created_at": created_at,
            "format_version": format_version,
            "source_text_mutated": False,
        },
    }


def _render_html_document(document_model: dict[str, Any]) -> str:
    pages = list(document_model.get("pages") or [])
    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <title>Workbench Review Export</title>",
        "  <style>",
        "    :root { color-scheme: light; --ink: #202124; --muted: #62717f; --line: #d9e1e8; --panel: #f7f9fb; --warn-bg: #fff8e5; --warn: #7c5700; --ok-bg: #eaf6ef; --ok: #25633f; }",
        "    body { background: #ffffff; color: var(--ink); font-family: system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; line-height: 1.5; margin: 0; }",
        "    main { margin: 0 auto; max-width: 1040px; padding: 2rem; }",
        "    header.document-header { border-bottom: 2px solid var(--line); margin-bottom: 1.5rem; padding-bottom: 1rem; }",
        "    h1 { font-size: 1.8rem; margin: 0 0 0.5rem; }",
        "    h2 { font-size: 1.35rem; margin: 0; }",
        "    h3 { font-size: 1rem; margin: 0; }",
        "    .summary-grid { display: grid; gap: 0.65rem; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 1rem; }",
        "    .summary-item { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 0.65rem; }",
        "    .summary-item strong { display: block; font-size: 0.78rem; text-transform: uppercase; color: var(--muted); }",
        "    nav.toc { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; margin: 1rem 0 2rem; padding: 1rem; }",
        "    nav.toc ol { margin: 0.5rem 0 0; padding-left: 1.25rem; }",
        "    .page { border-top: 3px solid #2f5f7d; margin-top: 2rem; padding-top: 1rem; }",
        "    .page-header { display: grid; gap: 0.5rem; grid-template-columns: 1fr; margin-bottom: 1rem; }",
        "    .page-meta, .region-meta { color: var(--muted); font-size: 0.9rem; }",
        "    .warning-list { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.75rem 0; }",
        "    .badge { border-radius: 999px; display: inline-block; font-size: 0.78rem; font-weight: 700; padding: 0.18rem 0.5rem; }",
        "    .badge.warning { background: var(--warn-bg); color: var(--warn); }",
        "    .badge.audit { background: #e8f1f8; color: #24506d; }",
        "    .badge.ok { background: var(--ok-bg); color: var(--ok); }",
        "    .region-block { border: 1px solid var(--line); border-radius: 7px; margin: 1rem 0; overflow: hidden; }",
        "    .region-header { align-items: start; background: var(--panel); display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: space-between; padding: 0.75rem 0.9rem; }",
        "    .region-body { padding: 0.9rem; }",
        "    .reviewed-text { white-space: pre-wrap; background: #ffffff; border-left: 4px solid #2f7d52; margin: 0.75rem 0; padding: 0.75rem 0.9rem; }",
        "    details.raw-ocr { margin-top: 0.75rem; }",
        "    details.raw-ocr summary { cursor: pointer; color: #24506d; font-weight: 700; }",
        "    pre { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #f3f5f7; border-radius: 5px; overflow-x: auto; padding: 0.75rem; }",
        "    figure { margin: 0.75rem 0; }",
        "    img { max-width: 100%; height: auto; border: 1px solid #cfd8df; border-radius: 4px; }",
        "    figcaption { color: var(--muted); font-size: 0.86rem; margin-top: 0.35rem; }",
        "    .muted { color: var(--muted); }",
        "    @media print { main { max-width: none; padding: 1rem; } .region-block { break-inside: avoid; } nav.toc { break-after: page; } }",
        "  </style>",
        "</head>",
        "<body>",
        "<main>",
        "  <header class=\"document-header\">",
        f"    <h1>Workbench Review Export: {escape(str(document_model.get('project_id') or ''))}</h1>",
        "    <div class=\"summary-grid\">",
        _html_summary_item("Created", document_model.get("created_at")),
        _html_summary_item("Pages", len(pages)),
        _html_summary_item("Export version", document_model.get("export_version")),
        _html_summary_item("Formats", ", ".join(str(item) for item in document_model.get("formats_requested") or [])),
        _html_summary_item("Source text mutated", "false"),
        "    </div>",
        "    <p><span class=\"badge ok\">source_text_mutated=false</span> <span class=\"badge audit\">raw OCR preserved in review bundle</span></p>",
        "  </header>",
        "  <nav class=\"toc\" aria-label=\"Page table of contents\">",
        "    <h2>Pages</h2>",
        "    <ol>",
    ]
    for page in pages:
        page_label = _page_label(page)
        lines.append(
            f"      <li><a href=\"#{escape(str(page.get('page_id')))}\">"
            f"Page {escape(str(page.get('page_index') or ''))} - {escape(page_label)}</a></li>"
        )
    lines.extend(["    </ol>", "  </nav>"])
    for page in pages:
        page_label = _page_label(page)
        blocks = list(page.get("blocks") or [])
        image_count = sum(1 for block in blocks if block.get("type") == "image")
        text_count = sum(1 for block in blocks if block.get("type") in {"text", "caption"})
        lines.extend(
            [
                f"  <section class=\"page\" id=\"{escape(str(page.get('page_id')))}\">",
                "    <div class=\"page-header\">",
                f"      <h2>Page {escape(str(page.get('page_index') or ''))} - {escape(page_label)}</h2>",
                f"      <div class=\"page-meta\">Page ID: <code>{escape(str(page.get('page_id')))}</code> · Source: <code>{escape(str(page.get('source_path') or ''))}</code></div>",
                f"      <div class=\"page-meta\">Orientation: {_html_orientation(page.get('orientation') or {})}</div>",
                f"      <div class=\"page-meta\">Regions: {len(blocks)} total · {text_count} text/caption · {image_count} image/diagram/photo</div>",
                "    </div>",
            ]
        )
        warnings = page.get("warnings") or []
        if warnings:
            lines.append("    <div class=\"warning-list\" aria-label=\"Page warnings\">")
            for warning in warnings:
                lines.append(f"      <span class=\"badge warning\">{escape(str(warning))}</span>")
            lines.append("    </div>")
        for block in blocks:
            if block.get("type") == "ignored":
                continue
            badges = _html_block_badges(block)
            lines.extend(
                [
                    "    <section class=\"region-block\">",
                    "      <div class=\"region-header\">",
                    f"        <h3>{escape(str(block.get('region_id')))} - {escape(str(block.get('region_type') or block.get('type') or ''))}</h3>",
                    f"        <div>{badges}</div>",
                    "      </div>",
                    "      <div class=\"region-body\">",
                    f"        <div class=\"region-meta\">BBox: <code>{escape(str(block.get('bbox')))}</code> · Source: {escape(str(block.get('source') or ''))} · Status: {escape(str(block.get('status') or ''))}</div>",
                ]
            )
            if block.get("ocr_route"):
                lines.append(f"        <div class=\"region-meta\">OCR route: <code>{escape(str(block.get('ocr_route')))}</code></div>")
            if block.get("asset_path"):
                lines.extend(
                    [
                        "        <figure>",
                        f"          <img src=\"{escape(str(block.get('asset_path')))}\" alt=\"{escape(str(block.get('region_id')))} crop\">",
                        f"          <figcaption>Crop asset: <code>{escape(str(block.get('asset_path')))}</code></figcaption>",
                        "        </figure>",
                    ]
                )
            if block.get("text"):
                lines.extend(
                    [
                        "        <h4>Reviewed / Display Text</h4>",
                        "        <div class=\"reviewed-text\">",
                        escape(str(block.get("text"))),
                        "        </div>",
                    ]
                )
            elif block.get("type") in {"text", "caption"}:
                lines.append("        <p class=\"muted\">No reviewed or OCR text available for this region.</p>")
            if block.get("raw_ocr"):
                lines.extend(
                    [
                        "        <details class=\"raw-ocr\">",
                        "          <summary>Raw / cleaned OCR evidence</summary>",
                        "          <pre>",
                        escape(str(block.get("raw_ocr"))),
                        "          </pre>",
                        "        </details>",
                    ]
                )
            if block.get("notes"):
                lines.append(f"        <p class=\"muted\">Notes: {escape(str(block.get('notes')))}</p>")
            if block.get("type") == "image" and not block.get("asset_path"):
                lines.append("        <p><span class=\"badge warning\">crop_missing</span></p>")
            lines.extend(["      </div>", "    </section>"])
        lines.append("  </section>")
    lines.extend(["</main>", "</body>", "</html>", ""])
    return "\n".join(lines)


def _write_docx_document(
    document_model: dict[str, Any],
    docx_path: Path,
    export_dir: Path,
    *,
    options: dict[str, Any] | None = None,
) -> None:
    docx_options = options or {}
    include_raw_ocr = bool(docx_options.get("include_raw_ocr"))
    image_relationships: list[dict[str, Any]] = []
    body_parts = [
        _docx_paragraph(f"Workbench Review Export: {document_model.get('project_id') or ''}", bold=True),
        _docx_paragraph(f"Created: {document_model.get('created_at') or ''}"),
        _docx_paragraph(f"Pages: {len(document_model.get('pages') or [])}"),
        _docx_paragraph("source_text_mutated=false"),
        _docx_paragraph("Raw OCR is preserved in the review bundle and page Markdown/JSON artifacts."),
    ]
    if include_raw_ocr:
        body_parts.append(_docx_paragraph("Raw OCR appendix: included at end of document."))
    else:
        body_parts.append(_docx_paragraph("DOCX main body omits full raw OCR by default for readability."))
    for page in document_model.get("pages") or []:
        body_parts.extend(
            [
                _docx_paragraph(f"Page {page.get('page_index') or ''} - {_page_label(page)}", bold=True),
                _docx_paragraph(f"Page ID: {page.get('page_id') or ''}"),
                _docx_paragraph(f"Source: {page.get('source_path') or ''}"),
                _docx_paragraph(f"Orientation: {_plain_orientation(page.get('orientation') or {})}"),
            ]
        )
        warnings = page.get("warnings") or []
        if warnings:
            body_parts.append(_docx_paragraph("Warnings: " + ", ".join(str(warning) for warning in warnings), bold=True))
        for block in page.get("blocks") or []:
            if block.get("type") == "ignored":
                continue
            body_parts.extend(
                [
                    _docx_paragraph(
                        f"Region {block.get('region_id') or ''} - {block.get('region_type') or block.get('type') or ''}",
                        bold=True,
                    ),
                    _docx_paragraph(
                        " | ".join(
                            [
                                f"BBox: {block.get('bbox')}",
                                f"Source: {block.get('source') or ''}",
                                f"Status: {block.get('status') or ''}",
                                f"Review: {block.get('review_status') or ''}",
                            ]
                        )
                    ),
                ]
            )
            if block.get("ocr_route"):
                body_parts.append(_docx_paragraph(f"OCR route: {block.get('ocr_route')}"))
            if block.get("text"):
                body_parts.append(_docx_paragraph("Reviewed / Display Text", bold=True))
                body_parts.extend(_docx_text_paragraphs(str(block.get("text"))))
            elif block.get("type") in {"text", "caption"}:
                body_parts.append(_docx_paragraph("No reviewed or OCR text available for this region."))
            if block.get("docx_asset_path"):
                relationship_id = f"rId{len(image_relationships) + 1}"
                image_path = export_dir / str(block.get("docx_asset_path"))
                image_relationships.append(
                    {
                        "relationship_id": relationship_id,
                        "source_path": image_path,
                        "target": f"media/{image_path.name}",
                    }
                )
                body_parts.append(_docx_image_paragraph(image_path, relationship_id, len(image_relationships)))
                body_parts.append(_docx_paragraph(f"Crop: {block.get('docx_asset_path')}"))
            if block.get("raw_ocr"):
                body_parts.append(_docx_paragraph("Raw OCR evidence preserved in review bundle."))
            if block.get("notes"):
                body_parts.append(_docx_paragraph(f"Notes: {block.get('notes')}"))
    if include_raw_ocr:
        appendix_parts = _docx_raw_ocr_appendix(document_model)
        if appendix_parts:
            body_parts.extend(appendix_parts)
    body_parts.append("<w:sectPr><w:pgSz w:w=\"12240\" w:h=\"15840\"/><w:pgMar w:top=\"1080\" w:right=\"1080\" w:bottom=\"1080\" w:left=\"1080\" w:header=\"720\" w:footer=\"720\" w:gutter=\"0\"/></w:sectPr>")
    document_xml = _docx_document_xml("\n".join(body_parts))
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _docx_content_types(image_relationships))
        archive.writestr("_rels/.rels", _docx_package_relationships())
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/_rels/document.xml.rels", _docx_document_relationships(image_relationships))
        for relationship in image_relationships:
            source_path = Path(relationship["source_path"])
            if source_path.exists():
                archive.write(source_path, f"word/{relationship['target']}")


def _html_summary_item(label: str, value: Any) -> str:
    return (
        "      <div class=\"summary-item\">"
        f"<strong>{escape(label)}</strong>"
        f"{escape(str(value if value is not None else ''))}"
        "</div>"
    )


def _page_label(page: dict[str, Any]) -> str:
    return str(page.get("filename") or page.get("page_id") or "page")


def _html_orientation(orientation: dict[str, Any]) -> str:
    if not orientation:
        return "not recorded"
    detected = orientation.get("detected_rotation_degrees")
    effective = orientation.get("effective_rotation_degrees")
    status = orientation.get("status") or ""
    source = orientation.get("source") or ""
    return escape(
        f"detected={detected}°, correction={effective}°, status={status}, source={source}"
    )


def _plain_orientation(orientation: dict[str, Any]) -> str:
    if not orientation:
        return "not recorded"
    return (
        f"detected={orientation.get('detected_rotation_degrees')} degrees, "
        f"correction={orientation.get('effective_rotation_degrees')} degrees, "
        f"status={orientation.get('status') or ''}, "
        f"source={orientation.get('source') or ''}"
    )


def _xml_escape(value: Any) -> str:
    return escape(str(value if value is not None else ""), quote=True)


def _docx_document_xml(body_xml: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f"<w:body>{body_xml}</w:body>"
        "</w:document>"
    )


def _docx_paragraph(text: Any, *, bold: bool = False) -> str:
    run_properties = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return (
        "<w:p><w:r>"
        f"{run_properties}<w:t xml:space=\"preserve\">{_xml_escape(text)}</w:t>"
        "</w:r></w:p>"
    )


def _docx_text_paragraphs(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return [_docx_paragraph(line) for line in lines] or [_docx_paragraph("")]


def _docx_raw_ocr_appendix(document_model: dict[str, Any]) -> list[str]:
    appendix_parts = [_docx_paragraph("Appendix: Raw OCR Evidence", bold=True)]
    raw_region_count = 0
    for page in document_model.get("pages") or []:
        for block in page.get("blocks") or []:
            raw_ocr = str(block.get("raw_ocr") or "")
            if not raw_ocr:
                continue
            raw_region_count += 1
            display_text = str(block.get("text") or "")
            appendix_parts.extend(
                [
                    _docx_paragraph(
                        f"Page: {page.get('page_id') or ''} | Region: {block.get('region_id') or ''}",
                        bold=True,
                    ),
                    _docx_paragraph(f"Type: {block.get('region_type') or block.get('type') or ''}"),
                    _docx_paragraph(f"Review status: {block.get('review_status') or ''}"),
                    _docx_paragraph(f"OCR route: {block.get('ocr_route') or ''}"),
                ]
            )
            if raw_ocr.strip() == display_text.strip():
                appendix_parts.append(_docx_paragraph("Raw OCR is identical to displayed text."))
            else:
                appendix_parts.append(_docx_paragraph("Raw OCR:", bold=True))
                appendix_parts.extend(_docx_text_paragraphs(raw_ocr))
    if raw_region_count == 0:
        return []
    return appendix_parts


def _docx_image_paragraph(image_path: Path, relationship_id: str, image_index: int) -> str:
    cx, cy = _docx_image_extent(image_path)
    name = _xml_escape(image_path.name)
    return (
        "<w:p><w:r><w:drawing>"
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:docPr id="{image_index}" name="{name}"/>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        "<pic:pic>"
        "<pic:nvPicPr>"
        f'<pic:cNvPr id="{image_index}" name="{name}"/>'
        "<pic:cNvPicPr/>"
        "</pic:nvPicPr>"
        "<pic:blipFill>"
        f'<a:blip r:embed="{relationship_id}"/>'
        "<a:stretch><a:fillRect/></a:stretch>"
        "</pic:blipFill>"
        "<pic:spPr>"
        '<a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        "</pic:spPr>"
        "</pic:pic>"
        "</a:graphicData></a:graphic>"
        "</wp:inline>"
        "</w:drawing></w:r></w:p>"
    )


def _docx_image_extent(image_path: Path) -> tuple[int, int]:
    max_width_emu = int(5.8 * 914400)
    fallback = int(2.5 * 914400)
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        return fallback, fallback
    if width <= 0 or height <= 0:
        return fallback, fallback
    width_emu = int(width / 96 * 914400)
    height_emu = int(height / 96 * 914400)
    if width_emu > max_width_emu:
        scale = max_width_emu / width_emu
        width_emu = max_width_emu
        height_emu = int(height_emu * scale)
    return max(1, width_emu), max(1, height_emu)


def _docx_content_types(image_relationships: list[dict[str, Any]]) -> str:
    extensions = {Path(item["target"]).suffix.lower().lstrip(".") for item in image_relationships}
    defaults = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
    ]
    if "png" in extensions:
        defaults.append('<Default Extension="png" ContentType="image/png"/>')
    if "jpg" in extensions or "jpeg" in extensions:
        defaults.append('<Default Extension="jpg" ContentType="image/jpeg"/>')
        defaults.append('<Default Extension="jpeg" ContentType="image/jpeg"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        + "".join(defaults)
        + '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        + "</Types>"
    )


def _docx_package_relationships() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )


def _docx_document_relationships(image_relationships: list[dict[str, Any]]) -> str:
    relationships = [
        (
            f'<Relationship Id="{_xml_escape(item["relationship_id"])}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="{_xml_escape(item["target"])}"/>'
        )
        for item in image_relationships
    ]
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(relationships)
        + "</Relationships>"
    )


def _html_block_badges(block: dict[str, Any]) -> str:
    badges = [
        _html_badge(str(block.get("type") or "unknown"), "audit"),
    ]
    review_status = block.get("review_status")
    if review_status:
        badges.append(_html_badge(str(review_status), "audit"))
    if block.get("needs_review"):
        badges.append(_html_badge("needs_review", "warning"))
    if block.get("source_text_mutated"):
        badges.append(_html_badge("source_text_mutated=true", "warning"))
    return " ".join(badges)


def _html_badge(label: str, kind: str) -> str:
    return f"<span class=\"badge {escape(kind)}\">{escape(label)}</span>"


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


def _normalize_formats(formats: list[str]) -> list[str]:
    requested = [str(item) for item in (formats or ["review_bundle"])]
    unsupported = [item for item in requested if item not in EXPORT_V2_FORMATS]
    if unsupported:
        raise ValueError(f"Unsupported export format(s): {', '.join(unsupported)}")
    normalized = []
    for item in requested:
        if item not in normalized:
            normalized.append(item)
    if not normalized:
        raise ValueError("at least one export format is required")
    return normalized


def _normalize_export_options(options: dict[str, Any] | None) -> dict[str, Any]:
    raw_options = options or {}
    docx_options = raw_options.get("docx") if isinstance(raw_options, dict) else {}
    if not isinstance(docx_options, dict):
        docx_options = {}
    return {
        "docx": {
            "include_raw_ocr": bool(docx_options.get("include_raw_ocr", False)),
        }
    }


def _page_range_ids(pages: list[dict[str, Any]], start: str, end: str) -> list[str]:
    page_ids = [str(page.get("page_id")) for page in pages]
    if start not in page_ids or end not in page_ids:
        raise ValueError("range start/end page IDs must exist")
    start_index = page_ids.index(start)
    end_index = page_ids.index(end)
    if start_index > end_index:
        start_index, end_index = end_index, start_index
    return page_ids[start_index:end_index + 1]


def _block_type(region: dict[str, Any]) -> str:
    if _is_ignored_region(region):
        return "ignored"
    region_type = region.get("effective_type")
    if region_type in IMAGE_REGION_TYPES:
        return "image"
    if region_type == "caption_label":
        return "caption"
    if region_type in TEXT_REGION_TYPES:
        return "text"
    return "unknown"


def _block_needs_review(region: dict[str, Any], ocr: dict[str, Any]) -> bool:
    metadata = region.get("metadata") or {}
    return bool(
        metadata.get("needs_review")
        or region.get("status") in {"detected", "needs_review"}
        or region.get("review_status") in {"unreviewed", "needs_review"}
        or ocr.get("review_status") in {"unreviewed", "needs_review"}
    )


def _page_review_status(exported_regions: list[dict[str, Any]]) -> str:
    if any(region.get("review_status") in {"unreviewed", None} for region in exported_regions):
        return "needs_review"
    return "reviewed"


def _page_warnings(
    page: dict[str, Any],
    exported_regions: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> list[str]:
    warnings = []
    if page.get("regions_stale"):
        warnings.append("regions_stale")
    if any(region.get("review_status") == "unreviewed" for region in exported_regions):
        warnings.append("regions_unreviewed")
    if any(block.get("type") in {"text", "caption"} and not block.get("text") for block in blocks):
        warnings.append("ocr_unreviewed")
    if not any(block.get("type") in {"text", "caption"} for block in blocks):
        warnings.append("no_text_regions")
    warnings.append("reading_order_uncertain")
    return warnings


def _document_warnings(page_models: list[dict[str, Any]]) -> list[str]:
    warnings = []
    if any(page.get("warnings") for page in page_models):
        warnings.append("page_warnings_present")
    if not page_models:
        warnings.append("no_pages_exported")
    return warnings


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


def _safe_filename_prefix(value: str) -> str:
    if not value:
        return ""
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))
    return safe if safe.endswith("_") else f"{safe}_"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
