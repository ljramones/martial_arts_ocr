"""Selected-region OCR service for the local review workbench."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    import pytesseract
except ImportError:  # pragma: no cover - local OCR availability varies
    pytesseract = None

from utils import TextCleaner


@dataclass(frozen=True)
class RegionOCRRoute:
    """OCR routing decision for one reviewed workbench region."""

    language: str | None
    psm: int | None
    preprocess_profile: str = "none"
    engine: str = "tesseract"
    status: str = "ready"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "language": self.language,
            "psm": self.psm,
            "preprocess_profile": self.preprocess_profile,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RegionOCRResult:
    """Serializable OCR result for a reviewed region crop."""

    text: str
    cleaned_text: str
    confidence: float | None
    route: RegionOCRRoute
    status: str
    processing_time: float = 0.0
    bounding_boxes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_attempt(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "cleaned_text": self.cleaned_text,
            "confidence": self.confidence,
            "route": self.route.to_dict(),
            "status": self.status,
            "processing_time": self.processing_time,
            "bounding_boxes": self.bounding_boxes,
            "metadata": self.metadata,
            "source_text_mutated": False,
        }


class RegionOCRService:
    """Run OCR on one reviewed region crop without changing runtime defaults."""

    def __init__(self) -> None:
        self.text_cleaner = TextCleaner()

    def route_for_region_type(self, region_type: str | None) -> RegionOCRRoute:
        region_type = str(region_type or "unknown_needs_review")
        routes = {
            "english_text": RegionOCRRoute(language="eng", psm=6),
            "romanized_japanese_text": RegionOCRRoute(language="eng", psm=6),
            "caption_label": RegionOCRRoute(language="eng", psm=7),
            "modern_japanese_horizontal": RegionOCRRoute(language="jpn", psm=6, preprocess_profile="upscale_2x"),
            "modern_japanese_vertical": RegionOCRRoute(language="jpn_vert", psm=5, preprocess_profile="upscale_2x"),
            "mixed_english_japanese": RegionOCRRoute(language="eng+jpn", psm=6),
        }
        if region_type in routes:
            return routes[region_type]
        return RegionOCRRoute(
            language=None,
            psm=None,
            status="skipped",
            reason=f"Region type {region_type} is not OCR-routable in this slice.",
        )

    def run(
        self,
        *,
        image_path: str | Path,
        bbox: list[int],
        region_type: str,
        region_id: str,
    ) -> RegionOCRResult:
        route = self.route_for_region_type(region_type)
        if route.status == "skipped":
            return RegionOCRResult(
                text="",
                cleaned_text="",
                confidence=None,
                route=route,
                status="skipped",
                metadata={"region_id": region_id, "skip_reason": route.reason},
            )
        if pytesseract is None:
            return RegionOCRResult(
                text="",
                cleaned_text="",
                confidence=None,
                route=route,
                status="unavailable",
                metadata={"region_id": region_id, "error": "pytesseract is not installed"},
            )

        start = time.time()
        try:
            crop = self._crop_image(Path(image_path), bbox)
            crop = self._preprocess(crop, route.preprocess_profile)
            image_array = np.array(crop)
            config = f"--psm {route.psm} --oem 1 -c preserve_interword_spaces=1"
            raw_text = pytesseract.image_to_string(image_array, lang=route.language, config=config)
            data = pytesseract.image_to_data(
                image_array,
                lang=route.language,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            confidence = _average_confidence(data)
            boxes = _word_boxes(data)
            cleaned_text, _stats = self.text_cleaner.clean_text(raw_text)
            return RegionOCRResult(
                text=raw_text,
                cleaned_text=cleaned_text,
                confidence=confidence,
                route=route,
                status="ok",
                processing_time=time.time() - start,
                bounding_boxes=boxes,
                metadata={
                    "region_id": region_id,
                    "bbox": bbox,
                    "crop_width": crop.width,
                    "crop_height": crop.height,
                    "config": config,
                },
            )
        except Exception as exc:
            return RegionOCRResult(
                text="",
                cleaned_text="",
                confidence=None,
                route=route,
                status="error",
                processing_time=time.time() - start,
                metadata={"region_id": region_id, "bbox": bbox, "error": str(exc)},
            )

    @staticmethod
    def _crop_image(image_path: Path, bbox: list[int]) -> Image.Image:
        with Image.open(image_path) as image:
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            x, y, width, height = [int(value) for value in bbox]
            x = max(0, min(x, oriented.width - 1))
            y = max(0, min(y, oriented.height - 1))
            width = max(1, min(width, oriented.width - x))
            height = max(1, min(height, oriented.height - y))
            return oriented.crop((x, y, x + width, y + height))

    @staticmethod
    def _preprocess(image: Image.Image, profile: str) -> Image.Image:
        if profile == "upscale_2x":
            return image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        return image


def _average_confidence(data: dict[str, Any]) -> float:
    values: list[float] = []
    for raw in data.get("conf", []):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value / 100.0)
    return sum(values) / len(values) if values else 0.0


def _word_boxes(data: dict[str, Any]) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    texts = data.get("text", [])
    for index, text in enumerate(texts):
        if not str(text or "").strip():
            continue
        try:
            confidence = float(data.get("conf", [])[index]) / 100.0
        except (IndexError, TypeError, ValueError):
            confidence = 0.0
        if confidence <= 0:
            continue
        boxes.append(
            {
                "text": str(text),
                "confidence": confidence,
                "x": int(data.get("left", [])[index]),
                "y": int(data.get("top", [])[index]),
                "width": int(data.get("width", [])[index]),
                "height": int(data.get("height", [])[index]),
                "source": "ocr_engine",
                "engine": "tesseract",
                "ocr_level": "word",
            }
        )
    return boxes
