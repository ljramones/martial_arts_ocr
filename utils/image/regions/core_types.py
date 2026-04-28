# utils/image/regions/core_types.py
"""
Core region and metadata types (no heavy dependencies).

This module intentionally avoids numpy and cv2 so it can be imported from
anywhere (including low-level utilities and type definitions) without
creating circular or heavy import chains.

Types:
- BBox:     tuple[int, int, int, int] -> (x1, y1, x2, y2)
- _Box:     internal immutable bbox helper with basic geometry
- ImageRegion: rectangular (with optional polygon) region + light helpers
- ImageInfo:   file/array metadata summary (width/height/etc.)

Notes:
- Coordinates are pixel-space with x2 >= x1 and y2 >= y1 (inclusive-exclusive safe).
- For polygon support, ImageRegion.points is optional List[(x, y)] in image coords.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

BBox = Tuple[int, int, int, int]
FBox = Tuple[float, float, float, float]

__all__ = ["BBox", "FBox", "_Box", "ImageRegion", "ImageInfo"]


# ---------------------------------------------------------------------------
# Internal bbox helper (immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Box:
    x1: int
    y1: int
    x2: int
    y2: int

    # ---- basic geometry ----
    @property
    def width(self) -> int:
        return max(0, int(self.x2) - int(self.x1))

    @property
    def height(self) -> int:
        return max(0, int(self.y2) - int(self.y1))

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.width / 2.0, self.y1 + self.height / 2.0)

    def to_tuple(self) -> BBox:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    # ---- relations ----
    def intersects(self, other: "_Box") -> bool:
        return not (
            self.x2 <= other.x1 or other.x2 <= self.x1 or
            self.y2 <= other.y1 or other.y2 <= self.y1
        )

    def intersection(self, other: "_Box") -> "_Box":
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x2 <= x1 or y2 <= y1:
            return _Box(0, 0, 0, 0)
        return _Box(x1, y1, x2, y2)

    def union(self, other: "_Box") -> "_Box":
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return _Box(x1, y1, x2, y2)

    def iou(self, other: "_Box") -> float:
        inter = self.intersection(other).area
        if inter == 0:
            return 0.0
        denom = float(self.area + other.area - inter) or 1.0
        return inter / denom

    # ---- transforms ----
    def translate(self, dx: int, dy: int) -> "_Box":
        return _Box(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def scale(self, sx: float, sy: Optional[float] = None) -> "_Box":
        sy = sx if sy is None else sy
        return _Box(
            int(round(self.x1 * sx)),
            int(round(self.y1 * sy)),
            int(round(self.x2 * sx)),
            int(round(self.y2 * sy)),
        )

    def expand(self, pad: int) -> "_Box":
        p = max(0, int(pad))
        return _Box(self.x1 - p, self.y1 - p, self.x2 + p, self.y2 + p)

    def shrink(self, pad: int) -> "_Box":
        p = max(0, int(pad))
        return _Box(self.x1 + p, self.y1 + p, self.x2 - p, self.y2 - p)

    def clamp(self, img_w: int, img_h: int) -> "_Box":
        x1 = max(0, min(self.x1, img_w))
        y1 = max(0, min(self.y1, img_h))
        x2 = max(0, min(self.x2, img_w))
        y2 = max(0, min(self.y2, img_h))
        if x2 < x1:
            x2 = x1
        if y2 < y1:
            y2 = y1
        return _Box(x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Region type (rectangular bbox + optional polygon, with light helpers)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False)
class ImageRegion:
    """
    Rectangular region with optional polygon and metadata.

    Required:
      - bbox: (x1,y1,x2,y2) integers (will be normalized by creator code)

    Optional:
      - region_type: semantic label (e.g., "line", "word", "title")
      - score: confidence/probability
      - id: stable identifier
      - page_index: for multi-page documents
      - points: polygon vertices [(x,y), ...] in image coords
      - metadata: free-form dict for extra fields (avoid heavy objects)
    """
    bbox: BBox
    region_type: Optional[str] = None
    score: Optional[float] = None
    id: Optional[str] = None
    page_index: Optional[int] = None
    points: Optional[List[Tuple[float, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        bbox: Optional[BBox] = None,
        region_type: Optional[str] = None,
        score: Optional[float] = None,
        id: Optional[str] = None,
        page_index: Optional[int] = None,
        points: Optional[List[Tuple[float, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        x: Optional[int] = None,
        y: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        confidence: Optional[float] = None,
    ) -> None:
        if bbox is None:
            if None in (x, y, width, height):
                raise TypeError("ImageRegion requires bbox or x/y/width/height")
            bbox = (int(x), int(y), int(x) + int(width), int(y) + int(height))
        x1, y1, x2, y2 = [int(value) for value in bbox]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        object.__setattr__(self, "bbox", (x1, y1, x2, y2))
        object.__setattr__(self, "region_type", region_type)
        object.__setattr__(self, "score", score if score is not None else confidence)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "page_index", page_index)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    # ---- properties backed by bbox ----
    @property
    def x(self) -> int: return self.x1

    @property
    def y(self) -> int: return self.y1

    @property
    def confidence(self) -> Optional[float]: return self.score

    @property
    def x1(self) -> int: return int(self.bbox[0])

    @property
    def y1(self) -> int: return int(self.bbox[1])

    @property
    def x2(self) -> int: return int(self.bbox[2])

    @property
    def y2(self) -> int: return int(self.bbox[3])

    @property
    def width(self) -> int: return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int: return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int: return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.width / 2.0, self.y1 + self.height / 2.0)

    def to_tuple(self) -> BBox:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.to_tuple(),
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "region_type": self.region_type,
            "score": self.score,
            "confidence": self.confidence,
            "id": self.id,
            "page_index": self.page_index,
            "points": self.points,
            "metadata": dict(self.metadata),
        }

    # ---- relations (via _Box for reuse) ----
    def _as_box(self) -> _Box:
        return _Box(self.x1, self.y1, self.x2, self.y2)

    def intersects(self, other: "ImageRegion") -> bool:
        return self._as_box().intersects(other._as_box())

    def iou(self, other: "ImageRegion") -> float:
        return self._as_box().iou(other._as_box())

    # ---- functional transforms (return new ImageRegion) ----
    def translate(self, dx: int, dy: int) -> "ImageRegion":
        nb = self._as_box().translate(dx, dy)
        npts = None
        if self.points:
            npts = [(x + dx, y + dy) for (x, y) in self.points]
        return replace(self, bbox=nb.to_tuple(), points=npts)

    def scale(self, sx: float, sy: Optional[float] = None) -> "ImageRegion":
        nb = self._as_box().scale(sx, sy)
        npts = None
        if self.points:
            sy = sx if sy is None else sy
            npts = [(x * sx, y * sy) for (x, y) in self.points]
        return replace(self, bbox=nb.to_tuple(), points=npts)

    def expand(self, pad: int) -> "ImageRegion":
        nb = self._as_box().expand(pad)
        return replace(self, bbox=nb.to_tuple())

    def shrink(self, pad: int) -> "ImageRegion":
        nb = self._as_box().shrink(pad)
        return replace(self, bbox=nb.to_tuple())

    def clamp(self, img_w: int, img_h: int) -> "ImageRegion":
        nb = self._as_box().clamp(img_w, img_h)
        return replace(self, bbox=nb.to_tuple())


# ---------------------------------------------------------------------------
# Image metadata summary (no decode required)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageInfo:
    """
    Lightweight image metadata.

    Fields typically come from PIL headers and/or a quick decode pass.
    """
    width: int
    height: int
    channels: int = 3
    dtype: str = "uint8"
    file_size: int = 0
    format: str = "unknown"
    dpi: Optional[int] = None
    color_space: Optional[str] = None  # e.g., "RGB", "grayscale", "CMYK"

    # derived
    @property
    def shape(self) -> Tuple[int, int, int]:
        return (int(self.height), int(self.width), int(self.channels))

    @property
    def size(self) -> Tuple[int, int]:
        return (int(self.width), int(self.height))

    @property
    def megapixels(self) -> float:
        return round((self.width * self.height) / 1_000_000.0, 3)

    @property
    def aspect_ratio(self) -> float:
        return (self.width / float(self.height)) if self.height else 0.0
