"""
Core image data structures and basic operations.

This module contains the fundamental data classes used throughout the image
processing pipeline for the Martial Arts OCR system.
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional


@dataclass
class ImageRegion:
    """
    Represents a rectangular region in an image.

    Used for identifying and tracking text blocks, diagrams, photos, and other
    content areas within document images.

    Attributes:
        x: Left coordinate of the region
        y: Top coordinate of the region
        width: Width of the region in pixels
        height: Height of the region in pixels
        confidence: Confidence score for the region detection (0.0-1.0)
        region_type: Type of content ("text", "image", "diagram", etc.)
    """
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    region_type: str = "unknown"  # "text", "image", "diagram", "photo", etc.

    @property
    def area(self) -> int:
        """Calculate the area of the region in pixels."""
        return self.width * self.height

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """
        Return bounding box as (x1, y1, x2, y2).

        Returns:
            Tuple of (left, top, right, bottom) coordinates
        """
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def center(self) -> Tuple[int, int]:
        """
        Return center point of the region.

        Returns:
            Tuple of (center_x, center_y) coordinates
        """
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def aspect_ratio(self) -> float:
        """
        Calculate aspect ratio (width/height) of the region.

        Returns:
            Aspect ratio, or 0.0 if height is 0
        """
        return (self.width / self.height) if self.height > 0 else 0.0

    def contains_point(self, x: int, y: int) -> bool:
        """
        Check if a point is inside this region.

        Args:
            x: X coordinate of the point
            y: Y coordinate of the point

        Returns:
            True if the point is inside the region
        """
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)

    def intersects(self, other: 'ImageRegion') -> bool:
        """
        Check if this region intersects with another region.

        Args:
            other: Another ImageRegion to check intersection with

        Returns:
            True if the regions intersect
        """
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other.bbox

        return not (x1_max <= x2_min or x2_max <= x1_min or
                   y1_max <= y2_min or y2_max <= y1_min)

    def intersection_area(self, other: 'ImageRegion') -> int:
        """
        Calculate the intersection area with another region.

        Args:
            other: Another ImageRegion

        Returns:
            Area of intersection in pixels
        """
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other.bbox

        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

        return x_overlap * y_overlap

    def union_area(self, other: 'ImageRegion') -> int:
        """
        Calculate the union area with another region.

        Args:
            other: Another ImageRegion

        Returns:
            Area of union in pixels
        """
        intersection = self.intersection_area(other)
        return self.area + other.area - intersection

    def iou(self, other: 'ImageRegion') -> float:
        """
        Calculate Intersection over Union (IoU) with another region.

        Args:
            other: Another ImageRegion

        Returns:
            IoU score (0.0-1.0)
        """
        union = self.union_area(other)
        return self.intersection_area(other) / union if union > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert region to dictionary representation.

        Returns:
            Dictionary containing all region attributes
        """
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence,
            'type': self.region_type,
            'area': self.area,
            'aspect_ratio': self.aspect_ratio,
            'center': self.center
        }

    @classmethod
    def from_bbox(cls, bbox: Tuple[int, int, int, int],
                  confidence: float = 0.0,
                  region_type: str = "unknown") -> 'ImageRegion':
        """
        Create an ImageRegion from a bounding box.

        Args:
            bbox: Tuple of (x1, y1, x2, y2)
            confidence: Confidence score
            region_type: Type of region

        Returns:
            New ImageRegion instance
        """
        x1, y1, x2, y2 = bbox
        return cls(
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            confidence=confidence,
            region_type=region_type
        )


@dataclass
class ImageInfo:
    """
    Information about an image file.

    Stores metadata and properties of an image for processing decisions.

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        channels: Number of color channels (1 for grayscale, 3 for RGB/BGR)
        dtype: Data type of pixel values (e.g., "uint8")
        file_size: Size of the image file in bytes
        format: Image format (e.g., "JPEG", "PNG")
    """
    width: int
    height: int
    channels: int
    dtype: str
    file_size: int
    format: str
    dpi: Optional[int] = None
    color_space: Optional[str] = None

    orientation: Optional[int] = None     # 0|90|180|270 (after CNN + tie-break)
    skew_angle: Optional[float] = None    # small-angle deskew (deg, +CCW)
    extras: Optional[Dict[str, Any]] = None   # free-form stash (e.g., debug, margins)


    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height) of the image."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def total_pixels(self) -> int:
        """Calculate total number of pixels in the image."""
        return self.width * self.height

    @property
    def is_grayscale(self) -> bool:
        """Check if the image is grayscale."""
        return self.channels == 1

    @property
    def is_color(self) -> bool:
        """Check if the image is color."""
        return self.channels >= 3

    @property
    def megapixels(self) -> float:
        """Calculate megapixels of the image."""
        return self.total_pixels / 1_000_000

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert image info to dictionary representation.

        Returns:
            Dictionary containing all image info attributes
        """
        return {
            'width': self.width,
            'height': self.height,
            'channels': self.channels,
            'dtype': self.dtype,
            'file_size': self.file_size,
            'format': self.format,
            'dpi': self.dpi,
            'color_space': self.color_space,
            'aspect_ratio': self.aspect_ratio,
            'total_pixels': self.total_pixels,
            'megapixels': self.megapixels,
            'is_grayscale': self.is_grayscale,
            'is_color': self.is_color
        }


@dataclass
class _Box:
    """
    Internal box representation for region operations.

    A simplified box class used internally for geometric calculations
    and region merging operations.

    Attributes:
        x: Left coordinate
        y: Top coordinate
        w: Width
        h: Height
    """
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        """Right coordinate of the box."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Bottom coordinate of the box."""
        return self.y + self.h

    @property
    def area(self) -> int:
        """Area of the box in pixels."""
        return self.w * self.h

    @property
    def center(self) -> Tuple[int, int]:
        """Center point of the box."""
        return (self.x + self.w // 2, self.y + self.h // 2)

    def to_region(self, confidence: float = 0.0,
                  region_type: str = "unknown") -> ImageRegion:
        """
        Convert box to ImageRegion.

        Args:
            confidence: Confidence score for the region
            region_type: Type of region

        Returns:
            ImageRegion instance
        """
        return ImageRegion(
            x=self.x,
            y=self.y,
            width=self.w,
            height=self.h,
            confidence=confidence,
            region_type=region_type
        )

    @classmethod
    def from_region(cls, region: ImageRegion) -> '_Box':
        """
        Create a box from an ImageRegion.

        Args:
            region: ImageRegion to convert

        Returns:
            _Box instance
        """
        return cls(
            x=region.x,
            y=region.y,
            w=region.width,
            h=region.height
        )


# Type aliases for clarity
BoundingBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
Point = Tuple[int, int]  # (x, y)
Size = Tuple[int, int]  # (width, height)