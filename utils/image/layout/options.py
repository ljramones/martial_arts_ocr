from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RegionDetectionOptions:
    """Configurable post-detection filters for image/diagram candidates."""

    min_area: int = 500
    min_width: int = 20
    min_height: int = 20
    max_text_like_density: float = 0.65
    min_diagram_aspect_ratio: float = 0.15
    max_diagram_aspect_ratio: float = 8.0
    reject_text_like: bool = True
    reject_rotated_text_like: bool = True
    text_like_min_components: int = 24
    text_like_min_density: float = 0.14
    text_like_max_density: float = 0.35
    text_like_min_median_component_area: float = 60.0
    text_like_max_median_component_area: float = 260.0
    text_like_max_small_component_fraction: float = 0.45
    title_text_max_components: int = 24
    title_text_max_row_occupancy: float = 0.55
    title_text_min_col_occupancy: float = 0.62
    text_line_max_height: int = 90
    text_line_min_aspect_ratio: float = 2.0
    text_line_min_density: float = 0.20
    text_line_max_density: float = 0.50
    text_line_min_col_occupancy: float = 0.75
    vertical_text_max_aspect_ratio: float = 0.45
    rotated_text_min_row_occupancy: float = 0.80
    rotated_text_min_col_occupancy: float = 0.75

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RegionDetectionOptions":
        """Build options from the existing layout config dictionary."""
        field_names = cls.__dataclass_fields__
        values = {
            name: cfg[f"region_{name}"]
            for name in field_names
            if f"region_{name}" in cfg
        }
        return cls(**values)
