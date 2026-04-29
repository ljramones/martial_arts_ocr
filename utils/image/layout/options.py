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
    title_text_max_components: int = 40
    title_text_max_row_occupancy: float = 0.55
    title_text_min_col_occupancy: float = 0.62
    text_line_max_height: int = 90
    text_line_min_aspect_ratio: float = 2.0
    text_line_min_density: float = 0.20
    text_line_max_density: float = 0.50
    text_line_min_col_occupancy: float = 0.75
    sparse_text_band_max_height: int = 180
    sparse_text_band_min_aspect_ratio: float = 3.0
    sparse_text_band_min_density: float = 0.12
    sparse_text_band_max_density: float = 0.35
    sparse_text_band_max_median_component_area: float = 70.0
    sparse_text_band_min_small_component_fraction: float = 0.55
    vertical_text_max_aspect_ratio: float = 0.45
    rotated_text_min_row_occupancy: float = 0.80
    rotated_text_min_col_occupancy: float = 0.75
    preserve_labeled_diagrams: bool = True
    labeled_diagram_min_component_area_ratio: float = 2.4
    labeled_diagram_min_small_component_fraction: float = 0.30
    labeled_diagram_max_density: float = 0.35
    merge_overlapping_regions: bool = True
    merge_adjacent_regions: bool = True
    overlap_merge_iou_threshold: float = 0.35
    contained_region_suppression_threshold: float = 0.85
    contained_parent_max_area_ratio: float = 5.0
    adjacent_merge_gap_px: int = 24
    adjacent_merge_max_area_growth_ratio: float = 1.75
    adjacent_merge_min_axis_overlap_ratio: float = 0.25
    text_score_reject_threshold: float = 0.72
    visual_score_override_threshold: float = 0.58
    broad_crop_area_ratio: float = 0.25
    broad_crop_visual_override_threshold: float = 0.78
    photo_like_min_std: float = 55.0
    photo_like_min_dark_fraction: float = 0.10
    photo_like_min_edge_density: float = 0.16
    visual_min_dimension_for_photo: int = 120
    enable_ocr_text_suppression: bool = True
    ocr_high_overlap_threshold: float = 0.60
    ocr_moderate_overlap_threshold: float = 0.25
    ocr_low_overlap_threshold: float = 0.10
    ocr_rescue_figure_score_threshold: float = 0.70
    ocr_rescue_photo_score_threshold: float = 0.70
    ocr_rescue_sparse_symbol_score_threshold: float = 0.65
    ocr_text_mask_dilation_px: int = 4
    enable_mixed_region_refinement: bool = False
    mixed_region_min_ocr_overlap: float = 0.25

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
