# DEPRECATED: this module has been split.
import warnings
warnings.warn(
    "utils.image_regions is deprecated; use modules under utils.image.regions: "
    "convert, geometry, grouping, layout, filters, text_fixups (or the package facade).",
    DeprecationWarning,
    stacklevel=2,
)

# Minimal, explicit re-exports to maintain old names:
from utils.image.regions.core_types import ImageRegion, ImageInfo, _Box

from utils.image.regions.convert import box_to_region, region_to_box, bbox_to_region
from utils.image.regions.geometry import expand_region, shrink_region  # and any others you used to export
from utils.image.regions.grouping import merge_regions_into_lines, group_regions_by_proximity, lines_to_regions
from utils.image.regions.layout import split_regions_into_columns, sort_regions_reading_order
from utils.image.regions.filters import filter_regions_by_size
from utils.image.regions.text_fixups import post_ocr_fixups

__all__ = [
    "ImageRegion", "ImageInfo", "_Box",
    "box_to_region", "region_to_box", "bbox_to_region",
    "expand_region", "shrink_region",
    "merge_regions_into_lines", "group_regions_by_proximity", "lines_to_regions",
    "split_regions_into_columns", "sort_regions_reading_order",
    "filter_regions_by_size",
    "post_ocr_fixups",
]
