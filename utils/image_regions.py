"""
Utilities for manipulating image regions.

This module provides functions for merging, splitting, and manipulating
image regions, particularly focused on grouping small text regions into
lines and cleaning up OCR output.
"""
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .core_image import ImageRegion, _Box

logger = logging.getLogger(__name__)


def merge_regions_into_lines(regions: List[ImageRegion],
                             max_gap_px: int = 28,
                             min_y_overlap: float = 0.45,
                             merge_vertically_aligned: bool = True) -> List[ImageRegion]:
    """
    Merge small text regions into line-level regions for better OCR.

    Uses a greedy left-to-right merging strategy to combine small regions
    (typically individual words or characters) into complete text lines.
    This significantly improves OCR accuracy on typewritten documents.

    Args:
        regions: List of input regions to merge
        max_gap_px: Maximum horizontal gap between regions to merge (pixels)
        min_y_overlap: Minimum vertical overlap ratio for merging (0.0-1.0)
        merge_vertically_aligned: Whether to merge vertically aligned regions

    Returns:
        List of merged line-level regions
    """
    if not regions:
        return []

    # Sort by y-coordinate (top to bottom), then x-coordinate (left to right)
    sorted_regions = sorted(regions, key=lambda r: (r.y, r.x))

    # Group regions into lines
    lines: List[List[ImageRegion]] = []

    for region in sorted_regions:
        placed = False

        # Try to add to existing line
        for line in lines:
            if _can_merge_to_line(region, line, max_gap_px, min_y_overlap):
                line.append(region)
                placed = True
                break

        # Start new line if not placed
        if not placed:
            lines.append([region])

    # Merge each line into a single region
    merged_regions = []
    for line in lines:
        if merge_vertically_aligned and len(line) > 1:
            # Sort line by x-coordinate for proper ordering
            line = sorted(line, key=lambda r: r.x)

        # Calculate bounding box for entire line
        min_x = min(r.x for r in line)
        min_y = min(r.y for r in line)
        max_x = max(r.x + r.width for r in line)
        max_y = max(r.y + r.height for r in line)

        # Determine confidence and type
        avg_confidence = np.mean([r.confidence for r in line])
        most_common_type = _most_common(r.region_type for r in line)

        merged_region = ImageRegion(
            x=min_x, y=min_y,
            width=max_x - min_x, height=max_y - min_y,
            confidence=avg_confidence,
            region_type=most_common_type
        )

        merged_regions.append(merged_region)

    logger.debug(f"Merged {len(regions)} regions into {len(merged_regions)} lines")
    return merged_regions


def _can_merge_to_line(region: ImageRegion, line: List[ImageRegion],
                       max_gap_px: int, min_y_overlap: float) -> bool:
    """
    Check if a region can be merged into an existing line.

    Args:
        region: Region to check
        line: Existing line of regions
        max_gap_px: Maximum horizontal gap allowed
        min_y_overlap: Minimum vertical overlap required

    Returns:
        True if region can be merged into the line
    """
    if not line:
        return True

    # Check against the rightmost region in the line
    last_region = max(line, key=lambda r: r.x + r.width)

    # Calculate vertical overlap
    y_overlap = _calculate_y_overlap_ratio(region, last_region)
    if y_overlap < min_y_overlap:
        return False

    # Calculate horizontal gap
    gap = region.x - (last_region.x + last_region.width)
    if gap < 0:  # Region is to the left of the last region
        # Check if it's to the left of all regions
        leftmost = min(line, key=lambda r: r.x)
        gap = leftmost.x - (region.x + region.width)

    return gap <= max_gap_px


def _calculate_y_overlap_ratio(region1: ImageRegion, region2: ImageRegion) -> float:
    """
    Calculate the vertical overlap ratio between two regions.

    Args:
        region1: First region
        region2: Second region

    Returns:
        Overlap ratio (0.0 = no overlap, 1.0 = complete overlap)
    """
    y1_min = region1.y
    y1_max = region1.y + region1.height
    y2_min = region2.y
    y2_max = region2.y + region2.height

    overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    min_height = min(region1.height, region2.height)

    return overlap / max(min_height, 1)


def split_region_into_columns(region: ImageRegion, num_columns: int = 2,
                              gap_width: Optional[int] = None) -> List[ImageRegion]:
    """
    Split a region into multiple columns.

    Useful for processing multi-column documents.

    Args:
        region: Region to split
        num_columns: Number of columns to create
        gap_width: Width of gap between columns (pixels)

    Returns:
        List of column regions
    """
    if num_columns <= 1:
        return [region]

    column_width = region.width // num_columns
    if gap_width is None:
        gap_width = max(10, column_width // 20)  # 5% of column width

    columns = []
    for i in range(num_columns):
        x = region.x + i * column_width
        width = column_width - gap_width if i < num_columns - 1 else region.width - (i * column_width)

        column = ImageRegion(
            x=x, y=region.y,
            width=width, height=region.height,
            confidence=region.confidence,
            region_type=region.region_type
        )
        columns.append(column)

    return columns


def group_regions_by_proximity(regions: List[ImageRegion],
                               max_distance: float = 50.0) -> List[List[ImageRegion]]:
    """
    Group regions by spatial proximity.

    Uses distance-based clustering to group nearby regions.

    Args:
        regions: List of regions to group
        max_distance: Maximum distance between regions in same group

    Returns:
        List of region groups
    """
    if not regions:
        return []

    groups: List[List[ImageRegion]] = []
    remaining = regions.copy()

    while remaining:
        # Start new group with first remaining region
        current = remaining.pop(0)
        group = [current]

        # Find all regions close to this group
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(remaining):
                if _is_close_to_group(remaining[i], group, max_distance):
                    group.append(remaining.pop(i))
                    changed = True
                else:
                    i += 1

        groups.append(group)

    return groups


def _is_close_to_group(region: ImageRegion, group: List[ImageRegion],
                       max_distance: float) -> bool:
    """
    Check if a region is close to any region in a group.

    Args:
        region: Region to check
        group: Group of regions
        max_distance: Maximum allowed distance

    Returns:
        True if region is close to the group
    """
    for member in group:
        distance = _calculate_region_distance(region, member)
        if distance <= max_distance:
            return True
    return False


def _calculate_region_distance(region1: ImageRegion, region2: ImageRegion) -> float:
    """
    Calculate the minimum distance between two regions.

    Args:
        region1: First region
        region2: Second region

    Returns:
        Minimum distance between regions
    """
    # Get bounding boxes
    x1_min, y1_min, x1_max, y1_max = region1.bbox
    x2_min, y2_min, x2_max, y2_max = region2.bbox

    # Calculate horizontal distance
    if x1_max < x2_min:
        h_dist = x2_min - x1_max
    elif x2_max < x1_min:
        h_dist = x1_min - x2_max
    else:
        h_dist = 0

    # Calculate vertical distance
    if y1_max < y2_min:
        v_dist = y2_min - y1_max
    elif y2_max < y1_min:
        v_dist = y1_min - y2_max
    else:
        v_dist = 0

    # Return Euclidean distance
    return np.sqrt(h_dist ** 2 + v_dist ** 2)


def sort_regions_reading_order(regions: List[ImageRegion],
                               tolerance: int = 20) -> List[ImageRegion]:
    """
    Sort regions in natural reading order (top-to-bottom, left-to-right).

    Args:
        regions: List of regions to sort
        tolerance: Vertical tolerance for considering regions on same line

    Returns:
        Sorted list of regions
    """
    if not regions:
        return []

    # Group regions by approximate y-position (lines)
    lines: Dict[int, List[ImageRegion]] = {}

    for region in regions:
        y_key = region.y // tolerance
        if y_key not in lines:
            lines[y_key] = []
        lines[y_key].append(region)

    # Sort each line by x-position, then combine
    sorted_regions = []
    for y_key in sorted(lines.keys()):
        line = sorted(lines[y_key], key=lambda r: r.x)
        sorted_regions.extend(line)

    return sorted_regions


def expand_region(region: ImageRegion, expansion: int = 5) -> ImageRegion:
    """
    Expand a region by a fixed number of pixels in all directions.

    Args:
        region: Region to expand
        expansion: Number of pixels to expand

    Returns:
        Expanded region
    """
    return ImageRegion(
        x=max(0, region.x - expansion),
        y=max(0, region.y - expansion),
        width=region.width + 2 * expansion,
        height=region.height + 2 * expansion,
        confidence=region.confidence,
        region_type=region.region_type
    )


def shrink_region(region: ImageRegion, shrinkage: int = 5) -> ImageRegion:
    """
    Shrink a region by a fixed number of pixels in all directions.

    Args:
        region: Region to shrink
        shrinkage: Number of pixels to shrink

    Returns:
        Shrunken region
    """
    new_width = max(1, region.width - 2 * shrinkage)
    new_height = max(1, region.height - 2 * shrinkage)

    return ImageRegion(
        x=region.x + shrinkage,
        y=region.y + shrinkage,
        width=new_width,
        height=new_height,
        confidence=region.confidence,
        region_type=region.region_type
    )


def filter_regions_by_size(regions: List[ImageRegion],
                           min_area: Optional[int] = None,
                           max_area: Optional[int] = None,
                           min_width: Optional[int] = None,
                           max_width: Optional[int] = None,
                           min_height: Optional[int] = None,
                           max_height: Optional[int] = None) -> List[ImageRegion]:
    """
    Filter regions based on size constraints.

    Args:
        regions: List of regions to filter
        min_area: Minimum area (pixels²)
        max_area: Maximum area (pixels²)
        min_width: Minimum width (pixels)
        max_width: Maximum width (pixels)
        min_height: Minimum height (pixels)
        max_height: Maximum height (pixels)

    Returns:
        Filtered list of regions
    """
    filtered = []

    for region in regions:
        # Check area constraints
        if min_area is not None and region.area < min_area:
            continue
        if max_area is not None and region.area > max_area:
            continue

        # Check width constraints
        if min_width is not None and region.width < min_width:
            continue
        if max_width is not None and region.width > max_width:
            continue

        # Check height constraints
        if min_height is not None and region.height < min_height:
            continue
        if max_height is not None and region.height > max_height:
            continue

        filtered.append(region)

    return filtered


def post_ocr_fixups(text: str,
                    fix_hyphens: bool = True,
                    merge_lines: bool = True,
                    remove_duplicates: bool = True) -> str:
    """
    Apply post-processing fixes to OCR output text.

    Performs deterministic cleanups to improve text quality:
    - Joins hyphenated line breaks
    - Merges soft-wrapped lines
    - Removes duplicate lines
    - Collapses excessive blank lines

    Args:
        text: Input text from OCR
        fix_hyphens: Whether to fix hyphenated line breaks
        merge_lines: Whether to merge soft-wrapped lines
        remove_duplicates: Whether to remove duplicate lines

    Returns:
        Cleaned text
    """
    if not text:
        return text

    lines = [line.rstrip() for line in text.splitlines()]

    # Step 1: Fix hyphenated line breaks
    if fix_hyphens:
        lines = _fix_hyphenated_breaks(lines)

    # Step 2: Merge soft-wrapped lines
    if merge_lines:
        lines = _merge_soft_wrapped_lines(lines)

    # Step 3: Remove duplicate consecutive lines
    if remove_duplicates:
        lines = _remove_duplicate_lines(lines)

    # Step 4: Collapse excessive blank lines
    lines = _collapse_blank_lines(lines)

    return "\n".join(lines).strip()


def _fix_hyphenated_breaks(lines: List[str]) -> List[str]:
    """
    Join hyphenated line breaks.

    Converts 'word-\\n' + 'continuation' to 'wordcontinuation'.

    Args:
        lines: Input lines

    Returns:
        Lines with hyphenation fixed
    """
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if line ends with hyphen and next line exists
        if line.endswith('-') and i + 1 < len(lines):
            next_line = lines[i + 1].lstrip()

            # Check if next line starts with lowercase or digit (continuation)
            if next_line and (next_line[0].islower() or next_line[0].isdigit()):
                # Join the words
                result.append(line[:-1] + next_line)
                i += 2  # Skip next line
                continue

        result.append(line)
        i += 1

    return result


def _merge_soft_wrapped_lines(lines: List[str]) -> List[str]:
    """
    Merge lines that appear to be soft-wrapped.

    Args:
        lines: Input lines

    Returns:
        Lines with soft wraps merged
    """
    if not lines:
        return lines

    merged = [lines[0]]

    for line in lines[1:]:
        # Check if current line looks like continuation of previous
        if line and merged:
            prev = merged[-1]
            # Line starts with lowercase and previous doesn't end with sentence terminator
            if (line[0].islower() and
                    prev and
                    not prev.endswith(('.', '!', '?', ':', ';', '"', "'"))):
                # Merge with previous line
                merged[-1] = (prev + ' ' + line).strip()
            else:
                merged.append(line)
        else:
            merged.append(line)

    return merged


def _remove_duplicate_lines(lines: List[str]) -> List[str]:
    """
    Remove consecutive duplicate lines.

    Args:
        lines: Input lines

    Returns:
        Lines with duplicates removed
    """
    if not lines:
        return lines

    result = [lines[0]]

    for line in lines[1:]:
        # Only add if different from previous
        if line != result[-1]:
            result.append(line)

    return result


def _collapse_blank_lines(lines: List[str]) -> List[str]:
    """
    Collapse multiple consecutive blank lines into single blank line.

    Args:
        lines: Input lines

    Returns:
        Lines with collapsed blanks
    """
    result = []
    prev_blank = False

    for line in lines:
        is_blank = line.strip() == ""

        # Only add blank if previous wasn't blank
        if is_blank:
            if not prev_blank:
                result.append(line)
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False

    return result


def _most_common(iterable):
    """
    Find the most common element in an iterable.

    Args:
        iterable: Input iterable

    Returns:
        Most common element
    """
    from collections import Counter
    counter = Counter(iterable)
    if counter:
        return counter.most_common(1)[0][0]
    return None


# Conversion utilities for legacy code compatibility

def box_to_region(box: _Box, confidence: float = 0.0,
                  region_type: str = "unknown") -> ImageRegion:
    """
    Convert a _Box to an ImageRegion.

    Args:
        box: Box to convert
        confidence: Confidence score for the region
        region_type: Type of region

    Returns:
        ImageRegion instance
    """
    return box.to_region(confidence, region_type)


def region_to_box(region: ImageRegion) -> _Box:
    """
    Convert an ImageRegion to a _Box.

    Args:
        region: Region to convert

    Returns:
        _Box instance
    """
    return _Box.from_region(region)
