from __future__ import annotations

import cv2
import numpy as np

from utils.image.layout.analyzer import LayoutAnalyzer
from utils.image.layout.options import RegionDetectionOptions
from utils.image.layout.post.merge import consolidate_regions
from utils.image.regions.core_types import ImageRegion


def _region(x: int, y: int, width: int, height: int) -> ImageRegion:
    return ImageRegion(x=x, y=y, width=width, height=height, region_type="diagram", confidence=0.85)


def test_overlapping_candidates_merge_to_one_region():
    regions = [_region(20, 30, 160, 120), _region(60, 60, 160, 120)]

    consolidated, events = consolidate_regions(regions, RegionDetectionOptions())

    assert len(consolidated) == 1
    assert consolidated[0].bbox == (20, 30, 220, 180)
    assert any(event["reason"] == "overlap_merge" for event in events)


def test_contained_child_is_suppressed_when_parent_is_reasonable():
    regions = [_region(40, 40, 240, 180), _region(90, 80, 150, 120)]

    consolidated, events = consolidate_regions(regions, RegionDetectionOptions())

    assert len(consolidated) == 1
    assert consolidated[0].bbox == (40, 40, 280, 220)
    assert any(event["reason"] == "contained_suppression" for event in events)


def test_contained_child_is_preserved_when_parent_is_too_broad():
    options = RegionDetectionOptions(contained_parent_max_area_ratio=2.0)
    regions = [_region(0, 0, 500, 400), _region(180, 140, 80, 70)]

    consolidated, events = consolidate_regions(regions, options)

    assert len(consolidated) == 2
    assert events == []


def test_adjacent_parts_of_one_diagram_merge_when_growth_is_small():
    regions = [_region(20, 80, 180, 160), _region(214, 70, 170, 170)]

    consolidated, events = consolidate_regions(regions, RegionDetectionOptions(adjacent_merge_gap_px=24))

    assert len(consolidated) == 1
    assert consolidated[0].bbox == (20, 70, 384, 240)
    assert any(event["reason"] == "adjacent_merge" for event in events)


def test_separate_nearby_diagrams_do_not_merge_when_area_growth_is_large():
    regions = [_region(20, 30, 120, 90), _region(162, 260, 120, 90)]

    consolidated, events = consolidate_regions(
        regions,
        RegionDetectionOptions(adjacent_merge_gap_px=200, adjacent_merge_max_area_growth_ratio=1.3),
    )

    assert len(consolidated) == 2
    assert events == []


def test_text_like_candidate_does_not_suppress_real_diagram_after_filtering():
    page = np.full((500, 760), 255, dtype=np.uint8)
    cv2.putText(page, "TYPEWRITTEN BODY TEXT", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 3, cv2.LINE_AA)
    cv2.putText(page, "MORE BODY TEXT", (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 3, cv2.LINE_AA)
    cv2.rectangle(page, (330, 130), (690, 380), 0, 4)
    cv2.circle(page, (500, 250), 75, 0, 4)
    cv2.line(page, (355, 355), (665, 145), 0, 3)

    analyzer = LayoutAnalyzer(
        {
            "enabled_detectors": ["contours"],
            "contours_always": True,
            "filter_text_like": True,
            "contour_min_area": 1200,
            "contour_max_area_ratio": 0.7,
            "contour_left_bias_xmax": 1.0,
            "contour_topk": 8,
            "region_text_like_min_components": 8,
        }
    )

    diagnostics = analyzer.detect_image_regions_with_diagnostics(page)

    assert any(region.x > 300 and region.y > 100 for region in diagnostics["accepted_regions"])
    assert not any(region.x < 250 and region.area > 20_000 for region in diagnostics["accepted_regions"])
