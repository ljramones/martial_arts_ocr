from __future__ import annotations

from utils.image.layout.fusion import fuse_paddle_layout_regions
from utils.image.regions.core_types import ImageRegion


def _mixed_region(x=0, y=0, width=500, height=500):
    return ImageRegion(
        x=x,
        y=y,
        width=width,
        height=height,
        region_type="diagram",
        confidence=0.7,
        id="classical_mixed",
        metadata={"mixed_region": True, "needs_review": True},
    )


def _paddle_region(x, y, width, height, *, label="image", confidence=0.92):
    return ImageRegion(
        x=x,
        y=y,
        width=width,
        height=height,
        region_type="figure" if label != "table" else "table",
        confidence=confidence,
        metadata={"layout_label": label, "raw_label": label, "layout_backend": "paddle_ppstructure"},
    )


def test_containment_not_iou_drives_paddle_refinement():
    classical = _mixed_region()
    paddle = _paddle_region(100, 100, 200, 200)

    result = fuse_paddle_layout_regions([classical], [paddle])

    fused = result.regions[0]
    assert fused.bbox == paddle.bbox
    assert fused.metadata["layout_fusion_applied"] is True
    assert fused.metadata["iou"] < 0.3
    assert fused.metadata["paddle_inside_classical_ratio"] == 1.0
    assert fused.metadata["area_tightness_ratio"] == 0.16


def test_tiny_paddle_micro_crop_does_not_replace_mixed_parent():
    classical = _mixed_region()
    tiny = _paddle_region(100, 100, 40, 40)

    result = fuse_paddle_layout_regions([classical], [tiny])

    assert result.regions[0].bbox == classical.bbox
    assert result.regions[0].metadata["layout_fusion_applied"] is False
    assert result.events[0]["fusion_reason"] == "no_qualifying_paddle_visual_region"


def test_clean_classical_region_stays_unchanged():
    classical = ImageRegion(x=0, y=0, width=500, height=500, region_type="diagram", metadata={})
    paddle = _paddle_region(100, 100, 200, 200)

    result = fuse_paddle_layout_regions([classical], [paddle])

    assert result.regions == [classical]
    assert result.events == []


def test_paddle_text_and_table_regions_do_not_become_image_regions():
    classical = _mixed_region()
    paddle_text = _paddle_region(100, 100, 200, 200, label="text")
    paddle_table = _paddle_region(100, 100, 200, 200, label="table")

    result = fuse_paddle_layout_regions([classical], [paddle_text, paddle_table])

    assert result.regions[0].bbox == classical.bbox
    assert result.regions[0].metadata["layout_fusion_applied"] is False


def test_best_single_paddle_region_selected_when_multiple_qualify():
    classical = _mixed_region()
    first = _paddle_region(50, 50, 250, 250, confidence=0.7)
    second = _paddle_region(100, 100, 200, 200, confidence=0.95)

    result = fuse_paddle_layout_regions([classical], [first, second])

    fused = result.regions[0]
    assert fused.bbox == second.bbox
    assert fused.metadata["multiple_paddle_candidates"] is True
    assert fused.metadata["num_paddle_candidates"] == 2
    assert fused.metadata["fusion_limitation"] == "v1_best_single_region_only"


def test_paddle_visual_partly_outside_mixed_parent_is_added_separately():
    classical = _mixed_region(x=100, y=100, width=300, height=240)
    paddle = _paddle_region(320, 180, 160, 180, confidence=0.91)

    result = fuse_paddle_layout_regions([classical], [paddle])

    assert len(result.regions) == 2
    assert result.regions[0].bbox == classical.bbox
    added = result.regions[1]
    assert added.bbox == paddle.bbox
    assert added.metadata["fusion_mode"] == "paddle_additive"
    assert added.metadata["relation_reason"] == "partial_overlap"
    assert added.metadata["horizontal_span_overlap_ratio"] >= 0.5


def test_larger_paddle_visual_near_partial_mixed_parent_is_added_not_replacement():
    classical = _mixed_region(x=100, y=100, width=120, height=120)
    paddle = _paddle_region(80, 80, 360, 300, confidence=0.94)

    result = fuse_paddle_layout_regions([classical], [paddle])

    assert len(result.regions) == 2
    assert result.regions[0].bbox == classical.bbox
    added = result.regions[1]
    assert added.bbox == paddle.bbox
    assert added.metadata["fusion_mode"] == "paddle_additive"
    assert added.metadata["area_tightness_ratio"] > 1.0
    assert added.metadata["needs_review"] is True
    assert added.metadata["region_role"] == "paddle_added_mixed_or_uncertain"


def test_unrelated_paddle_visual_region_is_not_added():
    classical = _mixed_region(x=100, y=100, width=120, height=120)
    paddle = _paddle_region(600, 600, 160, 160, confidence=0.95)

    result = fuse_paddle_layout_regions([classical], [paddle])

    assert result.regions[0].bbox == classical.bbox
    assert len(result.regions) == 1


def test_paddle_duplicate_of_clean_classical_region_is_not_added():
    clean = ImageRegion(x=300, y=100, width=180, height=180, region_type="diagram", metadata={})
    mixed = _mixed_region(x=120, y=100, width=120, height=180)
    duplicate = _paddle_region(310, 110, 150, 150, confidence=0.96)

    result = fuse_paddle_layout_regions([clean, mixed], [duplicate])

    assert result.regions == [clean, result.regions[1]]
    assert len(result.regions) == 2
    assert result.regions[1].bbox == mixed.bbox


def test_paddle_text_and_table_regions_are_not_added():
    classical = _mixed_region(x=100, y=100, width=120, height=120)
    paddle_text = _paddle_region(110, 110, 140, 140, label="text", confidence=0.97)
    paddle_table = _paddle_region(110, 110, 140, 140, label="table", confidence=0.97)

    result = fuse_paddle_layout_regions([classical], [paddle_text, paddle_table])

    assert len(result.regions) == 1
    assert result.regions[0].bbox == classical.bbox


def test_shared_span_relation_uses_numeric_threshold():
    classical = _mixed_region(x=100, y=100, width=120, height=120)
    paddle = _paddle_region(250, 130, 140, 60, confidence=0.95)

    result = fuse_paddle_layout_regions([classical], [paddle])

    assert len(result.regions) == 2
    added = result.regions[1]
    assert added.metadata["relation_reason"] == "shared_vertical_span"
    assert added.metadata["vertical_span_overlap_ratio"] >= 0.5
    assert added.metadata["horizontal_span_overlap_ratio"] == 0.0


def test_additive_metadata_is_present():
    classical = _mixed_region(x=100, y=100, width=120, height=120)
    paddle = _paddle_region(190, 130, 80, 80, confidence=0.95)

    result = fuse_paddle_layout_regions([classical], [paddle])

    metadata = result.regions[1].metadata
    assert metadata["layout_fusion_applied"] is True
    assert metadata["layout_backend"] == "paddle_ppstructure_v3"
    assert metadata["fusion_mode"] == "paddle_additive"
    assert metadata["layout_source_bbox"] == [190, 130, 80, 80]
    assert metadata["related_classical_bbox"] == [100, 100, 120, 120]
    assert metadata["paddle_confidence"] == 0.95
    assert metadata["paddle_label"] == "image"
    assert metadata["region_role"] == "paddle_added_visual"
    assert metadata["needs_review"] is False
