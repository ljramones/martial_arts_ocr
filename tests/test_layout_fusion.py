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
