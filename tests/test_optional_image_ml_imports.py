def test_yolo_detector_import_does_not_require_model_downloads():
    from utils.image.layout.detectors.yolo_figure import YOLOFigureDetector

    assert isinstance(YOLOFigureDetector.available, bool)


def test_layout_analyzer_falls_back_when_yolo_requested_without_model_downloads(monkeypatch):
    from utils.image.layout.detectors.yolo_figure import YOLOFigureDetector
    from utils.image.layout.analyzer import LayoutAnalyzer

    monkeypatch.setattr(YOLOFigureDetector, "available", False)
    analyzer = LayoutAnalyzer({"use_yolo_figure": True, "enabled_detectors": ["figure"]})

    assert analyzer.figure is not None
