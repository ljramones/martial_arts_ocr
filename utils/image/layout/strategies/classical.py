from __future__ import annotations

from typing import Any

import numpy as np

from utils.image.layout.analyzer import LayoutAnalyzer
from utils.image.layout.strategy import LayoutDetectionResult


class ClassicalLayoutStrategy:
    """Current OpenCV/heuristic image-region detector as a strategy."""

    name = "classical_opencv"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.analyzer = LayoutAnalyzer(self.config)

    @classmethod
    def is_available(cls) -> bool:
        return True

    def detect(self, image: np.ndarray) -> LayoutDetectionResult:
        diagnostics = self.analyzer.detect_image_regions_with_diagnostics(image)
        return LayoutDetectionResult(
            strategy_name=self.name,
            regions=list(diagnostics.get("accepted_regions", [])),
            available=True,
            metadata={
                "accepted": diagnostics.get("accepted", []),
                "rejected": diagnostics.get("rejected", []),
                "consolidation": diagnostics.get("consolidation", []),
                "config": self.config,
            },
        )
