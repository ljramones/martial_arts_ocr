from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from utils.image.regions.core_types import ImageRegion


@dataclass(frozen=True)
class LayoutDetectionResult:
    """Normalized result from a layout detection strategy."""

    strategy_name: str
    regions: list[ImageRegion] = field(default_factory=list)
    available: bool = True
    skipped_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "available": self.available,
            "skipped_reason": self.skipped_reason,
            "regions": [region.to_dict() for region in self.regions],
            "metadata": dict(self.metadata),
        }


class LayoutDetectionStrategy(Protocol):
    """Interface for classical and optional ML layout detectors."""

    name: str

    @classmethod
    def is_available(cls) -> bool:
        ...

    def detect(self, image: np.ndarray) -> LayoutDetectionResult:
        ...


def skipped_result(strategy_name: str, reason: str, *, metadata: dict[str, Any] | None = None) -> LayoutDetectionResult:
    return LayoutDetectionResult(
        strategy_name=strategy_name,
        available=False,
        skipped_reason=reason,
        metadata=dict(metadata or {}),
    )
