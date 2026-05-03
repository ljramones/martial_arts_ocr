"""Review-layer wrapper for the existing NN page-orientation subsystem.

This module intentionally does not implement a new detector. It wraps the
existing orientation CNN path so the review workbench can depend on a stable,
testable service contract without changing OCR/extraction runtime defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


VALID_ROTATIONS = {0, 90, 180, 270}
ORIENTATION_CONVENTION = "clockwise_rotation_to_apply_to_display_upright"
DEFAULT_MODEL_PATH = Path("experiments/orientation_model/checkpoints/orient_convnext_tiny.pth")
DEFAULT_ENSEMBLE_MODEL_PATH = Path("experiments/orientation_model/checkpoints/orient_effnetv2s.pth")


@dataclass(frozen=True)
class OrientationResult:
    """Page-orientation prediction result for review/workbench use."""

    rotation_degrees: int
    confidence: float | None = None
    source: str = "unavailable"
    status: str = "unavailable"
    metadata: dict[str, Any] = field(default_factory=dict)


class OrientationService:
    """Wrap the existing NN orientation detector for review-layer callers."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        ensemble_model_path: str | Path | None = DEFAULT_ENSEMBLE_MODEL_PATH,
        device: str | None = None,
        predictor_backend: Any | None = None,
        raise_errors: bool = False,
        use_ensemble_if_low_margin: bool = True,
        margin: float = 0.55,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.ensemble_model_path = (
            Path(ensemble_model_path) if ensemble_model_path is not None else None
        )
        self.device = device
        self.predictor_backend = predictor_backend
        self.raise_errors = raise_errors
        self.use_ensemble_if_low_margin = use_ensemble_if_low_margin
        self.margin = margin
        self._initialized = False

    def predict(self, image_path: str | Path) -> OrientationResult:
        """Predict page orientation for a local image path.

        `rotation_degrees` uses the review-layer convention:

        ```text
        clockwise rotation to apply to the original image to display/process it upright
        ```

        The existing NN output is currently passed through as the candidate
        rotation. The future workbench UI should keep this result reviewable and
        should not destructively rotate source files.
        """

        model_path = self._expand(self.model_path)
        if not model_path.exists():
            return self._manual_required(
                model_path,
                reason="orientation checkpoint is unavailable",
            )

        path = self._expand(Path(image_path))
        if not path.exists():
            return self._error_result(
                model_path,
                error=f"image path does not exist: {path}",
            )

        try:
            backend = self._backend()
            if not self._initialized:
                ensemble_path = self._available_ensemble_path()
                backend.init_orientation_model(
                    str(model_path),
                    str(ensemble_path) if ensemble_path is not None else None,
                )
                self._initialized = True

            np_image = self._load_image(path)
            raw_result = backend.predict_degrees(
                np_image,
                use_ensemble_if_low_margin=self.use_ensemble_if_low_margin,
                margin=self.margin,
            )
            rotation_degrees, scores_by_degree, confidence, model_used = _normalize_backend_result(raw_result)
            if rotation_degrees not in VALID_ROTATIONS:
                return self._error_result(
                    model_path,
                    error=f"invalid orientation rotation: {rotation_degrees}",
                    metadata={
                        "invalid_rotation_degrees": rotation_degrees,
                        "scores_by_degree": scores_by_degree,
                    },
                )
            return OrientationResult(
                rotation_degrees=rotation_degrees,
                confidence=confidence,
                source="orientation_cnn",
                status="ok",
                metadata={
                    "model_path": str(model_path),
                    "ensemble_model_path": (
                        str(self._available_ensemble_path())
                        if self._available_ensemble_path() is not None
                        else None
                    ),
                    "model_used": model_used,
                    "scores_by_degree": scores_by_degree,
                    "orientation_convention": ORIENTATION_CONVENTION,
                    "detector_family": "nn_orientation_cnn",
                    "heuristic_fallback_used": False,
                },
            )
        except Exception as exc:
            if self.raise_errors:
                raise
            return self._error_result(
                model_path,
                error=str(exc),
                metadata={"error_type": type(exc).__name__},
            )

    def _backend(self) -> Any:
        if self.predictor_backend is None:
            self.predictor_backend = import_module("utils.image.preprocessing.orientation_cnn")
        return self.predictor_backend

    def _available_ensemble_path(self) -> Path | None:
        if self.ensemble_model_path is None:
            return None
        ensemble_path = self._expand(self.ensemble_model_path)
        return ensemble_path if ensemble_path.exists() else None

    def _manual_required(self, model_path: Path, reason: str) -> OrientationResult:
        return OrientationResult(
            rotation_degrees=0,
            confidence=None,
            source="unavailable",
            status="manual_required",
            metadata={
                "model_path": str(model_path),
                "model_available": False,
                "reason": reason,
                "orientation_convention": ORIENTATION_CONVENTION,
                "heuristic_fallback_used": False,
            },
        )

    def _error_result(
        self,
        model_path: Path,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> OrientationResult:
        details = {
            "model_path": str(model_path),
            "error": error,
            "orientation_convention": ORIENTATION_CONVENTION,
            "heuristic_fallback_used": False,
        }
        details.update(metadata or {})
        return OrientationResult(
            rotation_degrees=0,
            confidence=None,
            source="orientation_cnn",
            status="error",
            metadata=details,
        )

    def _load_image(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            return np.asarray(oriented)

    def _expand(self, path: Path) -> Path:
        return path.expanduser()


def _normalize_backend_result(raw_result: Any) -> tuple[int, dict[int, float], float | None, str]:
    """Normalize existing orientation backend outputs into service fields."""
    if not isinstance(raw_result, tuple):
        raise ValueError(f"Unexpected orientation backend result: {raw_result!r}")
    if len(raw_result) == 4:
        rotation_degrees, scores_by_degree, confidence, model_used = raw_result
    elif len(raw_result) == 2:
        rotation_degrees, scores_by_degree = raw_result
        confidence = max(scores_by_degree.values()) if scores_by_degree else None
        model_used = "unknown"
    else:
        raise ValueError(f"Unexpected orientation backend result length: {len(raw_result)}")

    scores = {
        int(degree): float(score)
        for degree, score in dict(scores_by_degree or {}).items()
    }
    return int(rotation_degrees), scores, None if confidence is None else float(confidence), str(model_used)
