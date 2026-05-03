from __future__ import annotations

from pathlib import Path

from PIL import Image

from martial_arts_ocr.review.orientation_service import (
    ORIENTATION_CONVENTION,
    OrientationService,
)


class FakeOrientationBackend:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or (90, {0: 0.02, 90: 0.91, 180: 0.04, 270: 0.03}, 0.91, "convnext")
        self.error = error
        self.init_calls = []
        self.predict_calls = []

    def init_orientation_model(self, ckpt_convnext: str, ckpt_effnet: str | None = None) -> None:
        self.init_calls.append((ckpt_convnext, ckpt_effnet))

    def predict_degrees(self, np_img, use_ensemble_if_low_margin=True, margin=0.55):
        self.predict_calls.append(
            {
                "shape": np_img.shape,
                "use_ensemble_if_low_margin": use_ensemble_if_low_margin,
                "margin": margin,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result


def _write_image(path: Path) -> None:
    Image.new("RGB", (32, 24), "white").save(path)


def test_missing_model_returns_manual_required(tmp_path):
    image_path = tmp_path / "page.png"
    _write_image(image_path)

    service = OrientationService(model_path=tmp_path / "missing.pth")
    result = service.predict(image_path)

    assert result.rotation_degrees == 0
    assert result.confidence is None
    assert result.source == "unavailable"
    assert result.status == "manual_required"
    assert result.metadata["model_available"] is False
    assert result.metadata["heuristic_fallback_used"] is False


def test_fake_predictor_result_maps_to_orientation_result(tmp_path):
    model_path = tmp_path / "orient_convnext_tiny.pth"
    model_path.write_bytes(b"fake checkpoint")
    image_path = tmp_path / "page.png"
    _write_image(image_path)
    backend = FakeOrientationBackend()

    service = OrientationService(
        model_path=model_path,
        ensemble_model_path=None,
        predictor_backend=backend,
    )
    result = service.predict(image_path)

    assert result.rotation_degrees == 90
    assert result.confidence == 0.91
    assert result.source == "orientation_cnn"
    assert result.status == "ok"
    assert result.metadata["model_used"] == "convnext"
    assert result.metadata["scores_by_degree"][90] == 0.91
    assert result.metadata["orientation_convention"] == ORIENTATION_CONVENTION
    assert result.metadata["heuristic_fallback_used"] is False
    assert backend.init_calls == [(str(model_path), None)]
    assert backend.predict_calls[0]["shape"] == (24, 32, 3)


def test_confidence_is_derived_from_two_value_backend_result(tmp_path):
    model_path = tmp_path / "orient_convnext_tiny.pth"
    model_path.write_bytes(b"fake checkpoint")
    image_path = tmp_path / "page.png"
    _write_image(image_path)
    backend = FakeOrientationBackend(result=(180, {0: 0.1, 90: 0.2, 180: 0.6, 270: 0.1}))

    result = OrientationService(model_path=model_path, predictor_backend=backend).predict(image_path)

    assert result.rotation_degrees == 180
    assert result.confidence == 0.6
    assert result.metadata["model_used"] == "unknown"


def test_invalid_rotation_output_returns_error(tmp_path):
    model_path = tmp_path / "orient_convnext_tiny.pth"
    model_path.write_bytes(b"fake checkpoint")
    image_path = tmp_path / "page.png"
    _write_image(image_path)
    backend = FakeOrientationBackend(result=(45, {45: 1.0}, 1.0, "convnext"))

    result = OrientationService(model_path=model_path, predictor_backend=backend).predict(image_path)

    assert result.rotation_degrees == 0
    assert result.status == "error"
    assert result.metadata["invalid_rotation_degrees"] == 45
    assert result.metadata["heuristic_fallback_used"] is False


def test_prediction_exception_returns_error_without_crashing(tmp_path):
    model_path = tmp_path / "orient_convnext_tiny.pth"
    model_path.write_bytes(b"fake checkpoint")
    image_path = tmp_path / "page.png"
    _write_image(image_path)
    backend = FakeOrientationBackend(error=RuntimeError("backend failed"))

    result = OrientationService(model_path=model_path, predictor_backend=backend).predict(image_path)

    assert result.rotation_degrees == 0
    assert result.status == "error"
    assert "backend failed" in result.metadata["error"]
    assert result.metadata["error_type"] == "RuntimeError"
    assert result.metadata["heuristic_fallback_used"] is False


def test_missing_image_returns_error(tmp_path):
    model_path = tmp_path / "orient_convnext_tiny.pth"
    model_path.write_bytes(b"fake checkpoint")
    backend = FakeOrientationBackend()

    result = OrientationService(model_path=model_path, predictor_backend=backend).predict(
        tmp_path / "missing.png"
    )

    assert result.status == "error"
    assert "image path does not exist" in result.metadata["error"]
    assert backend.init_calls == []
