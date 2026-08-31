from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from glowclip.demo_degradations import CENTER_CROP, JPEG_COMPRESSION
from glowclip.inference import SingleImagePrediction
from glowclip.space_app import (
    apply_degradations_for_ui,
    build_demo,
    prediction_for_ui,
    resolve_space_checkpoint,
    undo_degradations_for_ui,
)


class FakePredictor:
    image_size = 224

    def predict(self, image: Image.Image) -> SingleImagePrediction:
        assert image.size == (32, 16)
        return SingleImagePrediction(0.01, 0.4474602938, 0.33)

    def model_input_preview(self, image: Image.Image) -> Image.Image:
        assert image.size == (32, 16)
        return Image.new("RGB", (self.image_size, self.image_size))


class FakeService:
    def predictor(self) -> FakePredictor:
        return FakePredictor()


def test_apply_and_undo_callbacks() -> None:
    original = Image.new("RGB", (40, 20), (80, 120, 160))
    processed, status, *cleared = apply_degradations_for_ui(
        original,
        [CENTER_CROP, JPEG_COMPRESSION],
        50,
        1.0,
        0.5,
        0.05,
        20,
        -10,
    )
    assert processed.size == (32, 16)
    assert "JPEG q50" in status
    assert "Crop 80%" in status
    assert cleared == [None, None, None]

    selected, restored, status, *cleared = undo_degradations_for_ui(original)
    assert selected == []
    assert restored is not None and restored.size == original.size
    assert "Original" in status
    assert cleared == [None, None, None]


def test_prediction_callback_reports_human_score_and_model_input() -> None:
    image = Image.new("RGB", (32, 16))
    score, class_scores, model_input = prediction_for_ui(
        image,
        FakeService(),  # type: ignore[arg-type]
    )
    assert "99.0% likely real" in score
    assert class_scores == pytest.approx({"Real": 0.99, "AI-generated": 0.01})
    assert model_input.size == (224, 224)


def test_checkpoint_resolution_prefers_explicit_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert resolve_space_checkpoint(checkpoint) == checkpoint.resolve()


def test_gradio_demo_builds_without_loading_checkpoint() -> None:
    demo = build_demo()
    config = demo.get_config_file()
    assert config["mode"] == "blocks"
    assert any(
        component.get("props", {}).get("value") == "Analyze"
        for component in config["components"]
    )
    assert not any(component["type"] == "json" for component in config["components"])
    html = "\n".join(
        str(component.get("props", {}).get("value", ""))
        for component in config["components"]
        if component["type"] == "html"
    )
    assert "color: #fff !important" in html
    assert "-webkit-text-fill-color: #fff !important" in html
    labels = {
        component.get("props", {}).get("label") for component in config["components"]
    }
    assert {"Brightness %", "Contrast %"}.issubset(labels)
    assert "Random seed" not in labels
    assert any(
        component["type"] == "accordion"
        and component.get("props", {}).get("label") == "Settings"
        and component.get("props", {}).get("open") is True
        for component in config["components"]
    )
