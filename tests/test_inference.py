from __future__ import annotations

import pytest

from glowclip.inference import SingleImagePrediction


def test_real_prediction_summary_and_payload() -> None:
    prediction = SingleImagePrediction(
        fake_probability=0.01,
        threshold=0.4474602938,
        gate_mean=0.31,
    )
    assert prediction.predicted_label == "Real"
    assert prediction.confidence == pytest.approx(0.99)
    assert prediction.summary == "99.0% likely real"
    assert prediction.to_dict() == {
        "prediction": "Real",
        "summary": "99.0% likely real",
        "real_probability": 0.99,
        "ai_generated_probability": 0.01,
        "decision_threshold": 0.44746,
        "gate_mean": 0.31,
    }


def test_aigc_decision_uses_checkpoint_threshold() -> None:
    prediction = SingleImagePrediction(
        fake_probability=0.46,
        threshold=0.4474602938,
        gate_mean=0.57,
    )
    assert prediction.predicted_label == "AIGC"
    assert prediction.confidence == pytest.approx(0.46)
    assert prediction.summary == "46.0% likely AI-generated"
