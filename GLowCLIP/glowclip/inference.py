from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from PIL import Image

from .config import ModelConfig
from .model import GLowCLIP, load_checkpoint
from .runtime import (
    autocast_context,
    configure_torch,
    resolve_device,
    resolve_precision,
)
from .transforms import CLIPImageTransform, decode_rgb, letterbox


@dataclass(frozen=True)
class SingleImagePrediction:
    fake_probability: float
    threshold: float
    gate_mean: float

    @property
    def real_probability(self) -> float:
        return 1.0 - self.fake_probability

    @property
    def predicted_label(self) -> str:
        return "AIGC" if self.fake_probability >= self.threshold else "Real"

    @property
    def confidence(self) -> float:
        if self.predicted_label == "AIGC":
            return self.fake_probability
        return self.real_probability

    @property
    def summary(self) -> str:
        label = "AI-generated" if self.predicted_label == "AIGC" else "real"
        return f"{self.confidence * 100.0:.1f}% likely {label}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.predicted_label,
            "summary": self.summary,
            "real_probability": round(self.real_probability, 6),
            "ai_generated_probability": round(self.fake_probability, 6),
            "decision_threshold": round(self.threshold, 6),
            "gate_mean": round(self.gate_mean, 6),
        }


class GLowCLIPPredictor:
    """Reusable, thread-safe predictor for one PIL image at a time."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        precision: str | None = None,
    ) -> None:
        configure_torch()
        checkpoint = load_checkpoint(checkpoint_path)
        experiment = checkpoint.get("experiment_config", {})
        model_config = ModelConfig(**checkpoint["model_config"])
        # Checkpointing only saves memory during training and slows inference.
        model_config = replace(model_config, gradient_checkpointing=False)

        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device)
        requested_precision = precision or experiment.get("train", {}).get(
            "precision", "bf16"
        )
        self.precision = resolve_precision(requested_precision, self.device)
        self.threshold = float(checkpoint.get("threshold", 0.5))
        self.image_size = model_config.image_size
        self.transform = CLIPImageTransform(self.image_size)
        self.model = GLowCLIP.from_pretrained(model_config)
        self.model.load_adapter_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self._inference_lock = Lock()

    def model_input_preview(self, image: Image.Image) -> Image.Image:
        return letterbox(decode_rgb(image), self.image_size)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> SingleImagePrediction:
        pixels = self.transform(image).unsqueeze(0).to(self.device)
        with self._inference_lock, autocast_context(self.device, self.precision):
            outputs = self.model(pixels)
        fake_probability = float(torch.sigmoid(outputs["fused_logit"].float()).item())
        gate_mean = float(outputs["gate"].float().mean().item())
        return SingleImagePrediction(
            fake_probability=fake_probability,
            threshold=self.threshold,
            gate_mean=gate_mean,
        )
