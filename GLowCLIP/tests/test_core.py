from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from glowclip.config import ModelConfig
from glowclip.losses import glowclip_loss
from glowclip.metrics import compute_metrics, roc_auc, select_youden_threshold
from glowclip.model import GLowCLIP, LoRALinear, split_outputs


class DummyAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)


class DummyLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = DummyAttention(hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + 0.1 * (
            self.self_attn.q_proj(tokens) + self.self_attn.v_proj(tokens)
        )


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 32, layer_count: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList(
            [DummyLayer(hidden_size) for _ in range(layer_count)]
        )
        self.input_proj = nn.Linear(3, hidden_size)

    def forward(self, pixel_values: torch.Tensor):
        pooled = pixel_values.mean(dim=(2, 3))
        token = self.input_proj(pooled)
        tokens = token[:, None, :].repeat(1, 5, 1)
        for layer in self.vision_model.encoder.layers:
            tokens = layer(tokens)
        return SimpleNamespace(last_hidden_state=tokens)


def test_model_shapes_lora_and_loss_backward() -> None:
    config = ModelConfig(
        pretrained_name="dummy",
        feature_dim=16,
        lora_rank=2,
        lora_alpha=4.0,
        lora_last_n_layers=2,
        gradient_checkpointing=False,
    )
    model = GLowCLIP(DummyBackbone(), config)
    lora_modules = [
        module for module in model.modules() if isinstance(module, LoRALinear)
    ]
    assert len(lora_modules) == 4
    assert all(
        not parameter.requires_grad
        for module in lora_modules
        for parameter in module.base.parameters()
    )

    outputs = model(torch.randn(6, 3, 16, 16))
    assert outputs["fused_logit"].shape == (6,)
    assert outputs["fused_feature"].shape == (6, 16)
    assert outputs["gate"].shape == (6, 16)
    clean, degraded = split_outputs(outputs, 3)
    losses = glowclip_loss(clean, degraded, torch.tensor([0.0, 1.0, 0.0]))
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert any(module.lora_b.grad is not None for module in lora_modules)

    adapter = model.adapter_state_dict()
    assert adapter
    assert not any(".base." in name for name in adapter)
    clone = GLowCLIP(DummyBackbone(), config)
    clone.load_adapter_state_dict(adapter)


def test_metrics_and_threshold() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(labels, scores) == 1.0
    threshold = select_youden_threshold(labels, scores)
    assert 0.2 < threshold <= 0.8


def test_roc_auc_ties() -> None:
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_family_aware_robust_auc() -> None:
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    metrics = compute_metrics(
        labels,
        scores,
        threshold=0.5,
        generators=["a", "a", "a", "a", "b", "b", "b", "b"],
        transform_levels=[1, 1, 3, 3, 1, 1, 3, 3],
        min_group_samples=2,
        transform_families=[
            "first",
            "first",
            "first",
            "first",
            "second",
            "second",
            "second",
            "second",
        ],
        source_datasets=["one", "one", "one", "one", "two", "two", "two", "two"],
    )
    assert metrics["robust_auc"] == 1.0
    assert metrics["robust_grouping"] == "transform_family_x_level"
    assert set(metrics["groups"]["transform_family"]) == {"first", "second"}
