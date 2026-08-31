from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class LoRALinear(nn.Module):
    """A frozen Linear layer plus a trainable low-rank residual."""

    def __init__(
        self, base: nn.Linear, rank: int, alpha: float, dropout: float
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        lora_output = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return base_output + lora_output * self.scaling


class GLowCLIP(nn.Module):
    """Fuse global and local CLIP features with an input-dependent gate."""

    def __init__(self, backbone: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        layers = self._encoder_layers()
        if config.lora_last_n_layers > len(layers):
            raise ValueError(
                f"Requested LoRA on {config.lora_last_n_layers} layers, but backbone has {len(layers)}"
            )
        for layer in layers[-config.lora_last_n_layers :]:
            attention = layer.self_attn
            attention.q_proj = LoRALinear(
                attention.q_proj,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )
            attention.v_proj = LoRALinear(
                attention.v_proj,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )

        hidden_size = self._hidden_size()
        dim = config.feature_dim
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
        )
        self.patch_reduce = nn.Sequential(
            nn.Conv2d(hidden_size, dim, kernel_size=1),
            nn.GroupNorm(1, dim),
            nn.GELU(),
        )
        self.patch_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.patch_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 3, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, dim),
            nn.Sigmoid(),
        )
        self.fused_norm = nn.LayerNorm(dim)
        self.global_cls = nn.Linear(dim, 1)
        self.patch_cls = nn.Linear(dim, 1)
        self.fused_cls = nn.Linear(dim, 1)

    @classmethod
    def from_pretrained(cls, config: ModelConfig) -> GLowCLIP:
        try:
            from transformers import CLIPVisionModel
        except ImportError as error:
            raise RuntimeError(
                "transformers is required; install the project dependencies"
            ) from error
        backbone = CLIPVisionModel.from_pretrained(config.pretrained_name)
        model = cls(backbone, config)
        if config.gradient_checkpointing:
            try:
                backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                backbone.gradient_checkpointing_enable()
            if hasattr(backbone, "enable_input_require_grads"):
                backbone.enable_input_require_grads()
        return model

    def _encoder_layers(self) -> nn.ModuleList:
        # Transformers 4.x nests this under ``vision_model``; 5.x exposes the
        # vision stack directly on CLIPVisionModel.
        if hasattr(self.backbone, "vision_model"):
            return self.backbone.vision_model.encoder.layers
        if hasattr(self.backbone, "encoder"):
            return self.backbone.encoder.layers
        raise TypeError("Backbone does not expose CLIP vision encoder layers")

    def _hidden_size(self) -> int:
        if hasattr(self.backbone, "config") and hasattr(
            self.backbone.config, "hidden_size"
        ):
            return int(self.backbone.config.hidden_size)
        first_layer = self._encoder_layers()[0]
        return int(first_layer.self_attn.q_proj.in_features)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise RuntimeError(f"Unexpected CLIP token shape: {tuple(tokens.shape)}")

        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        global_feature = self.global_proj(cls_token)

        batch, patch_count, channels = patch_tokens.shape
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise RuntimeError(f"Patch count is not a square: {patch_count}")
        feature_map = patch_tokens.transpose(1, 2).reshape(batch, channels, side, side)
        feature_map = self.patch_reduce(feature_map)
        feature_map = feature_map + F.gelu(self.patch_dwconv(feature_map))
        mean = feature_map.mean(dim=(2, 3))
        std = feature_map.var(dim=(2, 3), unbiased=False).add(1e-6).sqrt()
        patch_feature = self.patch_proj(torch.cat((mean, std), dim=-1))

        gate_input = torch.cat(
            (global_feature, patch_feature, torch.abs(global_feature - patch_feature)),
            dim=-1,
        )
        gate = self.fusion_gate(gate_input)
        fused_feature = self.fused_norm(
            gate * global_feature + (1.0 - gate) * patch_feature
        )
        return {
            "fused_logit": self.fused_cls(fused_feature).squeeze(-1),
            "global_logit": self.global_cls(global_feature).squeeze(-1),
            "patch_logit": self.patch_cls(patch_feature).squeeze(-1),
            "fused_feature": fused_feature,
            "global_feature": global_feature,
            "patch_feature": patch_feature,
            "gate": gate,
        }

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.modules():
            if isinstance(module, LoRALinear):
                yield module.lora_a
                yield module.lora_b

    def head_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone."):
                yield parameter

    def set_lora_trainable(self, trainable: bool) -> None:
        for parameter in self.lora_parameters():
            parameter.requires_grad = trainable

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            self._canonical_adapter_name(name): tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith("backbone.")
            or ".lora_a" in name
            or ".lora_b" in name
        }

    def load_adapter_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        actual_by_canonical = {
            self._canonical_adapter_name(name): name
            for name in self.state_dict()
            if not name.startswith("backbone.")
            or ".lora_a" in name
            or ".lora_b" in name
        }
        expected = set(actual_by_canonical)
        supplied = set(state_dict)
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        if missing or unexpected:
            raise ValueError(
                f"Adapter checkpoint mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        translated = {
            actual_by_canonical[name]: tensor for name, tensor in state_dict.items()
        }
        self.load_state_dict(translated, strict=False)

    @staticmethod
    def _canonical_adapter_name(name: str) -> str:
        # Keep adapter files portable across the Transformers 4.x/5.x CLIP layout.
        return name.replace("backbone.vision_model.encoder.", "backbone.encoder.")

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        lora = sum(parameter.numel() for parameter in self.lora_parameters())
        heads = sum(parameter.numel() for parameter in self.head_parameters())
        return {"total": total, "trainable": trainable, "lora": lora, "heads": heads}


def split_outputs(
    outputs: dict[str, torch.Tensor], split_size: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    clean: dict[str, torch.Tensor] = {}
    degraded: dict[str, torch.Tensor] = {}
    for key, value in outputs.items():
        if value.shape[0] != split_size * 2:
            raise ValueError(
                f"Cannot split {key} with batch size {value.shape[0]} at {split_size}"
            )
        clean[key], degraded[key] = value.split(split_size, dim=0)
    return clean, degraded


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Not a compatible GLowCLIP checkpoint: {path}")
    return checkpoint
