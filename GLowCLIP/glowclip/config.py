from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    pretrained_name: str = "openai/clip-vit-base-patch16"
    image_size: int = 224
    feature_dim: int = 256
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_last_n_layers: int = 4
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    image_root: str = "data/images"
    manifest_root: str = "data/manifests"
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    online_degradation: bool = True


@dataclass
class TrainConfig:
    head_epochs: int = 1
    joint_epochs: int = 4
    gradient_accumulation: int = 2
    head_learning_rate: float = 5e-4
    lora_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lr_warmup_fraction: float = 0.08
    consistency_warmup_fraction: float = 0.10
    lambda_aux: float = 0.2
    lambda_feature: float = 0.2
    lambda_prediction: float = 0.1
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    output_dir: str = "outputs/glowclip"
    log_every: int = 25
    compile: bool = False


@dataclass
class EvaluationConfig:
    threshold: float = 0.5
    threshold_strategy: str = "youden"
    min_group_samples: int = 20


@dataclass
class ExperimentConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTION_TYPES = {
    "model": ModelConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "evaluation": EvaluationConfig,
}


def _check_keys(values: dict[str, Any], cls: type, section: str) -> None:
    allowed = {item.name for item in dataclasses.fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {', '.join(unknown)}")


def _parse_scalar(value: str) -> Any:
    parsed = yaml.safe_load(value)
    if isinstance(parsed, (dict, list)):
        raise TypeError(f"Override values must be scalar, got: {value}")
    return parsed


def _apply_overrides(raw: dict[str, Any], overrides: Iterable[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must use section.key=value: {override}")
        dotted_key, raw_value = override.split("=", 1)
        parts = dotted_key.split(".")
        if len(parts) == 1 and parts[0] == "seed":
            raw["seed"] = _parse_scalar(raw_value)
            continue
        if len(parts) != 2 or parts[0] not in _SECTION_TYPES:
            raise ValueError(f"Unsupported override key: {dotted_key}")
        raw.setdefault(parts[0], {})[parts[1]] = _parse_scalar(raw_value)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("The configuration root must be a mapping")
    _apply_overrides(raw, overrides)

    allowed_root = {"seed", *_SECTION_TYPES}
    unknown_root = sorted(set(raw) - allowed_root)
    if unknown_root:
        raise ValueError(f"Unknown configuration sections: {', '.join(unknown_root)}")

    sections: dict[str, Any] = {}
    for name, cls in _SECTION_TYPES.items():
        values = raw.get(name, {})
        if not isinstance(values, dict):
            raise TypeError(f"Configuration section {name} must be a mapping")
        _check_keys(values, cls, name)
        sections[name] = cls(**values)

    config = ExperimentConfig(seed=int(raw.get("seed", 42)), **sections)
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.model.image_size <= 0:
        raise ValueError("model.image_size must be positive")
    if config.model.lora_rank <= 0 or config.model.lora_last_n_layers <= 0:
        raise ValueError("LoRA rank and layer count must be positive")
    if config.data.batch_size <= 0 or config.data.num_workers < 0:
        raise ValueError("Invalid data loader settings")
    if config.train.head_epochs < 0 or config.train.joint_epochs < 0:
        raise ValueError("Epoch counts cannot be negative")
    if config.train.head_epochs + config.train.joint_epochs <= 0:
        raise ValueError("At least one training epoch is required")
    if config.train.gradient_accumulation <= 0:
        raise ValueError("train.gradient_accumulation must be positive")
    if config.train.precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("train.precision must be bf16, fp16, or fp32")
    if config.evaluation.threshold_strategy not in {"fixed", "youden"}:
        raise ValueError("evaluation.threshold_strategy must be fixed or youden")
