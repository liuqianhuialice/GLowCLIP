from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class BuiltBaseline:
    model: nn.Module
    train_transform: Any
    evaluation_transform: Any
    model_metadata: dict[str, Any]


def _require_torchvision() -> Any:
    try:
        import torchvision
    except ImportError as error:
        raise RuntimeError(
            "This baseline needs the optional 'torchvision' package. "
            "It was intentionally not installed during repository adaptation; "
            "see docs/BASELINES_ADAPTATION.md."
        ) from error
    return torchvision


def _require_open_clip() -> Any:
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError(
            "This baseline needs the optional 'open_clip_torch' package. "
            "It was intentionally not installed during repository adaptation; "
            "see docs/BASELINES_ADAPTATION.md."
        ) from error
    return open_clip


class ResNetBaseline(nn.Module):
    def __init__(self, backbone: nn.Module, dropout: float = 0.3) -> None:
        super().__init__()
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(images))


class OpenCLIPLinearProbe(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        feature_dim = int(self.backbone.visual.output_dim)
        self.classifier = nn.Linear(feature_dim, 1)

    def train(self, mode: bool = True) -> OpenCLIPLinearProbe:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone.encode_image(images)
            features = F.normalize(features, p=2, dim=1)
        return self.classifier(features.float())


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return self.relu(output + identity)


class NPRDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(channels=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(channels=128, blocks=4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        output_channels = channels * Bottleneck.expansion
        downsample: nn.Module | None = None
        if stride != 1 or self.in_channels != output_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, output_channels, stride),
                nn.BatchNorm2d(output_channels),
            )
        layers: list[nn.Module] = [
            Bottleneck(self.in_channels, channels, stride, downsample)
        ]
        self.in_channels = output_channels
        layers.extend(Bottleneck(self.in_channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    @staticmethod
    def get_npr(inputs: torch.Tensor) -> torch.Tensor:
        half = F.interpolate(
            inputs, scale_factor=0.5, mode="nearest", recompute_scale_factor=True
        )
        reconstructed = F.interpolate(
            half, scale_factor=2.0, mode="nearest", recompute_scale_factor=True
        )
        return inputs - reconstructed

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.get_npr(inputs)
        inputs = self.maxpool(self.relu(self.bn1(self.conv1(inputs * (2.0 / 3.0)))))
        inputs = self.layer2(self.layer1(inputs))
        inputs = torch.flatten(self.avgpool(inputs), 1)
        return self.classifier(inputs)


class VIBDetector(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        latent_dim: int = 256,
        dropout: float = 0.5,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(feature_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc_statistics = nn.Linear(1024, latent_dim * 2)
        self.classifier = nn.Linear(latent_dim, 1)

    def train(self, mode: bool = True) -> VIBDetector:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def reparameterize(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return mu + std * torch.randn_like(std)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone.encode_image(images).float()
        else:
            features = self.backbone.encode_image(images).float()
        hidden = F.relu(self.fc1(self.dropout(features)))
        hidden = F.relu(self.fc2(hidden))
        statistics = self.fc_statistics(hidden)
        mu = statistics[:, : self.latent_dim]
        raw_std = statistics[:, self.latent_dim :]
        std = F.softplus(raw_std - 5.0)
        latent = self.reparameterize(mu, std) if self.training else mu
        return self.classifier(latent), mu, std


def kl_divergence(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    variance = std.pow(2)
    values = 0.5 * (mu.pow(2) + variance - torch.log(variance + 1e-8) - 1)
    return values.sum(dim=1).mean()


def build_baseline(name: str, config: dict[str, Any]) -> BuiltBaseline:
    """Build one notebook baseline; optional packages are imported on demand."""
    if name == "resnet18":
        torchvision = _require_torchvision()
        transforms = torchvision.transforms
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        backbone = torchvision.models.resnet18(weights=weights)
        model = ResNetBaseline(backbone, dropout=float(config["dropout"]))
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        evaluation_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        metadata = {
            "architecture": "torchvision resnet18",
            "pretrained": "ResNet18_Weights.DEFAULT",
            "checkpoint_state": "full model",
        }
    elif name == "openclip_linear":
        open_clip = _require_open_clip()
        backbone, _, evaluation_transform = open_clip.create_model_and_transforms(
            str(config["model_name"]), pretrained=str(config["pretrained"])
        )
        model = OpenCLIPLinearProbe(backbone)
        train_transform = evaluation_transform
        metadata = {
            "architecture": config["model_name"],
            "pretrained": config["pretrained"],
            "checkpoint_state": "trainable classifier only",
            "transform_note": "Notebook used the OpenCLIP evaluation transform for training",
        }
    elif name == "npr":
        torchvision = _require_torchvision()
        transforms = torchvision.transforms
        model = NPRDetector()
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        evaluation_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        metadata = {
            "architecture": "notebook NPR two-stage bottleneck detector",
            "pretrained": None,
            "checkpoint_state": "full model",
        }
    elif name == "vib":
        open_clip = _require_open_clip()
        backbone, _, evaluation_transform = open_clip.create_model_and_transforms(
            str(config["model_name"]), pretrained=str(config["pretrained"])
        )
        feature_dim = int(backbone.visual.output_dim)
        model = VIBDetector(
            backbone,
            feature_dim=feature_dim,
            latent_dim=int(config["latent_dim"]),
            dropout=float(config["dropout"]),
            freeze_backbone=bool(config["freeze_backbone"]),
        )
        train_transform = evaluation_transform
        metadata = {
            "architecture": f"VIB head over {config['model_name']}",
            "feature_dim": feature_dim,
            "pretrained": config["pretrained"],
            "checkpoint_state": "trainable VIB head only",
            "evaluation_latent": "posterior mean (deterministic)",
        }
    else:
        raise ValueError(f"Unknown baseline: {name}")
    return BuiltBaseline(model, train_transform, evaluation_transform, metadata)
