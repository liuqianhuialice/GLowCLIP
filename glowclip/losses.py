from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_distribution(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Compute the clamp in FP32: 1 - 1e-6 rounds back to 1 in BF16.
    probability = torch.sigmoid(logits.float()).clamp(eps, 1.0 - eps)
    return torch.stack((1.0 - probability, probability), dim=-1)


def symmetric_kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_to_right = (left * (left.log() - right.log())).sum(dim=-1)
    right_to_left = (right * (right.log() - left.log())).sum(dim=-1)
    return 0.5 * (left_to_right + right_to_left).mean()


def cosine_consistency(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left.float(), dim=-1)
    right = F.normalize(right.float(), dim=-1)
    return (1.0 - (left * right).sum(dim=-1)).mean()


def glowclip_loss(
    clean: dict[str, torch.Tensor],
    degraded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_aux: float = 0.2,
    lambda_feature: float = 0.2,
    lambda_prediction: float = 0.1,
) -> dict[str, torch.Tensor]:
    labels = labels.float()
    fused_classification = 0.5 * (
        F.binary_cross_entropy_with_logits(clean["fused_logit"], labels)
        + F.binary_cross_entropy_with_logits(degraded["fused_logit"], labels)
    )
    auxiliary_classification = 0.25 * (
        F.binary_cross_entropy_with_logits(clean["global_logit"], labels)
        + F.binary_cross_entropy_with_logits(degraded["global_logit"], labels)
        + F.binary_cross_entropy_with_logits(clean["patch_logit"], labels)
        + F.binary_cross_entropy_with_logits(degraded["patch_logit"], labels)
    )
    feature_consistency = cosine_consistency(
        clean["fused_feature"], degraded["fused_feature"]
    )
    prediction_consistency = symmetric_kl(
        binary_distribution(clean["fused_logit"]),
        binary_distribution(degraded["fused_logit"]),
    )
    loss = (
        fused_classification
        + lambda_aux * auxiliary_classification
        + lambda_feature * feature_consistency
        + lambda_prediction * prediction_consistency
    )
    return {
        "loss": loss,
        "fused_cls": fused_classification.detach(),
        "aux_cls": auxiliary_classification.detach(),
        "feat_cons": feature_consistency.detach(),
        "pred_cons": prediction_consistency.detach(),
    }
