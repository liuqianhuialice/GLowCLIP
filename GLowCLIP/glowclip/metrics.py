from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


def roc_auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    positives = int(labels_array.sum())
    negatives = int(labels_array.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores_array, kind="mergesort")
    sorted_scores = scores_array[order]
    ranks = np.empty(scores_array.size, dtype=np.float64)
    start = 0
    while start < scores_array.size:
        end = start + 1
        while end < scores_array.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = ranks[labels_array == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def select_youden_threshold(labels: Sequence[float], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    positives = labels_array.sum()
    negatives = labels_array.size - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(-scores_array, kind="mergesort")
    sorted_labels = labels_array[order]
    sorted_scores = scores_array[order]
    true_positives = np.cumsum(sorted_labels)
    false_positives = np.cumsum(1 - sorted_labels)
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    statistic = (
        true_positives[group_ends] / positives - false_positives[group_ends] / negatives
    )
    best = group_ends[int(np.argmax(statistic))]
    return float(sorted_scores[best])


def _classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = scores >= threshold
    positives = labels == 1
    negatives = ~positives
    true_positive = int(np.sum(predictions & positives))
    true_negative = int(np.sum(~predictions & negatives))
    false_positive = int(np.sum(predictions & negatives))
    false_negative = int(np.sum(~predictions & positives))
    sensitivity = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    return {
        "count": int(labels.size),
        "roc_auc": roc_auc(labels, scores),
        "accuracy": float((true_positive + true_negative) / max(1, labels.size)),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def compute_metrics(
    labels: Sequence[float],
    scores: Sequence[float],
    threshold: float,
    generators: Sequence[str] | None = None,
    transform_levels: Sequence[int] | None = None,
    min_group_samples: int = 20,
    transform_families: Sequence[str] | None = None,
    source_datasets: Sequence[str] | None = None,
) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.shape != scores_array.shape or labels_array.ndim != 1:
        raise ValueError(
            "labels and scores must be one-dimensional arrays of equal length"
        )
    if labels_array.size == 0:
        raise ValueError("Cannot compute metrics for an empty dataset")
    result: dict[str, Any] = _classification_metrics(
        labels_array, scores_array, threshold
    )
    result["threshold"] = float(threshold)

    if generators is None or transform_levels is None:
        return result
    if (
        len(generators) != labels_array.size
        or len(transform_levels) != labels_array.size
    ):
        raise ValueError("Group metadata length does not match predictions")

    grouped: dict[str, dict[str, list[int]]] = {
        "transform_level": defaultdict(list),
        "generator": defaultdict(list),
        "generator_x_level": defaultdict(list),
    }
    has_families = transform_families is not None and any(
        family != "unknown" for family in transform_families
    )
    has_sources = source_datasets is not None
    if transform_families is not None and len(transform_families) != labels_array.size:
        raise ValueError("Transform-family metadata length does not match predictions")
    if source_datasets is not None and len(source_datasets) != labels_array.size:
        raise ValueError("Source metadata length does not match predictions")
    if has_families:
        grouped["transform_family"] = defaultdict(list)
        grouped["transform_family_x_level"] = defaultdict(list)
    if has_sources:
        grouped["source_dataset"] = defaultdict(list)
        grouped["source_x_level"] = defaultdict(list)

    for index, (generator, level) in enumerate(zip(generators, transform_levels)):
        grouped["transform_level"][str(level)].append(index)
        grouped["generator"][str(generator)].append(index)
        grouped["generator_x_level"][f"{generator}/level_{level}"].append(index)
        if has_families:
            family = str(transform_families[index])
            grouped["transform_family"][family].append(index)
            grouped["transform_family_x_level"][f"{family}/level_{level}"].append(index)
        if has_sources:
            source = str(source_datasets[index])
            grouped["source_dataset"][source].append(index)
            grouped["source_x_level"][f"{source}/level_{level}"].append(index)

    all_group_aucs: list[float] = []
    level_aucs: list[float] = []
    result["groups"] = {}
    for category, groups in grouped.items():
        category_metrics: dict[str, Any] = {}
        for name, indices in sorted(groups.items()):
            index_array = np.asarray(indices)
            metrics = _classification_metrics(
                labels_array[index_array], scores_array[index_array], threshold
            )
            category_metrics[name] = metrics
            auc = float(metrics["roc_auc"])
            if len(indices) >= min_group_samples and np.isfinite(auc):
                robust_category = (
                    "transform_family_x_level" if has_families else "transform_level"
                )
                if category == robust_category:
                    level_aucs.append(auc)
                if category == "generator_x_level":
                    all_group_aucs.append(auc)
        result["groups"][category] = category_metrics
    result["robust_auc"] = float(np.mean(level_aucs)) if level_aucs else float("nan")
    result["robust_grouping"] = (
        "transform_family_x_level" if has_families else "transform_level"
    )
    result["worst_group_auc"] = (
        float(min(all_group_aucs)) if all_group_aucs else float("nan")
    )
    return result
