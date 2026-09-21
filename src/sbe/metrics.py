"""Statistical helpers for detector evaluation.

The helpers keep model execution separate from score analysis so that operating
points, uncertainty, and data splits can be unit tested without loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def paired_split_indices(
    n_items: int,
    *,
    train_fraction: float = 0.5,
    validation_fraction: float = 0.25,
    seed: int = 0,
) -> SplitIndices:
    """Split item IDs once and reuse them across every prompt condition."""
    if n_items < 12:
        raise ValueError("At least 12 paired items are required")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_items)
    train_end = int(n_items * train_fraction)
    validation_end = train_end + int(n_items * validation_fraction)
    if train_end == 0 or validation_end == train_end or validation_end == n_items:
        raise ValueError("Split produced an empty partition")
    return SplitIndices(
        train=indices[:train_end],
        validation=indices[train_end:validation_end],
        test=indices[validation_end:],
    )


def tpr_at_fpr(
    negative_scores: np.ndarray,
    positive_scores: np.ndarray,
    *,
    target_fpr: float = 0.01,
) -> float:
    """Return the largest empirical TPR whose empirical FPR is at most target."""
    negatives = np.asarray(negative_scores, dtype=float)
    positives = np.asarray(positive_scores, dtype=float)
    y = np.concatenate([np.zeros(len(negatives)), np.ones(len(positives))])
    scores = np.concatenate([negatives, positives])
    fpr, tpr, _ = roc_curve(y, scores)
    eligible = fpr <= target_fpr
    return float(tpr[eligible].max()) if eligible.any() else 0.0


def detection_summary(
    negative_scores: np.ndarray,
    positive_scores: np.ndarray,
    *,
    target_fpr: float = 0.01,
    bootstrap_replicates: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | bool | list[float]]:
    """Summarize discrimination with paired-bootstrap AUROC uncertainty.

    When conditions use the same questions, paired resampling preserves the
    question-level dependence between negative and positive scores.
    """
    negatives = np.asarray(negative_scores, dtype=float)
    positives = np.asarray(positive_scores, dtype=float)
    if negatives.ndim != 1 or positives.ndim != 1:
        raise ValueError("Detector scores must be one-dimensional")
    if not len(negatives) or not len(positives):
        raise ValueError("Both score classes must be non-empty")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates cannot be negative")

    y = np.concatenate([np.zeros(len(negatives)), np.ones(len(positives))])
    scores = np.concatenate([negatives, positives])
    auroc = float(roc_auc_score(y, scores))

    rng = np.random.default_rng(seed)
    bootstrap_aurocs: list[float] = []
    for _ in range(bootstrap_replicates):
        if len(negatives) == len(positives):
            sampled = rng.integers(0, len(negatives), size=len(negatives))
            sampled_negatives = negatives[sampled]
            sampled_positives = positives[sampled]
        else:
            sampled_negatives = rng.choice(negatives, size=len(negatives), replace=True)
            sampled_positives = rng.choice(positives, size=len(positives), replace=True)
        sampled_y = np.concatenate(
            [np.zeros(len(sampled_negatives)), np.ones(len(sampled_positives))]
        )
        sampled_scores = np.concatenate([sampled_negatives, sampled_positives])
        bootstrap_aurocs.append(float(roc_auc_score(sampled_y, sampled_scores)))

    if bootstrap_aurocs:
        auroc_ci = [
            float(np.quantile(bootstrap_aurocs, 0.025)),
            float(np.quantile(bootstrap_aurocs, 0.975)),
        ]
    else:
        auroc_ci = [auroc, auroc]

    # Exact one-sided upper bound after observing zero false positives in an
    # independently evaluated negative sample. This is a sample-size diagnostic,
    # not a claim that the selected threshold achieved zero population FPR.
    zero_fp_upper_95 = float(1 - 0.05 ** (1 / len(negatives)))
    return {
        "n_negative": len(negatives),
        "n_positive": len(positives),
        "auroc": auroc,
        "auroc_paired_bootstrap_95_ci": auroc_ci,
        "target_fpr": target_fpr,
        "empirical_tpr_at_target_fpr": tpr_at_fpr(
            negatives,
            positives,
            target_fpr=target_fpr,
        ),
        "empirical_fpr_resolution": 1 / len(negatives),
        "zero_false_positive_rate_one_sided_95_upper": zero_fp_upper_95,
        "sample_size_supports_target_fpr_at_zero_fp": zero_fp_upper_95 <= target_fpr,
    }
