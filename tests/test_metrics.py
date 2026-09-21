import numpy as np
import pytest

from sbe.metrics import detection_summary, paired_split_indices, tpr_at_fpr


def test_paired_split_is_disjoint_complete_and_deterministic():
    first = paired_split_indices(100, seed=7)
    second = paired_split_indices(100, seed=7)

    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.test, second.test)
    combined = np.concatenate([first.train, first.validation, first.test])
    assert len(np.unique(combined)) == 100
    assert set(combined) == set(range(100))


def test_split_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        paired_split_indices(100, train_fraction=0.8, validation_fraction=0.2)


def test_tpr_at_fpr_uses_empirical_threshold():
    negatives = np.array([0.0, 0.1, 0.2, 0.3])
    positives = np.array([0.4, 0.5, 0.6, 0.7])
    assert tpr_at_fpr(negatives, positives, target_fpr=0.01) == 1.0


def test_summary_exposes_low_fpr_sample_size_limit():
    negatives = np.linspace(0.0, 0.4, 100)
    positives = np.linspace(0.6, 1.0, 100)
    summary = detection_summary(
        negatives,
        positives,
        target_fpr=0.01,
        bootstrap_replicates=20,
        seed=3,
    )

    assert summary["auroc"] == 1.0
    assert summary["empirical_tpr_at_target_fpr"] == 1.0
    assert summary["zero_false_positive_rate_one_sided_95_upper"] > 0.01
    assert summary["sample_size_supports_target_fpr_at_zero_fp"] is False
