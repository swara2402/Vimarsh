"""Leakage and coverage checks for the MAGIC benchmark protocol."""
import numpy as np
import pandas as pd

from experiments.run_magic_gamma_policy_comparison import grouped_magic_split


def test_grouped_magic_split_keeps_duplicate_features_together():
    rng = np.random.default_rng(12)
    unique = rng.normal(size=(120, 4))
    X = pd.DataFrame(np.repeat(unique, 2, axis=0), columns=list("abcd"))
    y = pd.Series(np.tile(np.repeat([0, 1], 60), 2))
    train, validation, test = grouped_magic_split(X, y, seed=17)

    groups = pd.MultiIndex.from_frame(X).factorize(sort=False)[0]
    partitions = [set(groups[idx]) for idx in (train, validation, test)]
    assert not (partitions[0] & partitions[1])
    assert not (partitions[0] & partitions[2])
    assert not (partitions[1] & partitions[2])
    assert sorted(np.concatenate([train, validation, test]).tolist()) == list(range(len(X)))


def test_grouped_magic_split_is_deterministic_for_seed():
    rng = np.random.default_rng(21)
    X = pd.DataFrame(rng.normal(size=(300, 5)))
    y = pd.Series(np.tile([0, 1], 150))
    first = grouped_magic_split(X, y, seed=42)
    second = grouped_magic_split(X, y, seed=42)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


from experiments.run_magic_gamma_policy_comparison import validation_threshold_metrics


def test_validation_threshold_is_selected_without_using_test_labels():
    validation_scores = np.arange(100, dtype=float) / 100.0
    test_scores = np.array([0.1, 0.2, 0.8, 0.9])
    predictions = np.array([0, 0, 1, 1])
    labels = np.array([0, 1, 1, 0])

    result = validation_threshold_metrics(
        validation_scores, test_scores, predictions, labels, target_coverage=0.5
    )

    assert result["validation_selected_threshold"] == 0.5
    assert result["target_coverage"] == 0.5
    assert result["test_coverage"] == 0.5
    assert result["accepted_test_observations"] == 2
    assert result["observed_error_rate"] == 0.5


def test_validation_threshold_rejects_nonfinite_scores():
    with np.testing.assert_raises(ValueError):
        validation_threshold_metrics(
            np.array([0.1, np.nan]), np.array([0.2]), np.array([0]), np.array([0])
        )
