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
