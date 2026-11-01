"""Paired bootstrap uncertainty for selective-policy comparisons.

Intervals are conditional on fitted models, validation-selected thresholds, and
the observed test partition. They quantify test-sample uncertainty only.
"""
import numpy as np
from .threshold_diagnostics import select_validation_threshold


def paired_bootstrap_selective_comparison(
    reference_accepted,
    candidate_accepted,
    test_predictions,
    test_labels,
    groups=None,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=42,
):
    """Estimate paired coverage and accepted-error differences.

    Differences are candidate minus reference. Both policies use the same
    resampled observations or duplicate-feature clusters. Threshold selection
    must happen before calling this function, using validation scores only.
    """
    reference = np.asarray(reference_accepted, dtype=bool)
    candidate = np.asarray(candidate_accepted, dtype=bool)
    predictions = np.asarray(test_predictions)
    labels = np.asarray(test_labels)
    arrays = (reference, candidate, predictions, labels)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Acceptance masks, predictions, and labels must be one-dimensional")
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Acceptance masks, predictions, and labels must have equal non-zero lengths")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 100:
        raise ValueError("n_bootstrap must be an integer of at least 100")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    if groups is None:
        cluster_ids = np.arange(len(reference))
        resampling_unit = "test_observation"
    else:
        cluster_ids = np.asarray(groups)
        if cluster_ids.ndim != 1 or len(cluster_ids) != len(reference):
            raise ValueError("groups must be one-dimensional and aligned with test observations")
        try:
            if np.asarray(cluster_ids != cluster_ids).any():
                raise ValueError("groups must not contain missing values")
        except TypeError:
            pass
        resampling_unit = "supplied_group"

    unique_groups, inverse = np.unique(cluster_ids, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique_groups))]
    if len(members) < 2:
        raise ValueError("At least two resampling groups are required")

    errors = (predictions != labels).astype(float)
    reference_error = _accepted_error(reference, errors)
    candidate_error = _accepted_error(candidate, errors)
    rng = np.random.default_rng(random_state)
    coverage_differences = np.empty(n_bootstrap, dtype=float)
    error_differences = []

    for replicate in range(n_bootstrap):
        sampled_groups = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[index] for index in sampled_groups])
        ref_mask = reference[indices]
        cand_mask = candidate[indices]
        coverage_differences[replicate] = float(cand_mask.mean() - ref_mask.mean())
        ref_rate = _accepted_error(ref_mask, errors[indices])
        cand_rate = _accepted_error(cand_mask, errors[indices])
        if ref_rate is not None and cand_rate is not None:
            error_differences.append(cand_rate - ref_rate)

    alpha = (1.0 - confidence_level) / 2.0
    coverage_ci = np.quantile(coverage_differences, [alpha, 1.0 - alpha])
    error_ci = (
        np.quantile(error_differences, [alpha, 1.0 - alpha]).tolist()
        if error_differences else None
    )
    return {
        "difference_direction": "candidate_minus_reference",
        "reference_coverage": float(reference.mean()),
        "candidate_coverage": float(candidate.mean()),
        "coverage_difference": float(candidate.mean() - reference.mean()),
        "coverage_difference_ci": coverage_ci.tolist(),
        "reference_accepted_error_rate": reference_error,
        "candidate_accepted_error_rate": candidate_error,
        "accepted_error_rate_difference": (
            candidate_error - reference_error
            if candidate_error is not None and reference_error is not None
            else None
        ),
        "accepted_error_rate_difference_ci": error_ci,
        "bootstrap_replicates": int(n_bootstrap),
        "valid_error_replicates": int(len(error_differences)),
        "confidence_level": float(confidence_level),
        "resampling_unit": resampling_unit,
    }


def paired_bootstrap_policy_comparisons(
    validation_scores_by_policy,
    test_scores_by_policy,
    test_predictions,
    test_labels,
    reference_policy="confidence_only",
    target_coverage=0.75,
    groups=None,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=42,
):
    """Compare validation-thresholded policies against one reference policy."""
    if reference_policy not in validation_scores_by_policy:
        raise ValueError("reference_policy is missing validation scores")
    if reference_policy not in test_scores_by_policy:
        raise ValueError("reference_policy is missing test scores")
    if set(validation_scores_by_policy) != set(test_scores_by_policy):
        raise ValueError("Validation and test score policy names must match")

    accepted_masks = {}
    thresholds = {}
    for policy in validation_scores_by_policy:
        accepted_masks[policy], thresholds[policy] = select_validation_threshold(
            validation_scores_by_policy[policy],
            test_scores_by_policy[policy],
            target_coverage=target_coverage,
        )

    results = {}
    for index, policy in enumerate(validation_scores_by_policy):
        if policy == reference_policy:
            continue
        results[policy] = {
            "reference_policy": reference_policy,
            "reference_threshold": float(thresholds[reference_policy]),
            "candidate_threshold": float(thresholds[policy]),
            **paired_bootstrap_selective_comparison(
                accepted_masks[reference_policy],
                accepted_masks[policy],
                test_predictions,
                test_labels,
                groups=groups,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=random_state + index,
            ),
        }
    return results


def _accepted_error(mask, errors):
    return float(errors[mask].mean()) if np.any(mask) else None
