import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "run_primary_model_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("primary_model_sensitivity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(accuracy, error, coverage, accepted_error):
    return {
        "test_accuracy": accuracy,
        "test_error_rate": error,
        "validation_selected_nominal_75pct_coverage": {
            "reliability_driven": {
                "test_coverage": coverage,
                "observed_error_rate": accepted_error,
            },
            "confidence_only": {
                "test_coverage": coverage,
                "observed_error_rate": accepted_error + 0.01,
            },
            "predictive_entropy": {
                "test_coverage": coverage - 0.01,
                "observed_error_rate": accepted_error - 0.01,
            },
            "ensemble_disagreement": {
                "test_coverage": coverage + 0.02,
                "observed_error_rate": accepted_error + 0.02,
            },
        },
    }


def test_summary_keeps_primary_model_families_separate():
    results = []
    for seed in (17, 42, 73):
        results.append({
            "seed": seed,
            "primary_model_results": {
                "logistic_regression": _row(0.8, 0.2, 0.75, 0.1),
                "random_forest": _row(0.9, 0.1, 0.76, 0.08),
                "gradient_boosting": _row(0.85, 0.15, 0.74, 0.09),
            },
        })

    summary = MODULE.summarize_seed_results(results)

    assert set(summary) == {
        "logistic_regression", "random_forest", "gradient_boosting"
    }
    assert summary["random_forest"]["test_accuracy"]["mean"] == 0.9
    assert summary["random_forest"]["test_accuracy"]["population_sd"] == 0.0
    assert summary["logistic_regression"]["matched_coverage_policies"][
        "reliability_driven"
    ]["coverage_mean"] == 0.75


def test_summary_reports_population_sd_for_seed_variation():
    results = []
    for seed, accuracy in zip((17, 42, 73), (0.8, 0.9, 1.0)):
        row = _row(accuracy, 1.0 - accuracy, 0.75, 0.1)
        results.append({
            "seed": seed,
            "primary_model_results": {
                name: row for name in (
                    "logistic_regression", "random_forest", "gradient_boosting"
                )
            },
        })

    summary = MODULE.summarize_seed_results(results)
    assert abs(
        summary["logistic_regression"]["test_accuracy"]["population_sd"]
        - (2.0 / 3.0) ** 0.5 * 0.1
    ) < 1e-12
