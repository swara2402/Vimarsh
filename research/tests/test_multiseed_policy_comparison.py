import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "experiments" / "run_multiseed_policy_comparison.py"
SPEC = importlib.util.spec_from_file_location("multiseed_runner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_result(seed, coverage, error_rate, corrections):
    return {
        "seed": seed,
        "policy_comparisons": {
            policy: {
                "coverage": coverage,
                "observed_error_rate": error_rate,
                "net_corrections": corrections,
            }
            for policy in MODULE.POLICIES
        },
    }


def test_summary_reports_mean_population_std_and_per_seed_values():
    summary = MODULE.summarize([
        make_result(17, 0.2, 0.1, 1),
        make_result(42, 0.4, 0.2, -1),
    ])
    reliability = summary["reliability_driven"]
    assert reliability["coverage"]["mean"] == pytest.approx(0.3)
    assert reliability["coverage"]["population_std"] == pytest.approx(0.1)
    assert reliability["coverage"]["values_by_seed"] == {"17": 0.2, "42": 0.4}
    assert reliability["net_corrections"]["mean"] == 0.0


def test_summary_preserves_missing_metric_values():
    first = make_result(17, 0.2, None, 0)
    second = make_result(42, 0.4, 0.2, 0)
    summary = MODULE.summarize([first, second])
    metric = summary["confidence_only"]["observed_error_rate"]
    assert metric["n"] == 1
    assert metric["mean"] == 0.2
    assert metric["values_by_seed"] == {"17": None, "42": 0.2}
