"""Run VIMARSH's complete grouped Wine Quality experiment across independent seeds.

Each seed retrains the base models and reliability estimator, writes a seed-specific
inference bundle, then runs the same five-policy verification comparison. The final
summary reports mean and population standard deviation across seeds.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = RESEARCH_DIR / "experiments"
RESULTS_DIR = RESEARCH_DIR / "results" / "multiseed"
BUNDLE_DIR = RESEARCH_DIR / "models" / "reliability"

POLICIES = (
    "reliability_driven",
    "confidence_only",
    "predictive_entropy",
    "ensemble_disagreement",
    "always_accept",
)
METRICS = ("coverage", "observed_error_rate", "net_corrections")


def run_seed(seed: int) -> dict:
    """Train fresh models and evaluate all policies for one seed."""
    bundle = BUNDLE_DIR / f"wine_quality_reliability_bundle_seed_{seed}.joblib"
    output_dir = RESULTS_DIR / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, str(EXPERIMENT_DIR / "run_reliability_experiments.py"), "--seed", str(seed)],
        cwd=RESEARCH_DIR,
        check=True,
    )
    if not bundle.exists():
        raise FileNotFoundError(f"Expected seed-specific bundle was not created: {bundle}")

    subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "run_integrated_verification.py"),
            "--bundle",
            str(bundle),
            "--output",
            str(output_dir),
        ],
        cwd=RESEARCH_DIR,
        check=True,
    )

    result_file = output_dir / "integrated_outcome_validation.json"
    with result_file.open(encoding="utf-8") as handle:
        result = json.load(handle)
    result["seed"] = seed
    return result


def summarize(results: list[dict]) -> dict:
    """Aggregate policy metrics without silently dropping missing values."""
    summary = {}
    for policy in POLICIES:
        summary[policy] = {}
        for metric in METRICS:
            values = [
                float(item["policy_comparisons"][policy][metric])
                for item in results
                if item["policy_comparisons"][policy].get(metric) is not None
            ]
            summary[policy][metric] = {
                "n": len(values),
                "mean": sum(values) / len(values) if values else None,
                "population_std": (
                    (sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values)) ** 0.5
                    if values else None
                ),
                "values_by_seed": {
                    str(item["seed"]): item["policy_comparisons"][policy].get(metric)
                    for item in results
                },
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique.")
    if not args.seeds:
        parser.error("Provide at least one seed.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for seed in args.seeds:
        print(f"\\n{'=' * 24} VIMARSH seed {seed} {'=' * 24}", flush=True)
        all_results.append(run_seed(seed))

    report = {
        "experiment": "VIMARSH grouped multi-seed five-policy comparison",
        "dataset": "UCI Wine Quality (red + white)",
        "seeds": args.seeds,
        "seed_count": len(all_results),
        "aggregation": "arithmetic mean and population standard deviation across seeds",
        "policy_summary": summarize(all_results),
        "per_seed_results": all_results,
        "limitations": [
            "All seeds use the same Wine Quality benchmark; this is not cross-domain validation.",
            "Human review is simulated and ground truth is used only for offline outcome evaluation.",
            "Evidence-provider actions are disabled.",
            "Action-effectiveness factors remain configured assumptions, not measured causal effects.",
            "The standard deviation across a small number of seeds is descriptive, not a confidence interval.",
        ],
    }
    output_file = RESULTS_DIR / "multiseed_policy_summary.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\\nMulti-seed comparison complete: {output_file}")
    for policy, metrics in report["policy_summary"].items():
        coverage = metrics["coverage"]
        error = metrics["observed_error_rate"]
        print(
            f"{policy}: coverage={coverage['mean']:.2%} ± {coverage['population_std']:.2%}; "
            f"covered_error={error['mean'] if error['mean'] is not None else float('nan'):.2%}; "
            f"net_corrections_mean={metrics['net_corrections']['mean']}"
        )


if __name__ == "__main__":
    main()
