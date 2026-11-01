"""Held-out calibration audit on sklearn's Breast Cancer Wisconsin dataset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.calibration_audit import calibration_audit


def run_seed(seed: int) -> list[dict]:
    data = load_breast_cancer()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        data.data, data.target, test_size=0.20, stratify=data.target, random_state=seed
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=seed + 1000
    )
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
    }
    records = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        cal_p = model.predict_proba(X_cal)[:, 1]
        test_p = model.predict_proba(X_test)[:, 1]
        sigmoid = LogisticRegression(max_iter=1000, random_state=seed)
        sigmoid.fit(cal_p.reshape(-1, 1), y_cal)
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(cal_p, y_cal)
        variants = {
            "raw": test_p,
            "sigmoid": sigmoid.predict_proba(test_p.reshape(-1, 1))[:, 1],
            "isotonic": isotonic.predict(test_p),
        }
        for method, probabilities in variants.items():
            records.append({
                "seed": seed, "model": model_name, "calibration": method,
                **calibration_audit(y_test, probabilities),
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--output", default="results/calibration_safety")
    args = parser.parse_args()
    frame = pd.DataFrame([row for seed in args.seeds for row in run_seed(seed)])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "calibration_safety_by_seed.csv", index=False)
    summary = frame.groupby(["model", "calibration"]).agg(
        ece_mean=("ece", "mean"), ece_std=("ece", "std"),
        adaptive_ece_mean=("adaptive_ece", "mean"), brier_mean=("brier_score", "mean"),
        log_loss_mean=("log_loss", "mean"), top_label_ece_mean=("top_label_ece", "mean"),
        high_confidence_error_rate_mean=("high_confidence_error_rate", "mean"),
        risk_coverage_auc_mean=("risk_coverage_auc", "mean"),
    ).reset_index()
    summary.to_csv(output / "calibration_safety_summary.csv", index=False)
    (output / "calibration_safety_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2) + "\\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
