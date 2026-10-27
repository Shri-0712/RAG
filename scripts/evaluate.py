"""
Evaluation script for FraudSense models.
Computes PR-AUC, recall at fixed false-hold ceilings (0.5%, 1.0%, 2.0%),
expected operational cost per 10,000 payments, and threshold sensitivity curves.
Saves all artifacts to reports/.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraudsense.config import get_project_root
from fraudsense.features.service import FeatureService
from fraudsense.feedback.reward import RewardCalculator
from fraudsense.ingest.paysim_loader import load_paysim
from fraudsense.models.baselines import BaselineA, BaselineB


def compute_recall_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fprs: list[float] | None = None,
) -> dict[str, float]:
    """
    Computes recall (true positive rate) at fixed false positive (false-hold) rates.
    """
    if target_fprs is None:
        target_fprs = [0.005, 0.01, 0.02]
    negatives = y_score[y_true == 0]
    positives = y_score[y_true == 1]

    if len(negatives) == 0 or len(positives) == 0:
        return {f"recall_at_fpr_{int(fpr * 1000)}bp": 0.0 for fpr in target_fprs}

    results = {}
    for fpr in target_fprs:
        # Quantile threshold on negative distribution
        thresh = float(np.percentile(negatives, (1.0 - fpr) * 100))
        # Recall at this threshold
        recall = float(np.mean(positives >= thresh))
        key = f"recall_at_fpr_{fpr * 100:.1f}pct"
        results[key] = recall

    return results


def compute_threshold_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Computes hold rate vs missed-fraud rate across a range of decision thresholds.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    records = []
    n_fraud = int(np.sum(y_true))

    for th in thresholds:
        holds = (y_score >= th).astype(int)
        hold_rate = float(np.mean(holds))

        # Missed fraud: actual fraud but not held
        missed_fraud = (y_true == 1) & (holds == 0)
        missed_fraud_rate = float(np.sum(missed_fraud) / max(1, n_fraud))

        records.append(
            {
                "threshold": round(float(th), 3),
                "hold_rate": round(hold_rate, 4),
                "missed_fraud_rate": round(missed_fraud_rate, 4),
            }
        )

    return pd.DataFrame(records)


def run_baseline_evaluation(n_samples: int = 40000) -> dict[str, Any]:
    """
    Trains and benchmarks Baseline A (GBDT) and Baseline B (Cost-Sensitive)
    on time-ordered PaySim split.
    """
    print(f"Loading {n_samples} PaySim events for baseline benchmark...")
    events = load_paysim(n_rows=n_samples)

    print("Extracting causal streaming features...")
    feat_service = FeatureService()
    X, _ = feat_service.extract_batch(events)
    y = np.array([e.label for e in events], dtype=int)
    amounts = np.array([e.amount for e in events], dtype=float)

    # Time-ordered split: 60% train, 20% validation, 20% test
    n = len(events)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val, amt_val = (
        X[train_end:val_end],
        y[train_end:val_end],
        amounts[train_end:val_end],
    )
    X_test, y_test, amt_test = X[val_end:], y[val_end:], amounts[val_end:]

    print(
        f"Split sizes: Train={len(y_train)} (Frauds={sum(y_train)}), "
        f"Val={len(y_val)} (Frauds={sum(y_val)}), "
        f"Test={len(y_test)} (Frauds={sum(y_test)})"
    )

    reward_calc = RewardCalculator()

    # 1. Train Baseline A
    print("Training Baseline A (GBDT on tabular features)...")
    base_a = BaselineA()
    base_a.fit(X_train, y_train, X_val, y_val, amt_val)
    probs_a = base_a.predict_proba(X_test)
    actions_a = base_a.predict_actions(X_test)
    cost_a = reward_calc.evaluate_batch(actions_a, list(y_test), list(amt_test))

    pr_auc_a = float(average_precision_score(y_test, probs_a)) if sum(y_test) > 0 else 0.0
    recalls_a = compute_recall_at_fpr(y_test, probs_a)

    # 2. Train Baseline B (Cost-Sensitive)
    print("Evaluating Baseline B (Cost-Sensitive Classifier)...")
    base_b = BaselineB()
    base_b.base_model = base_a  # Share fitted probability estimator
    actions_b = base_b.predict_actions(X_test, amt_test)
    cost_b = reward_calc.evaluate_batch(actions_b, list(y_test), list(amt_test))

    # Action distributions
    act_dist_a = {
        act: round(float(actions_a.count(act) / len(actions_a)), 4) for act in set(actions_a)
    }
    act_dist_b = {
        act: round(float(actions_b.count(act) / len(actions_b)), 4) for act in set(actions_b)
    }

    # Sensitivity table for test split
    sensitivity_df = compute_threshold_sensitivity(y_test, probs_a)

    results = {
        "dataset": "PaySim",
        "test_events": len(y_test),
        "test_frauds": int(sum(y_test)),
        "baseline_a": {
            "name": "Baseline A (GBDT Tabular with Tuned Thresholds)",
            "pr_auc": round(pr_auc_a, 4),
            "recall_at_fpr": recalls_a,
            "cost_per_10k": round(cost_a["cost_per_10k"], 2),
            "total_reward": round(cost_a["total_reward"], 2),
            "action_distribution": act_dist_a,
            "tuned_threshold_challenge": base_a.threshold_challenge,
            "tuned_threshold_hold": base_a.threshold_hold,
        },
        "baseline_b": {
            "name": "Baseline B (Bayes-Optimal Cost-Sensitive without Graph)",
            "cost_per_10k": round(cost_b["cost_per_10k"], 2),
            "total_reward": round(cost_b["total_reward"], 2),
            "action_distribution": act_dist_b,
        },
        "sensitivity_curve": sensitivity_df.to_dict(orient="records"),
    }

    # Save to reports/
    reports_dir = get_project_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "baseline_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    sens_path = reports_dir / "threshold_sensitivity.csv"
    sensitivity_df.to_csv(sens_path, index=False)

    print(f"Metrics saved to {json_path}")
    print(f"Sensitivity table saved to {sens_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FraudSense Baselines")
    parser.add_argument("--samples", type=int, default=40000, help="Number of samples to evaluate")
    args = parser.parse_args()
    run_baseline_evaluation(n_samples=args.samples)
