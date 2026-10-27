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
from fraudsense.decision.router import DecisionRouter
from fraudsense.features.service import FeatureService
from fraudsense.feedback.reward import RewardCalculator
from fraudsense.graph.builder import DynamicGraphBuilder
from fraudsense.ingest.paysim_loader import load_paysim
from fraudsense.models.baselines import BaselineA, BaselineB
from fraudsense.models.conformal import SplitConformalCalibrator
from fraudsense.models.dqn import DQNActionPolicy
from fraudsense.models.gat import GATEncoder
from fraudsense.models.risk_head import CalibratedRiskHead
from fraudsense.policy.mask import PolicyMask


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


def run_phase3_evaluation(n_samples: int = 50000) -> dict[str, Any]:
    """
    Trains and benchmarks GAT Encoder, Calibrated Risk Head, and Split Conformal Calibrator.
    Reports empirical conformal coverage, mean set size, and operational cost.
    """
    print(
        f"\n--- Phase 3: GAT + Calibrated Risk Head + Conformal Calibrator ({n_samples} events) ---"
    )
    events = load_paysim(n_rows=n_samples)

    print("Extracting tabular features and building dynamic transaction graph...")
    feat_service = FeatureService()
    X_tab, _ = feat_service.extract_batch(events)
    y = np.array([e.label for e in events], dtype=int)
    amounts = np.array([e.amount for e in events], dtype=float)

    graph_builder = DynamicGraphBuilder()
    for e in events:
        graph_builder.add_event(e)

    pyg_data = graph_builder.build_pyg_data()
    print(f"Graph constructed: {pyg_data.num_nodes} nodes, {pyg_data.edge_index.shape[1]} edges.")

    print("Encoding account topology with 2-layer 4-head GAT (dim=64)...")
    gat = GATEncoder(in_channels=4, hidden_channels=16, out_channels=64, heads=4)
    gat.eval()
    with __import__("torch").no_grad():
        node_embeds = gat(pyg_data.x, pyg_data.edge_index, pyg_data.edge_weight).numpy()

    # Create transaction-level GAT embeddings: lookup payer embedding
    gat_embeds = []
    for e in events:
        p_idx = graph_builder.node_to_idx.get(e.payer_id, 0)
        gat_embeds.append(node_embeds[p_idx])
    X_gat = np.array(gat_embeds, dtype=np.float32)

    # Concatenate [X_tab (18), X_gat (64)] -> (N, 82)
    X_full = np.hstack([X_gat, X_tab])

    # Time-ordered split: 60% train, 20% calibration, 20% test
    n = len(events)
    train_end = int(n * 0.6)
    cal_end = int(n * 0.8)

    X_train, y_train, amt_train = X_full[:train_end], y[:train_end], amounts[:train_end]
    X_cal, y_cal, amt_cal = (
        X_full[train_end:cal_end],
        y[train_end:cal_end],
        amounts[train_end:cal_end],
    )
    X_test, y_test, amt_test = X_full[cal_end:], y[cal_end:], amounts[cal_end:]

    print("Fitting Risk Head MLP and calibrating Temperature Scaling on calibration split...")
    risk_head = CalibratedRiskHead(in_features=82)
    risk_head.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_cal,
        y_val=y_cal,
        amounts_train=amt_train,
        amounts_val=amt_cal,
        epochs=5,
    )

    probs_cal = risk_head.predict_multi_proba(X_cal)
    y_cal_multi = risk_head._prepare_labels(y_cal, amt_cal).numpy()

    print("Calibrating Split Conformal Predictor (target coverage = 0.90)...")
    conformal = SplitConformalCalibrator()
    q_thresh = conformal.calibrate(probs_cal, y_cal_multi)

    print("Evaluating Conformal Prediction Sets on held-out test split...")
    probs_test = risk_head.predict_multi_proba(X_test)
    y_test_multi = risk_head._prepare_labels(y_test, amt_test).numpy()
    p_fraud_test = probs_test[:, 2]

    conformal_metrics = conformal.evaluate_coverage(probs_test, y_test_multi)
    pred_sets = conformal.predict_batch(probs_test)

    # Evaluate decision logic using conformal sets
    reward_calc = RewardCalculator()
    actions = []
    for s, amt, p_val in zip(pred_sets, amt_test, p_fraud_test):
        # Section 5 decision rules:
        # 1. High value (>= 50,000): at least CHALLENGE; becomes HOLD if set includes FRAUD
        if amt >= 50000:
            if "FRAUD" in s or p_val >= 0.5:
                actions.append("HOLD_FOR_REVIEW")
            else:
                actions.append("CHALLENGE")
        # 2. C = {CHALLENGE} or contains both FRAUD and LEGITIMATE outcomes
        elif s == {"CHALLENGE"} or ("FRAUD" in s and "ALLOW" in s):
            if len(s) >= 2 and "FRAUD" in s:
                actions.append("HOLD_FOR_REVIEW")
            else:
                actions.append("CHALLENGE")
        elif "FRAUD" in s:
            actions.append("HOLD_FOR_REVIEW")
        elif s == {"ALLOW"}:
            actions.append("ALLOW")
        else:
            actions.append("CHALLENGE")

    cost_metrics = reward_calc.evaluate_batch(actions, list(y_test), list(amt_test))
    pr_auc_head = float(average_precision_score(y_test, p_fraud_test)) if sum(y_test) > 0 else 0.0

    act_dist = {act: round(float(actions.count(act) / len(actions)), 4) for act in set(actions)}

    results = {
        "model": "GAT + Calibrated Risk Head + Split Conformal",
        "test_events": len(y_test),
        "test_frauds": int(sum(y_test)),
        "risk_head": {
            "pr_auc": round(pr_auc_head, 4),
            "learned_temperature": round(float(risk_head.scaler.temperature.item()), 4),
        },
        "conformal": {
            "target_coverage": conformal_metrics["target_coverage"],
            "empirical_coverage": round(conformal_metrics["empirical_coverage"], 4),
            "coverage_gap": round(
                conformal_metrics["empirical_coverage"] - conformal_metrics["target_coverage"], 4
            ),
            "mean_set_size": round(conformal_metrics["mean_set_size"], 4),
            "singleton_rate": round(conformal_metrics["singleton_rate"], 4),
            "wide_set_rate": round(conformal_metrics["wide_set_rate"], 4),
            "quantile_threshold": round(q_thresh, 4),
        },
        "decision_evaluation": {
            "cost_per_10k": round(cost_metrics["cost_per_10k"], 2),
            "total_reward": round(cost_metrics["total_reward"], 2),
            "action_distribution": act_dist,
        },
    }

    reports_dir = get_project_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "phase3_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Phase 3 evaluation complete! Results saved to {out_file}")
    return results


def run_phase4_evaluation(n_samples: int = 50000) -> dict[str, Any]:
    """
    Evaluates all 4 models: Baseline A, Baseline B, Risk Head alone, and Full Pipeline (GAT + Conformal + DQN + Mask).
    Reports cost per 10,000 payments for every model under identical operational parameters.
    """
    print("\n========================================================")
    print(f"--- Phase 4: Full Multi-Model Comparison ({n_samples} events) ---")
    print("========================================================")

    # 1. Evaluate Baselines
    res_base = run_baseline_evaluation(n_samples=n_samples)
    base_a_metrics = res_base["baseline_a"]
    base_b_metrics = res_base["baseline_b"]

    # 2. Setup Data for Deep Pipelines
    events = load_paysim(n_rows=n_samples)
    feat_service = FeatureService()
    X_tab, _ = feat_service.extract_batch(events)
    y = np.array([e.label for e in events], dtype=int)
    amounts = np.array([e.amount for e in events], dtype=float)

    graph_builder = DynamicGraphBuilder()
    for e in events:
        graph_builder.add_event(e)
    pyg_data = graph_builder.build_pyg_data()

    gat = GATEncoder(in_channels=4, hidden_channels=16, out_channels=64, heads=4)
    gat.eval()
    with __import__("torch").no_grad():
        node_embeds = gat(pyg_data.x, pyg_data.edge_index, pyg_data.edge_weight).numpy()

    gat_embeds = [node_embeds[graph_builder.node_to_idx.get(e.payer_id, 0)] for e in events]
    X_full = np.hstack([np.array(gat_embeds, dtype=np.float32), X_tab])

    n = len(events)
    train_end = int(n * 0.6)
    cal_end = int(n * 0.8)

    X_train, y_train, amt_train = X_full[:train_end], y[:train_end], amounts[:train_end]
    X_cal, y_cal, amt_cal = (
        X_full[train_end:cal_end],
        y[train_end:cal_end],
        amounts[train_end:cal_end],
    )
    X_test, y_test, amt_test = X_full[cal_end:], y[cal_end:], amounts[cal_end:]
    events_test = events[cal_end:]
    X_tab_test = X_tab[cal_end:]
    X_gat_test = np.array(gat_embeds[cal_end:], dtype=np.float32)

    risk_head = CalibratedRiskHead(in_features=82)
    risk_head.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_cal,
        y_val=y_cal,
        amounts_train=amt_train,
        amounts_val=amt_cal,
        epochs=5,
    )

    conformal = SplitConformalCalibrator()
    probs_cal = risk_head.predict_multi_proba(X_cal)
    y_cal_multi = risk_head._prepare_labels(y_cal, amt_cal).numpy()
    conformal.calibrate(probs_cal, y_cal_multi)

    reward_calc = RewardCalculator()

    # 3. Model 3: Risk Head Alone (Probability Thresholding)
    probs_test = risk_head.predict_multi_proba(X_test)
    p_fraud_test = probs_test[:, 2]
    actions_risk_head = []
    for p_val, amt in zip(p_fraud_test, amt_test):
        if p_val >= 0.55:
            actions_risk_head.append("HOLD_FOR_REVIEW")
        elif p_val >= 0.20 or amt >= 50000:
            actions_risk_head.append("CHALLENGE")
        else:
            actions_risk_head.append("ALLOW")

    cost_risk_head = reward_calc.evaluate_batch(actions_risk_head, list(y_test), list(amt_test))
    act_dist_head = {
        act: round(float(actions_risk_head.count(act) / len(actions_risk_head)), 4)
        for act in set(actions_risk_head)
    }

    # 4. Model 4: Full Pipeline (GAT + Conformal + DQN + Mask + Override)
    print("Training DQN Policy with policy mask on historical training split...")
    dqn = DQNActionPolicy(state_dim=7)
    policy_mask = PolicyMask()

    # Create historical DQN training tuples
    dqn_states = []
    allowed_list = []
    probs_train = risk_head.predict_multi_proba(X_train)
    p_train = probs_train[:, 2]
    sets_train = conformal.predict_batch(probs_train)

    for s, amt, p_val, tab in zip(sets_train, amt_train, p_train, X_tab[:train_end]):
        dqn_states.append(dqn.encode_state(s, amt, p_val, tab))
        allowed, _ = policy_mask.get_allowed_actions(amount=amt)
        allowed_list.append(allowed)

    dqn.train_on_history(
        states=dqn_states,
        labels=list(y_train),
        amounts=list(amt_train),
        allowed_actions_list=allowed_list,
        epochs=3,
        batch_size=64,
    )

    print("Evaluating Full Pipeline with DecisionRouter...")
    router = DecisionRouter(
        risk_head=risk_head,
        conformal_calibrator=conformal,
        dqn_policy=dqn,
    )

    actions_full = []
    latencies = []
    for evt, tab, gat_e in zip(events_test, X_tab_test, X_gat_test):
        res = router.route(evt, tabular_features=tab, gat_embedding=gat_e)
        actions_full.append(res.action)
        latencies.append(res.latency_ms)

    cost_full = reward_calc.evaluate_batch(actions_full, list(y_test), list(amt_test))
    act_dist_full = {
        act: round(float(actions_full.count(act) / len(actions_full)), 4)
        for act in set(actions_full)
    }

    p50_lat = float(np.percentile(latencies, 50))
    p99_lat = float(np.percentile(latencies, 99))

    comparison = {
        "dataset": "PaySim",
        "test_events": len(y_test),
        "test_frauds": int(sum(y_test)),
        "models": {
            "baseline_a": {
                "name": "Baseline A (GBDT Tabular)",
                "cost_per_10k": base_a_metrics["cost_per_10k"],
                "total_reward": base_a_metrics["total_reward"],
                "action_distribution": base_a_metrics["action_distribution"],
            },
            "baseline_b": {
                "name": "Baseline B (Bayes-Optimal Cost-Sensitive)",
                "cost_per_10k": base_b_metrics["cost_per_10k"],
                "total_reward": base_b_metrics["total_reward"],
                "action_distribution": base_b_metrics["action_distribution"],
            },
            "risk_head_alone": {
                "name": "Risk Head Alone (GAT + MLP + Calibrated Thresholds)",
                "cost_per_10k": round(cost_risk_head["cost_per_10k"], 2),
                "total_reward": round(cost_risk_head["total_reward"], 2),
                "action_distribution": act_dist_head,
            },
            "full_pipeline": {
                "name": "Full Pipeline (GAT + Conformal + DQN + Mask + Override)",
                "cost_per_10k": round(cost_full["cost_per_10k"], 2),
                "total_reward": round(cost_full["total_reward"], 2),
                "action_distribution": act_dist_full,
                "latency_p50_ms": round(p50_lat, 3),
                "latency_p99_ms": round(p99_lat, 3),
            },
        },
    }

    reports_dir = get_project_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    comp_file = reports_dir / "all_models_comparison.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nAll models comparison saved to {comp_file}")
    print("\n--- Summary Cost per 10,000 Payments ---")
    for v in comparison["models"].values():
        print(f"  {v['name']}: Cost/10k = INR {v['cost_per_10k']:,.2f}")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FraudSense Baselines & Models")
    parser.add_argument("--samples", type=int, default=50000, help="Number of samples to evaluate")
    parser.add_argument(
        "--phase",
        type=int,
        default=4,
        help="Phase evaluation: 2 for baselines, 3 for conformal, 4 for full comparison",
    )
    args = parser.parse_args()

    if args.phase == 2:
        run_baseline_evaluation(n_samples=args.samples)
    elif args.phase == 3:
        run_phase3_evaluation(n_samples=args.samples)
    else:
        run_phase4_evaluation(n_samples=args.samples)
