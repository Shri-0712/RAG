"""Transparent operational and model-quality metrics for FraudSense monitoring."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def population_stability_index(
    reference: Iterable[float], current: Iterable[float], bins: int = 10
) -> float:
    """Computes PSI using reference quantile bins with stable zero handling."""
    ref = np.asarray(list(reference), dtype=float)
    cur = np.asarray(list(current), dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("PSI requires non-empty reference and current samples")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.maximum(ref_counts / len(ref), 1e-6)
    cur_pct = np.maximum(cur_counts / len(cur), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def operational_metrics(
    actions: list[str],
    labels: list[int],
    latencies_ms: list[float],
    prediction_sets: list[set[str]] | None = None,
) -> dict[str, float]:
    """Calculates hold/error, challenge, latency, and conformal coverage measures."""
    if not (len(actions) == len(labels) == len(latencies_ms)):
        raise ValueError("actions, labels, and latencies_ms must have equal lengths")
    n = len(actions)
    if n == 0:
        return {
            "hold_rate": 0.0,
            "false_hold_rate": 0.0,
            "challenge_denial_rate": 0.0,
            "median_time_to_decision_ms": 0.0,
            "p99_latency_ms": 0.0,
            "conformal_coverage": 0.0,
            "label_delay_days": 0.0,
        }
    held = np.array([a in {"HOLD", "HOLD_FOR_REVIEW"} for a in actions])
    challenged = np.array([a == "CHALLENGE" for a in actions])
    labels_a = np.asarray(labels)
    metrics = {
        "hold_rate": float(np.mean(held)),
        "false_hold_rate": float(np.mean(held & (labels_a == 0))),
        "challenge_denial_rate": float(
            np.mean(challenged & (labels_a == 1)) / max(np.mean(challenged), 1e-9)
        ),
        "median_time_to_decision_ms": float(np.median(latencies_ms)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "label_delay_days": 0.0,
    }
    if prediction_sets is None:
        metrics["conformal_coverage"] = 0.0
    else:
        outcomes = ["FRAUD" if label else "ALLOW" for label in labels]
        metrics["conformal_coverage"] = float(
            np.mean([outcome in pred for outcome, pred in zip(outcomes, prediction_sets)])
        )
    return metrics


def monitoring_alerts(metrics: dict[str, float], config: dict | None = None) -> list[str]:
    """Returns human-review alerts; it never changes a decision automatically."""
    settings = (config or {}).get("monitoring", config or {})
    alerts = []
    if metrics.get("psi", 0.0) >= float(settings.get("psi_alert", 0.2)):
        alerts.append("Score-distribution drift exceeds PSI alert threshold.")
    if metrics.get("p99_latency_ms", 0.0) > float(settings.get("p99_latency_alert_ms", 200)):
        alerts.append("p99 scoring latency exceeds the prototype budget.")
    if metrics.get("false_hold_rate", 0.0) > float(settings.get("false_hold_ceiling", 0.01)):
        alerts.append("False-hold rate exceeds the configured review ceiling.")
    return alerts
