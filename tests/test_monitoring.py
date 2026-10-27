"""Tests for transparent Phase 9 monitoring metrics."""

from fraudsense.monitoring.metrics import (
    monitoring_alerts,
    operational_metrics,
    population_stability_index,
)


def test_psi_detects_distribution_shift():
    assert population_stability_index([0.1] * 100 + [0.2] * 100, [0.9] * 200) > 0.2


def test_operational_metrics_and_alerts():
    metrics = operational_metrics(
        ["HOLD_FOR_REVIEW", "CHALLENGE", "ALLOW"],
        [0, 1, 0],
        [10, 20, 30],
        [{"ALLOW"}, {"FRAUD"}, {"ALLOW"}],
    )
    assert metrics["false_hold_rate"] == 1 / 3
    assert metrics["conformal_coverage"] == 1.0
    assert monitoring_alerts(
        {**metrics, "psi": 0.3}, {"psi_alert": 0.2, "false_hold_ceiling": 0.01}
    )
