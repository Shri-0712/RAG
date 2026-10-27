"""
Unit tests for fast-path Decision Router, priority rules, and fail-safe defaults.
"""

from datetime import UTC, datetime

import numpy as np

from fraudsense.decision.router import DecisionResult, DecisionRouter
from fraudsense.ingest.schema import TransactionEvent
from fraudsense.models.conformal import SplitConformalCalibrator
from fraudsense.models.risk_head import CalibratedRiskHead


def _make_dummy_event(amount: float) -> TransactionEvent:
    return TransactionEvent(
        event_id="EVT_ROUTER_TEST",
        ts=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        amount=amount,
        payer_id="ACC_SRC",
        payee_id="ACC_DST",
        channel="TRANSFER",
        device_id="DEV_TEST",
        label=0,
    )


def test_decision_router_fail_safe_on_missing_models():
    """Verify router immediately triggers fail-safe if models are not initialized."""
    router = DecisionRouter(risk_head=None, conformal_calibrator=None)

    # Low value payment <= ₹2000 -> ALLOW with fail_safe flag
    res_low = router.route(_make_dummy_event(1500.0), tabular_features=np.zeros(18))
    assert isinstance(res_low, DecisionResult)
    assert res_low.is_fail_safe is True
    assert res_low.action == "ALLOW"

    # High value payment >= ₹50000 -> CHALLENGE with fail_safe flag
    res_high = router.route(_make_dummy_event(60000.0), tabular_features=np.zeros(18))
    assert res_high.is_fail_safe is True
    assert res_high.action == "CHALLENGE"


def test_decision_router_section_5_priority_rules():
    """Verify Section 5 hierarchical rules: high-value, wide set, clear allow."""
    # Setup trained risk head and conformal predictor
    np.random.seed(42)
    risk_head = CalibratedRiskHead(in_features=82)
    X = np.random.randn(200, 82).astype(np.float32)
    y = np.random.choice([0, 1], size=200, p=[0.8, 0.2])
    amt = np.random.uniform(500.0, 60000.0, size=200)
    risk_head.fit(X, y, X, y, amt, amt, epochs=2)

    conformal = SplitConformalCalibrator()
    probs = risk_head.predict_multi_proba(X)
    y_multi = risk_head._prepare_labels(y, amt).numpy()
    conformal.calibrate(probs, y_multi)

    router = DecisionRouter(risk_head=risk_head, conformal_calibrator=conformal)

    # 1. High value >= ₹50,000 must result in at least CHALLENGE or HOLD
    high_event = _make_dummy_event(amount=75000.0)
    res1 = router.route(high_event, tabular_features=np.zeros(18))
    assert res1.action in {"CHALLENGE", "HOLD_FOR_REVIEW"}
    assert res1.is_fail_safe is False

    # 2. Case where can_open_case is False -> HOLD_FOR_REVIEW must be downgraded
    res2 = router.route(high_event, tabular_features=np.zeros(18), can_open_case=False)
    assert res2.action != "HOLD_FOR_REVIEW"
    assert res2.action == "CHALLENGE"

    # 3. Execution latency must be well below 200ms
    assert res1.latency_ms < 200.0
