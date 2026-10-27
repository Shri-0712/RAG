"""
Unit tests for Baseline A (GBDT) and Baseline B (Cost-Sensitive).
"""

import numpy as np

from fraudsense.models.baselines import BaselineA, BaselineB


def test_baseline_a_fit_predict():
    """Verify Baseline A fits GBDT, outputs valid probabilities and discrete actions."""
    np.random.seed(42)
    N = 200
    D = 10
    X = np.random.randn(N, D).astype(np.float32)
    # Binary labels with 10% fraud
    y = (np.random.rand(N) < 0.1).astype(int)
    amounts = np.random.uniform(100.0, 50000.0, size=N)

    base_a = BaselineA()
    base_a.fit(X[:150], y[:150], X[150:], y[150:], amounts[150:])

    probs = base_a.predict_proba(X[150:])
    assert len(probs) == 50
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    actions = base_a.predict_actions(X[150:])
    assert len(actions) == 50
    assert set(actions).issubset({"ALLOW", "CHALLENGE", "HOLD_FOR_REVIEW"})


def test_baseline_b_cost_sensitive_decisions():
    """Verify Baseline B selects cost-optimal actions based on Bayes risk."""
    base_b = BaselineB()

    # Case 1: Very low fraud risk (p = 0.0001), low amount (₹100) -> Should ALLOW (cost 0)
    act_low = base_b.decide_action(p_fraud=0.0001, amount=100.0)
    assert act_low == "ALLOW"

    # Case 2: Certain high fraud risk (p = 0.999), high amount (₹100,000)
    # HOLD fraud reward = +0.9 * 100,000 - 150 = +89,850 (cost -89,850)
    # -> Should HOLD_FOR_REVIEW
    act_high = base_b.decide_action(p_fraud=0.999, amount=100000.0)
    assert act_high == "HOLD_FOR_REVIEW"

    # Case 3: Moderate fraud risk (p = 0.4), small amount (₹500)
    # ALLOW cost: 0.4 * 500 = 200
    # CHALLENGE cost: 0.4 * 0.2 * 500 + 0.6 * 5 = 40 + 3 = 43
    # HOLD cost: 0.4 * (150 - 450) + 0.6 * (300 + 5) = -120 + 183 = 63
    # CHALLENGE has lowest expected cost (43 < 63 < 200) -> Should CHALLENGE
    act_mid = base_b.decide_action(p_fraud=0.4, amount=500.0)
    assert act_mid == "CHALLENGE"
