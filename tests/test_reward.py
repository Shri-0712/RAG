"""
Unit tests for operational reward and cost calculation matrix.
"""

import pytest

from fraudsense.feedback.reward import RewardCalculator


def test_reward_calculator_fraud_payoffs():
    """Verify reward calculation for fraudulent transactions (label = 1)."""
    calc = RewardCalculator()
    amt = 10000.0

    # ALLOW, fraud: -alpha * amount (-1.0 * 10000 = -10000)
    rew_allow = calc.compute_reward("ALLOW", label=1, amount=amt)
    assert rew_allow == -10000.0

    # CHALLENGE, fraud: -lambda * amount (-0.2 * 10000 = -2000)
    rew_chal = calc.compute_reward("CHALLENGE", label=1, amount=amt)
    assert rew_chal == -2000.0

    # HOLD, fraud: +gamma * amount - c_a (+0.9 * 10000 - 150 = 8850)
    rew_hold = calc.compute_reward("HOLD", label=1, amount=amt)
    assert rew_hold == 8850.0

    # Cost is negative reward
    assert calc.compute_cost("ALLOW", label=1, amount=amt) == 10000.0
    assert calc.compute_cost("HOLD", label=1, amount=amt) == -8850.0


def test_reward_calculator_legitimate_payoffs():
    """Verify reward calculation for legitimate transactions (label = 0)."""
    calc = RewardCalculator()
    amt = 10000.0

    # ALLOW, legit: 0
    rew_allow = calc.compute_reward("ALLOW", label=0, amount=amt)
    assert rew_allow == 0.0

    # CHALLENGE, legit: -u (-5)
    rew_chal = calc.compute_reward("CHALLENGE", label=0, amount=amt)
    assert rew_chal == -5.0

    # HOLD, legit: -h - rho * amount (-300 - 0.01 * 10000 = -400)
    rew_hold = calc.compute_reward("HOLD", label=0, amount=amt)
    assert rew_hold == -400.0


def test_reward_calculator_batch_metrics():
    """Verify batch cost per 10k normalization."""
    calc = RewardCalculator()
    actions = ["ALLOW", "CHALLENGE", "HOLD"]
    labels = [0, 0, 1]
    amounts = [1000.0, 500.0, 10000.0]

    # Rewards:
    # ALLOW legit: 0.0
    # CHALLENGE legit: -5.0
    # HOLD fraud: 0.9 * 10000 - 150 = 8850.0
    # Total reward = 8845.0
    # Total cost = -8845.0
    # Cost per 10k = (-8845 / 3) * 10000
    res = calc.evaluate_batch(actions, labels, amounts)
    assert res["total_reward"] == 8845.0
    assert res["total_cost"] == -8845.0
    assert pytest.approx(res["cost_per_10k"], 0.1) == (-8845.0 / 3.0) * 10000.0
