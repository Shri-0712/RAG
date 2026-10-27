"""
Unit tests for post-DQN amount overrides.
"""

from fraudsense.policy.override import AmountOverride


def test_amount_override_high_value_allow_to_challenge():
    """Verify ALLOW on >= ₹50,000 is upgraded to CHALLENGE."""
    override = AmountOverride()
    action, was_overridden, reason = override.apply(
        candidate_action="ALLOW",
        amount=60000.0,
        conformal_set={"ALLOW"},
    )
    assert action == "CHALLENGE"
    assert was_overridden is True
    assert "requires at least CHALLENGE" in reason


def test_amount_override_high_value_with_fraud_to_hold():
    """Verify high value with FRAUD in prediction set escalates to HOLD_FOR_REVIEW."""
    override = AmountOverride()
    action, was_overridden, reason = override.apply(
        candidate_action="CHALLENGE",
        amount=80000.0,
        conformal_set={"CHALLENGE", "FRAUD"},
    )
    assert action == "HOLD_FOR_REVIEW"
    assert was_overridden is True
    assert "includes FRAUD" in reason


def test_amount_override_normal_pass_through():
    """Verify small amounts pass through without override."""
    override = AmountOverride()
    action, was_overridden, _ = override.apply(
        candidate_action="ALLOW",
        amount=1500.0,
        conformal_set={"ALLOW"},
    )
    assert action == "ALLOW"
    assert was_overridden is False
