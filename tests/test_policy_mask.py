"""
Unit tests for regulatory policy mask rules and clause constraints.
"""

from fraudsense.policy.mask import PolicyMask


def test_policy_mask_natural_justice_case_open():
    """Verify HOLD_FOR_REVIEW is disallowed if a case record cannot be opened (MD-2.1.1)."""
    mask = PolicyMask()

    # When can_open_case is False, HOLD_FOR_REVIEW must be removed
    allowed, clauses = mask.get_allowed_actions(amount=1000.0, can_open_case=False)
    assert "HOLD_FOR_REVIEW" not in allowed
    assert "ALLOW" in allowed
    assert "CHALLENGE" in allowed
    assert "MD-2.1.1" in clauses


def test_policy_mask_high_value_containment():
    """Verify ALLOW is masked for transactions >= high_value_threshold (MD-3.4.2)."""
    mask = PolicyMask()

    # Amount >= ₹50,000 cannot be silently allowed
    allowed, clauses = mask.get_allowed_actions(amount=75000.0, can_open_case=True)
    assert "ALLOW" not in allowed
    assert "CHALLENGE" in allowed
    assert "HOLD_FOR_REVIEW" in allowed
    assert "MD-3.4.2" in clauses


def test_policy_mask_combined_restrictions():
    """Verify only CHALLENGE remains if both case record is unavailable and amount >= ₹50,000."""
    mask = PolicyMask()
    allowed, clauses = mask.get_allowed_actions(amount=100000.0, can_open_case=False)

    assert allowed == {"CHALLENGE"}
    assert "MD-2.1.1" in clauses
    assert "MD-3.4.2" in clauses
