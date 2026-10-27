"""
Unit tests for customer verification challenge handling and timeout escalation.
"""

from fraudsense.challenge.handler import ChallengeHandler, ChallengeStatus


def test_challenge_prompt_creation():
    """Verify challenge prompt structure, clause citation, and SIMULATED disclaimer."""
    handler = ChallengeHandler()
    prompt = handler.create_challenge_prompt(
        event_id="EVT_CHAL_01",
        amount=15000.0,
        channel="UPI",
    )
    assert prompt["status"] == ChallengeStatus.SENT.value
    assert prompt["simulated_flag"] == "SIMULATED"
    assert prompt["clause_citation"] == "MD-3.4.2"
    assert "SIMULATED BANK ALERT" in prompt["prompt_text"]
    assert prompt["timeout_minutes"] == 5.0


def test_challenge_timeout_never_releases_funds():
    """Non-negotiable rule: Challenge timeout must escalate to HOLD_FOR_REVIEW, never release."""
    handler = ChallengeHandler()

    # Elapsed 6 minutes (> 5 minutes timeout limit)
    status, action, reason = handler.simulate_response(true_label=0, elapsed_minutes=6.0)

    assert status == ChallengeStatus.TIMED_OUT
    assert action == "HOLD_FOR_REVIEW"
    assert "timed out" in reason
    assert "MD-3.4.2" in reason
