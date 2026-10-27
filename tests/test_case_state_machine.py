"""
Unit tests for Case State Machine and statutory lifecycle transitions.
"""

import pytest

from fraudsense.cases.state_machine import CaseState, CaseStateMachine, InvalidStateTransitionError


def test_valid_case_lifecycle():
    """Verify standard legal investigation lifecycle transitions."""
    sm = CaseStateMachine(case_id="CASE_TEST_01", amount=15000.0)
    assert sm.state == CaseState.OPENED

    # OPENED -> ENRICHED
    sm.transition_to(CaseState.ENRICHED, reason="Enriched with Text2SQL and RAG")
    assert sm.state == CaseState.ENRICHED

    # ENRICHED -> ASSIGNED
    sm.transition_to(CaseState.ASSIGNED, reason="Assigned to senior analyst")
    assert sm.state == CaseState.ASSIGNED

    # ASSIGNED -> DECIDED_FRAUD
    sm.transition_to(
        CaseState.DECIDED_FRAUD, reason="Confirmed mule account pattern", clause_id="MD-3.4.2"
    )
    assert sm.state == CaseState.DECIDED_FRAUD

    # DECIDED_FRAUD -> REPORT_DRAFTED
    sm.transition_to(CaseState.REPORT_DRAFTED, reason="CPFIR checklist drafted")
    assert sm.state == CaseState.REPORT_DRAFTED

    # REPORT_DRAFTED -> CLOSED
    sm.transition_to(CaseState.CLOSED, reason="Case closed after compliance review")
    assert sm.state == CaseState.CLOSED


def test_invalid_case_transition_raises_error():
    """Verify illegal transitions raise InvalidStateTransitionError."""
    sm = CaseStateMachine(case_id="CASE_TEST_02", amount=5000.0)

    # Illegal: OPENED cannot jump directly to CLOSED
    with pytest.raises(InvalidStateTransitionError):
        sm.transition_to(CaseState.CLOSED)

    # Illegal: OPENED cannot jump directly to DECIDED_FRAUD
    with pytest.raises(InvalidStateTransitionError):
        sm.transition_to(CaseState.DECIDED_FRAUD)


def test_high_value_supervisor_gate():
    """Verify high-value hold (>= ₹50,000) cannot be closed as LEGITIMATE without supervisor sign-off."""
    sm = CaseStateMachine(case_id="CASE_HIGH_VAL", amount=75000.0, high_value_threshold=50000.0)
    assert sm.requires_supervisor is True

    sm.transition_to(CaseState.ENRICHED)
    sm.transition_to(CaseState.ASSIGNED)

    # Attempt to release high-value hold as legitimate WITHOUT supervisor sign-off -> MUST FAIL
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        sm.transition_to(CaseState.DECIDED_LEGITIMATE, is_supervisor=False)
    assert "requires supervisor approval" in str(exc_info.value)

    # With supervisor approval -> MUST SUCCEED
    sm.transition_to(
        CaseState.DECIDED_LEGITIMATE, is_supervisor=True, reason="Supervisor sign-off granted"
    )
    assert sm.state == CaseState.DECIDED_LEGITIMATE
    assert sm.supervisor_approved is True
