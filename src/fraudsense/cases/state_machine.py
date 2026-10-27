"""
Case lifecycle state machine for FraudSense.
Enforces strict statutory and operational transition rules across case states.
"""

from enum import Enum


class CaseState(str, Enum):
    OPENED = "OPENED"
    ENRICHED = "ENRICHED"
    ASSIGNED = "ASSIGNED"
    PENDING_INFO = "PENDING_INFO"
    PENDING_SUPERVISOR = "PENDING_SUPERVISOR"
    DECIDED_FRAUD = "DECIDED_FRAUD"
    DECIDED_LEGITIMATE = "DECIDED_LEGITIMATE"
    REPORT_DRAFTED = "REPORT_DRAFTED"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal transition is attempted on a case record."""


# Strict allowed state transitions
ALLOWED_TRANSITIONS: dict[CaseState, set[CaseState]] = {
    CaseState.OPENED: {CaseState.ENRICHED, CaseState.EXPIRED},
    CaseState.ENRICHED: {CaseState.ASSIGNED, CaseState.EXPIRED},
    CaseState.ASSIGNED: {
        CaseState.PENDING_INFO,
        CaseState.PENDING_SUPERVISOR,
        CaseState.DECIDED_FRAUD,
        CaseState.DECIDED_LEGITIMATE,
        CaseState.EXPIRED,
    },
    CaseState.PENDING_INFO: {CaseState.ASSIGNED, CaseState.EXPIRED},
    CaseState.PENDING_SUPERVISOR: {
        CaseState.DECIDED_FRAUD,
        CaseState.DECIDED_LEGITIMATE,
        CaseState.ASSIGNED,
        CaseState.EXPIRED,
    },
    CaseState.DECIDED_FRAUD: {CaseState.REPORT_DRAFTED, CaseState.CLOSED},
    CaseState.DECIDED_LEGITIMATE: {CaseState.CLOSED},
    CaseState.REPORT_DRAFTED: {CaseState.CLOSED},
    CaseState.EXPIRED: {CaseState.CLOSED},
    CaseState.CLOSED: set(),  # Terminal state
}


class CaseStateMachine:
    """
    Validates and executes lifecycle transitions for fraud investigation cases.
    """

    def __init__(
        self,
        case_id: str,
        initial_state: CaseState = CaseState.OPENED,
        amount: float = 0.0,
        high_value_threshold: float = 50000.0,
    ):
        self.case_id = case_id
        self.state = initial_state
        self.amount = amount
        self.high_value_threshold = high_value_threshold
        self.requires_supervisor = amount >= high_value_threshold
        self.supervisor_approved = False
        self.history: list[dict[str, str]] = [
            {"from": "INIT", "to": self.state.value, "reason": "Case initialized"}
        ]

    def transition_to(
        self,
        new_state: CaseState,
        reason: str = "",
        clause_id: str | None = None,
        is_supervisor: bool = False,
    ) -> CaseState:
        """
        Executes transition to new_state.
        Raises InvalidStateTransitionError if transition is illegal or supervisor gate is violated.
        """
        allowed_targets = ALLOWED_TRANSITIONS.get(self.state, set())
        if new_state not in allowed_targets:
            raise InvalidStateTransitionError(
                f"Invalid transition for case '{self.case_id}': cannot move from '{self.state.value}' to '{new_state.value}'. "
                f"Allowed transitions: {[s.value for s in allowed_targets]}."
            )

        # High-value supervisor gate:
        # If case amount >= ₹50,000, closing as LEGITIMATE requires supervisor sign-off (MD-3.4.3)
        if (
            self.requires_supervisor
            and new_state == CaseState.DECIDED_LEGITIMATE
            and not self.supervisor_approved
            and not is_supervisor
        ):
            raise InvalidStateTransitionError(
                f"Case '{self.case_id}' amount (₹{self.amount:,.2f}) >= ₹{self.high_value_threshold:,.2f} "
                f"requires supervisor approval before release to DECIDED_LEGITIMATE (MD-3.4.3)."
            )

        if is_supervisor:
            self.supervisor_approved = True

        old_state = self.state
        self.state = new_state
        self.history.append(
            {
                "from": old_state.value,
                "to": new_state.value,
                "reason": reason,
                "clause_id": clause_id or "MD-3.4.2",
            }
        )
        return self.state
