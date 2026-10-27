"""
Regulatory policy mask engine for FraudSense.
Filters available actions based on statutory RBI clauses and operational case constraints.
"""

from typing import Any

from ..config import load_config

ALL_ACTIONS = {"ALLOW", "CHALLENGE", "HOLD_FOR_REVIEW"}


class PolicyMask:
    """
    Evaluates regulatory and systemic rules to determine allowable actions.
    Keyed by RBI Master Direction clause IDs.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.high_val_threshold: float = float(
            self.config.get("decision", {}).get("high_value_threshold", 50000.0)
        )

    def get_allowed_actions(
        self,
        amount: float,
        can_open_case: bool = True,
        has_supervisor_approval: bool = False,
        is_high_value_hold_release: bool = False,
    ) -> tuple[set[str], list[str]]:
        """
        Determines the set of legally/operationally allowable actions.

        Args:
            amount: Transaction amount in INR.
            can_open_case: Whether a case record can be created in the case management store.
            has_supervisor_approval: Whether supervisor has signed off on release.
            is_high_value_hold_release: Whether this action is attempting to release a high-value hold.

        Returns:
            Tuple of (allowed_actions_set, applied_clause_ids).
        """
        allowed = set(ALL_ACTIONS)
        applied_clauses = []

        # Clause MD-2.1.1 & MD-4.1.4: Natural Justice / Case Governance
        # HOLD_FOR_REVIEW is restricted to cases with an opened case record.
        if not can_open_case:
            allowed.discard("HOLD_FOR_REVIEW")
            applied_clauses.append("MD-2.1.1")

        # Clause MD-3.4.2: Real-time mule & fraud containment
        # High value transactions (>= ₹50,000) cannot be silently released without at least a challenge
        if amount >= self.high_val_threshold:
            allowed.discard("ALLOW")
            applied_clauses.append("MD-3.4.2")

        # Supervisor approval gate for high-value holds
        if (
            is_high_value_hold_release
            and amount >= self.high_val_threshold
            and not has_supervisor_approval
        ):
            allowed.discard("ALLOW")
            applied_clauses.append("MD-3.4.3")

        # Fail-safe guarantee: allowed must never be empty
        if not allowed:
            allowed = {"CHALLENGE"}

        return allowed, applied_clauses
