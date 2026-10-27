"""
Amount override policy rules for FraudSense.
Applies deterministic post-DQN amount constraints for high-value payments.
"""

from typing import Any

from ..config import load_config


class AmountOverride:
    """
    Applies regulatory hard ceilings:
    - Amount >= high_value_threshold forces action to be at least CHALLENGE.
    - If conformal set includes FRAUD, escalates to HOLD_FOR_REVIEW.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.high_val_threshold: float = float(
            self.config.get("decision", {}).get("high_value_threshold", 50000.0)
        )

    def apply(
        self,
        candidate_action: str,
        amount: float,
        conformal_set: set[str] | None = None,
    ) -> tuple[str, bool, str]:
        """
        Evaluates amount override rules.

        Args:
            candidate_action: Proposed action from policy network or router.
            amount: Transaction amount in INR.
            conformal_set: Active conformal prediction set.

        Returns:
            Tuple of (final_action, was_overridden, reason_string).
        """
        conf_set = conformal_set or set()
        act = candidate_action.upper()

        if amount >= self.high_val_threshold:
            if "FRAUD" in conf_set:
                if act != "HOLD_FOR_REVIEW":
                    return (
                        "HOLD_FOR_REVIEW",
                        True,
                        f"Amount ₹{amount:,.2f} >= high_value_threshold (₹{self.high_val_threshold:,.2f}) and conformal set includes FRAUD.",
                    )
            elif act == "ALLOW":
                return (
                    "CHALLENGE",
                    True,
                    f"Amount ₹{amount:,.2f} >= high_value_threshold (₹{self.high_val_threshold:,.2f}) requires at least CHALLENGE.",
                )

        return candidate_action, False, "No override applied."
