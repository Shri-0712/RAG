"""
Simulated Challenge Handler for FraudSense.
Simulates SMS/In-App customer verification prompts with probabilistic customer behavior
and strict 5-minute timeout escalation to HOLD_FOR_REVIEW.
"""

import random
from enum import Enum
from typing import Any

from ..config import load_config


class ChallengeStatus(str, Enum):
    SENT = "SENT"
    CONFIRMED = "CONFIRMED"
    DENIED = "DENIED"
    TIMED_OUT = "TIMED_OUT"


class ChallengeHandler:
    """
    Simulates interactive customer verification challenge workflows.
    All external interactions are explicitly stamped SIMULATED.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or load_config()
        sim_cfg = cfg.get("simulation", {})
        self.p_confirm_legit: float = float(sim_cfg.get("p_confirm_legit", 0.90))
        self.p_deny_fraud: float = float(sim_cfg.get("p_deny_fraud", 0.70))

        timers_cfg = cfg.get("timers", {})
        self.timeout_min: float = float(timers_cfg.get("challenge_timeout_min", 5.0))

    def create_challenge_prompt(
        self,
        event_id: str,
        amount: float,
        channel: str,
        template_id: str = "TMPL_CHALLENGE_UPI_01",
    ) -> dict[str, Any]:
        """Creates a synthetic customer verification prompt with simulated disclaimer."""
        return {
            "status": ChallengeStatus.SENT.value,
            "event_id": event_id,
            "template_id": template_id,
            "clause_citation": "MD-3.4.2",
            "prompt_text": f"[SIMULATED BANK ALERT] Did you initiate a {channel} payment of INR {amount:,.2f}? Reply 1 to Confirm, 2 to Report Fraud.",
            "simulated_flag": "SIMULATED",
            "timeout_minutes": self.timeout_min,
        }

    def simulate_response(
        self,
        true_label: int,
        elapsed_minutes: float = 0.0,
    ) -> tuple[ChallengeStatus, str, str]:
        """
        Simulates customer response given ground truth label and elapsed simulated time.

        Guarantees:
        - A timeout (elapsed >= timeout_min) MUST NEVER release funds.
          It strictly resolves to HOLD_FOR_REVIEW (Clause MD-3.4.2 fail-safe).

        Returns:
            Tuple of (ChallengeStatus, resolved_action, audit_reason).
        """
        # Strict timeout guard
        if elapsed_minutes >= self.timeout_min:
            return (
                ChallengeStatus.TIMED_OUT,
                "HOLD_FOR_REVIEW",
                f"SIMULATED: Challenge timed out after {elapsed_minutes:.1f}m (limit {self.timeout_min}m). Funds held under fail-safe MD-3.4.2.",
            )

        if true_label == 0:  # Legitimate payment
            # Customer confirms legitimate payment with probability p_confirm_legit
            if random.random() < self.p_confirm_legit:
                return (
                    ChallengeStatus.CONFIRMED,
                    "ALLOW",
                    "SIMULATED: Customer confirmed legitimate transaction. Payment released.",
                )
            else:
                return (
                    ChallengeStatus.DENIED,
                    "HOLD_FOR_REVIEW",
                    "SIMULATED: Customer denied transaction (false positive alarm). Escalated to HOLD.",
                )
        else:  # Fraudulent payment
            # Customer denies fraud payment with probability p_deny_fraud
            if random.random() < self.p_deny_fraud:
                return (
                    ChallengeStatus.DENIED,
                    "HOLD_FOR_REVIEW",
                    "SIMULATED: Victim denied fraudulent transaction. Escalated to HOLD.",
                )
            else:
                # Attacker/compromised credential confirmation
                return (
                    ChallengeStatus.CONFIRMED,
                    "ALLOW",
                    "SIMULATED: Fraud challenge confirmed via compromised channel (missed fraud).",
                )
