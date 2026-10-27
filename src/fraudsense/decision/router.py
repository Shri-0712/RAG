"""
Decision Router for FraudSense.
Arbitrates between Policy Mask, Conformal Prediction Sets, DQN Policy, Amount Overrides,
and the sub-200ms Fail-Safe Default.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import load_config
from ..ingest.schema import TransactionEvent
from ..models.conformal import SplitConformalCalibrator
from ..models.dqn import DQNActionPolicy
from ..models.risk_head import CalibratedRiskHead
from ..policy.mask import PolicyMask
from ..policy.override import AmountOverride


@dataclass
class DecisionResult:
    """
    Structured outcome of the fast-path decision router.

    Attributes:
        action: Final selected action ('ALLOW', 'CHALLENGE', 'HOLD_FOR_REVIEW').
        p_fraud: Calibrated risk probability P(fraud).
        conformal_set: Active prediction set C(x).
        policy_clauses: RBI clauses invoked in masking or arbitration.
        is_fail_safe: True if decision was produced by timeout/exception fallback.
        was_overridden: True if an amount override altered the proposed action.
        latency_ms: Total fast-path execution time in milliseconds.
        reason: Explanatory justification for audit trail.
    """

    action: str
    p_fraud: float
    conformal_set: set[str] = field(default_factory=set)
    policy_clauses: list[str] = field(default_factory=list)
    is_fail_safe: bool = False
    was_overridden: bool = False
    latency_ms: float = 0.0
    reason: str = ""


class DecisionRouter:
    """
    Fast-path decision engine applying Section 5 rules in strict hierarchical sequence.
    """

    def __init__(
        self,
        risk_head: CalibratedRiskHead | None = None,
        conformal_calibrator: SplitConformalCalibrator | None = None,
        dqn_policy: DQNActionPolicy | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.config = config or load_config()
        self.risk_head = risk_head
        self.conformal = conformal_calibrator
        self.dqn = dqn_policy

        self.policy_mask = PolicyMask(self.config)
        self.amount_override = AmountOverride(self.config)

        dec_cfg = self.config.get("decision", {})
        self.high_val_thresh: float = float(dec_cfg.get("high_value_threshold", 50000.0))
        self.low_val_thresh: float = float(dec_cfg.get("low_value_threshold", 2000.0))
        self.timeout_ms: float = float(dec_cfg.get("fast_path_timeout_ms", 200.0))
        self.wide_set_min_size: int = int(dec_cfg.get("wide_set_min_size", 2))

    def _apply_fail_safe(self, amount: float, reason: str, start_time: float) -> DecisionResult:
        """Applies deterministic fail-safe default when timeout or exception occurs."""
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        if amount >= self.high_val_thresh:
            action = "CHALLENGE"
        elif amount <= self.low_val_thresh:
            action = "ALLOW"
        else:
            action = "ALLOW"

        return DecisionResult(
            action=action,
            p_fraud=0.5,
            conformal_set={"CHALLENGE"} if action == "CHALLENGE" else {"ALLOW"},
            policy_clauses=["MD-3.4.2"],
            is_fail_safe=True,
            was_overridden=False,
            latency_ms=latency_ms,
            reason=f"FAIL-SAFE APPLIED ({reason}): amount ₹{amount:,.2f}",
        )

    def route(
        self,
        event: TransactionEvent,
        tabular_features: np.ndarray,
        gat_embedding: np.ndarray | None = None,
        can_open_case: bool = True,
        is_novel: bool = False,
    ) -> DecisionResult:
        """
        Executes fast-path decision routing for an incoming payment transaction.
        Enforces Section 5 rules in strict priority order.
        """
        start_time = time.perf_counter()
        amt = float(event.amount)

        # 1. Model Availability & Timeout Guard
        if self.risk_head is None or self.conformal is None:
            return self._apply_fail_safe(amt, "Models not initialized", start_time)

        try:
            # Prepare concatenated feature vector: [GAT (64), Tabular (18)]
            if gat_embedding is not None:
                x_full = np.concatenate([gat_embedding, tabular_features]).reshape(1, -1)
            else:
                # Fallback zero GAT embedding if graph node is not yet embedded
                zero_gat = np.zeros(64, dtype=np.float32)
                x_full = np.concatenate([zero_gat, tabular_features]).reshape(1, -1)

            # Predict calibrated multi-class distribution and conformal prediction set
            probs_3 = self.risk_head.predict_multi_proba(x_full)[0]
            p_fraud = float(probs_3[2])
            conf_set = self.conformal.predict_set(probs_3)

            # Check latency budget
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if elapsed_ms > self.timeout_ms:
                return self._apply_fail_safe(
                    amt, f"Timeout {elapsed_ms:.1f}ms > {self.timeout_ms}ms", start_time
                )

            # 2. Evaluate Policy Mask
            allowed_actions, applied_clauses = self.policy_mask.get_allowed_actions(
                amount=amt,
                can_open_case=can_open_case,
            )

            # 3. DQN Proposal (if trained)
            dqn_action = None
            if self.dqn is not None:
                dqn_state = self.dqn.encode_state(
                    conformal_set=conf_set,
                    amount=amt,
                    p_fraud=p_fraud,
                    tabular_features=tabular_features,
                )
                dqn_action = self.dqn.select_action(dqn_state, allowed_actions=allowed_actions)

            # 4. Hierarchical Decision Priority Rules (Section 5)
            # Rule 1: High value threshold override
            if amt >= self.high_val_thresh:
                if "FRAUD" in conf_set or p_fraud >= 0.5:
                    action = "HOLD_FOR_REVIEW"
                    reason = f"High value (₹{amt:,.2f} >= ₹{self.high_val_thresh:,.2f}) with FRAUD in prediction set."
                else:
                    action = "CHALLENGE"
                    reason = f"High value (₹{amt:,.2f} >= ₹{self.high_val_thresh:,.2f}) requires at least CHALLENGE."
            # Rule 2: Ambiguous or Novel set rule
            elif conf_set == {"CHALLENGE"} or ("FRAUD" in conf_set and "ALLOW" in conf_set):
                is_wide = len(conf_set) >= self.wide_set_min_size and "FRAUD" in conf_set
                if is_novel or is_wide:
                    action = "HOLD_FOR_REVIEW"
                    reason = "Uncertain/wide conformal set including FRAUD or novel payment flag."
                else:
                    action = "CHALLENGE"
                    reason = "Conformal set indicates CHALLENGE or mixed FRAUD/ALLOW outcome."
            # Rule 3: Single FRAUD prediction
            elif conf_set == {"FRAUD"}:
                action = "HOLD_FOR_REVIEW"
                reason = "Conformal set isolated to FRAUD."
            # Rule 4: Clear ALLOW prediction
            elif conf_set == {"ALLOW"}:
                action = "ALLOW"
                reason = "Conformal set isolated to ALLOW."
            else:
                # Default to DQN action or CHALLENGE
                action = dqn_action if dqn_action else "CHALLENGE"
                reason = "Routed via policy action network."

            # 5. Amount Override always applies after DQN
            action, was_overridden, override_reason = self.amount_override.apply(
                candidate_action=action,
                amount=amt,
                conformal_set=conf_set,
            )
            if was_overridden:
                reason = f"{reason} | {override_reason}"

            # 6. Final Policy Mask enforcement
            if action not in allowed_actions and action == "HOLD_FOR_REVIEW":
                action = "CHALLENGE"
                reason = f"{reason} | Policy mask downgraded HOLD to CHALLENGE (Case record unavailable, MD-2.1.1)."
            elif action not in allowed_actions:
                action = "CHALLENGE"

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            return DecisionResult(
                action=action,
                p_fraud=p_fraud,
                conformal_set=conf_set,
                policy_clauses=applied_clauses,
                is_fail_safe=False,
                was_overridden=was_overridden,
                latency_ms=latency_ms,
                reason=reason,
            )

        except (RuntimeError, ValueError, KeyError, IndexError) as exc:
            return self._apply_fail_safe(amt, f"Exception: {exc!s}", start_time)
