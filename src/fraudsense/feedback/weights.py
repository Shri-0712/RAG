"""
Inverse Propensity Scoring and Importance Weighting for FraudSense.
Corrects selection and verification bias arising from partial analyst review of payments.
"""

from typing import Any

import numpy as np

from ..config import load_config


class PropensityWeightCalculator:
    """
    Computes inverse propensity weights (IPW) to rectify selection bias in fraud training data.

    Mathematical Formulation:
    - Reviewed payments (HOLD or CHALLENGE) are examined by analysts (P(Audit|HOLD) = 1.0).
    - Silently released payments (ALLOW) are only audited if randomly sampled (P(Audit|ALLOW) = random_audit_rate).
    - Unbiased sample weight: w_i = 1 / P(Audit | A_i).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or load_config()
        self.random_audit_rate: float = float(
            cfg.get("feedback", {}).get("random_audit_rate", 0.01)
        )
        # Weight clipping to prevent gradient instability
        self.max_weight: float = 1.0 / max(1e-4, self.random_audit_rate)

    def compute_weight(self, action: str, is_random_audit: bool = False) -> float:
        """
        Computes the importance weight for an audited transaction.

        Args:
            action: Selected action ('ALLOW', 'CHALLENGE', 'HOLD_FOR_REVIEW').
            is_random_audit: Whether this transaction was selected via random audit.

        Returns:
            Positive float sample weight w_i.
        """
        act = action.upper()
        if act == "ALLOW":
            if is_random_audit:
                # 1 / p_audit = 1 / 0.01 = 100.0
                return float(np.clip(1.0 / self.random_audit_rate, 1.0, self.max_weight))
            # Normal dispute/chargeback on ALLOW has unknown reporting delay; baseline weight 1.0
            return 1.0
        elif act in ("HOLD", "HOLD_FOR_REVIEW"):
            # P(Audit | HOLD) = 1.0 => w = 1.0
            return 1.0
        elif act == "CHALLENGE":
            # P(Audit/Response | CHALLENGE) = 1.0 => w = 1.0
            return 1.0
        return 1.0

    def compute_batch_weights(
        self,
        actions: list[str],
        random_audit_flags: list[bool],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Computes stabilized importance weights for a batch of matured training samples.
        If normalize=True, scales weights so mean(weights) = 1.0.
        """
        weights = np.array(
            [self.compute_weight(a, flag) for a, flag in zip(actions, random_audit_flags)],
            dtype=np.float32,
        )
        if normalize and len(weights) > 0 and weights.sum() > 0:
            weights = (weights / weights.mean()).astype(np.float32)
        return weights
