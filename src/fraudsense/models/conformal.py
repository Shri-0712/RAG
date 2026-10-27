"""
Transparent Split Conformal Predictor for FraudSense.
Implements finite-sample valid prediction sets over {ALLOW, CHALLENGE, FRAUD}
using the canonical nonconformity score s(x, y) = 1 - P(Y = y | x).
"""

from typing import Any

import numpy as np

from ..config import load_config

CLASSES = ["ALLOW", "CHALLENGE", "FRAUD"]


class SplitConformalCalibrator:
    """
    Split conformal calibrator providing distribution-free finite-sample coverage guarantees.
    Target coverage: 1 - alpha (default 0.90).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or load_config()
        self.target_coverage: float = float(cfg.get("decision", {}).get("conformal_coverage", 0.90))
        self.alpha: float = 1.0 - self.target_coverage
        self.quantile_threshold: float = 1.0
        self.is_calibrated: bool = False

    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """
        Calibrates the nonconformity quantile threshold on held-out calibration split.

        Args:
            probs_cal: Array of shape (n, 3) containing predicted class probabilities.
            y_cal: Array of shape (n,) containing true class indices (0, 1, or 2).
        """
        n = len(y_cal)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Compute nonconformity score for true class: s_i = 1 - P(y_i)
        true_class_probs = probs_cal[np.arange(n), y_cal]
        scores = 1.0 - true_class_probs

        # Finite-sample adjusted quantile index: ceil((n + 1) * (1 - alpha)) / n
        q_level = float(np.clip(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0, 1.0))
        self.quantile_threshold = float(np.quantile(scores, q_level, method="higher"))
        self.is_calibrated = True
        return self.quantile_threshold

    def predict_set(self, prob_vector: np.ndarray) -> set[str]:
        """
        Produces prediction set C(x) subset of {ALLOW, CHALLENGE, FRAUD}.
        Includes class k if 1 - p_k <= q (i.e. p_k >= 1 - q).
        """
        if not self.is_calibrated:
            raise RuntimeError("SplitConformalCalibrator must be calibrated before prediction.")

        thresh = 1.0 - self.quantile_threshold
        pred_set = {CLASSES[k] for k in range(len(CLASSES)) if prob_vector[k] >= thresh}
        # Guarantee non-empty prediction set by adding argmax if empty
        if not pred_set:
            pred_set.add(CLASSES[int(np.argmax(prob_vector))])
        return pred_set

    def predict_batch(self, probs: np.ndarray) -> list[set[str]]:
        """Produces prediction sets for a batch of probability vectors."""
        return [self.predict_set(p) for p in probs]

    def evaluate_coverage(self, probs_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Evaluates empirical coverage and mean set size on test split."""
        pred_sets = self.predict_batch(probs_test)
        covered = sum(CLASSES[int(y)] in s for y, s in zip(y_test, pred_sets))
        n = len(y_test)
        return {
            "empirical_coverage": float(covered / max(1, n)),
            "target_coverage": self.target_coverage,
            "mean_set_size": float(np.mean([len(s) for s in pred_sets])),
            "singleton_rate": float(np.mean([len(s) == 1 for s in pred_sets])),
            "wide_set_rate": float(np.mean([len(s) >= 2 for s in pred_sets])),
        }
