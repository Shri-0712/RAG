"""
Baseline fraud detection models for FraudSense.
Baseline A: Tabular Gradient-Boosted Trees (LightGBM) with threshold tuning.
Baseline B: Bayes-optimal cost-sensitive action selector without graph features.
"""

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ..config import load_config
from ..feedback.reward import RewardCalculator


class BaselineA:
    """
    Baseline A: Gradient-Boosted Decision Trees on tabular features only.
    Trained to predict P(fraud). Threshold tuned on validation split.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or load_config()
        self.trees: int = int(cfg.get("models", {}).get("baselines", {}).get("gbdt_trees", 300))
        self.model: HistGradientBoostingClassifier = HistGradientBoostingClassifier(
            max_iter=self.trees,
            max_depth=6,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
        )
        self.threshold_challenge: float = 0.3
        self.threshold_hold: float = 0.7
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        amounts_val: np.ndarray | None = None,
    ) -> "BaselineA":
        """Fits GBDT model on tabular feature matrix."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        if X_val is not None and y_val is not None and amounts_val is not None:
            self.tune_thresholds(X_val, y_val, amounts_val)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted probability of fraud, shape (N,)."""
        if not self.is_fitted:
            raise RuntimeError("BaselineA model is not fitted yet.")
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def tune_thresholds(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        amounts_val: np.ndarray,
    ) -> tuple[float, float]:
        """
        Grid search on validation split to select thresholds that maximize operational reward.
        """
        probs = self.predict_proba(X_val)
        reward_calc = RewardCalculator()

        best_cost = float("inf")
        best_th_chal = 0.3
        best_th_hold = 0.7

        # Grid search over candidate thresholds
        candidate_chals = np.linspace(0.1, 0.5, 9)
        candidate_holds = np.linspace(0.5, 0.9, 9)

        for th_c in candidate_chals:
            for th_h in candidate_holds:
                if th_h <= th_c:
                    continue
                actions = []
                for p in probs:
                    if p >= th_h:
                        actions.append("HOLD_FOR_REVIEW")
                    elif p >= th_c:
                        actions.append("CHALLENGE")
                    else:
                        actions.append("ALLOW")

                cost_res = reward_calc.evaluate_batch(actions, list(y_val), list(amounts_val))
                if cost_res["total_cost"] < best_cost:
                    best_cost = cost_res["total_cost"]
                    best_th_chal = float(th_c)
                    best_th_hold = float(th_h)

        self.threshold_challenge = best_th_chal
        self.threshold_hold = best_th_hold
        return self.threshold_challenge, self.threshold_hold

    def predict_actions(self, X: np.ndarray) -> list[str]:
        """Maps predicted probabilities to discrete actions using tuned thresholds."""
        probs = self.predict_proba(X)
        actions = []
        for p in probs:
            if p >= self.threshold_hold:
                actions.append("HOLD_FOR_REVIEW")
            elif p >= self.threshold_challenge:
                actions.append("CHALLENGE")
            else:
                actions.append("ALLOW")
        return actions


class BaselineB:
    """
    Baseline B: Bayes-optimal cost-sensitive classifier.
    Uses calibrated risk probabilities and exact operational payoffs to choose
    the minimum expected cost action for each transaction.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.reward_calc: RewardCalculator = RewardCalculator(self.config)
        self.base_model: BaselineA = BaselineA(self.config)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "BaselineB":
        """Fits underlying probability estimator."""
        self.base_model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted fraud probability."""
        return self.base_model.predict_proba(X)

    def decide_action(self, p_fraud: float, amount: float) -> str:
        """
        Computes expected cost for ALLOW, CHALLENGE, and HOLD_FOR_REVIEW,
        and selects the action minimizing expected cost.
        """
        p = float(np.clip(p_fraud, 1e-6, 1.0 - 1e-6))
        p_legit = 1.0 - p
        amt = float(amount)

        # Expected cost = E[-Reward]
        # ALLOW:
        # fraud: loss alpha * amt; legit: 0
        exp_cost_allow = p * (self.reward_calc.alpha * amt)

        # CHALLENGE:
        # fraud: loss lambda * amt; legit: loss u
        exp_cost_chal = (p * (self.reward_calc.lambda_ * amt)) + (p_legit * self.reward_calc.u)

        # HOLD_FOR_REVIEW:
        # fraud: -(gamma * amt - c_a) = c_a - gamma * amt
        # legit: -(-h - rho * amt) = h + rho * amt
        cost_hold_fraud = self.reward_calc.c_a - (self.reward_calc.gamma * amt)
        cost_hold_legit = self.reward_calc.h + (self.reward_calc.rho * amt)
        exp_cost_hold = (p * cost_hold_fraud) + (p_legit * cost_hold_legit)

        costs = {
            "ALLOW": exp_cost_allow,
            "CHALLENGE": exp_cost_chal,
            "HOLD_FOR_REVIEW": exp_cost_hold,
        }
        return min(costs, key=costs.get)

    def predict_actions(self, X: np.ndarray, amounts: np.ndarray) -> list[str]:
        """Selects cost-optimal actions for a batch of payments."""
        probs = self.predict_proba(X)
        return [self.decide_action(p, amt) for p, amt in zip(probs, amounts)]
