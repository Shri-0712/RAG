"""
Operational reward and cost calculator for FraudSense.
Implements the exact payoff matrix defined in Section 6 of the specification.
"""

from typing import Any

from ..config import load_config


class RewardCalculator:
    """
    Computes financial and friction payoffs based on transaction action and true label.

    Payoff parameters (loaded from config/params.yaml):
    - alpha: Loss multiplier for allowed fraud (-alpha * amount)
    - lambda: Friction/loss multiplier for challenged fraud (-lambda * amount)
    - u: Customer friction cost for legitimate challenge (-u)
    - gamma: Prevented loss fraction for held fraud (+gamma * amount - c_a)
    - c_a: Investigation cost per held case (-c_a)
    - h: Fixed operational friction cost for false hold (-h)
    - rho: Amount-proportional customer churn penalty for false hold (-rho * amount)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or load_config()
        reward_cfg = cfg.get("reward", {})
        self.alpha: float = float(reward_cfg.get("alpha", 1.0))
        self.lambda_: float = float(reward_cfg.get("lambda", 0.2))
        self.u: float = float(reward_cfg.get("u", 5.0))
        self.gamma: float = float(reward_cfg.get("gamma", 0.9))
        self.c_a: float = float(reward_cfg.get("c_a", 150.0))
        self.h: float = float(reward_cfg.get("h", 300.0))
        self.rho: float = float(reward_cfg.get("rho", 0.01))

    def compute_reward(self, action: str, label: int, amount: float) -> float:
        """
        Computes the operational payoff for an individual decision.

        Args:
            action: One of 'ALLOW', 'CHALLENGE', 'HOLD_FOR_REVIEW'.
            label: Ground truth label (0 = legitimate, 1 = fraudulent).
            amount: Transaction amount in INR.

        Returns:
            Operational payoff (higher is better).
        """
        act = action.upper()
        if label == 1:  # Fraudulent transaction
            if act == "ALLOW":
                return -self.alpha * amount
            elif act == "CHALLENGE":
                return -self.lambda_ * amount
            elif act in ("HOLD", "HOLD_FOR_REVIEW"):
                return (self.gamma * amount) - self.c_a
            else:
                raise ValueError(f"Unknown action: {action}")
        else:  # Legitimate transaction
            if act == "ALLOW":
                return 0.0
            elif act == "CHALLENGE":
                return -self.u
            elif act in ("HOLD", "HOLD_FOR_REVIEW"):
                return -self.h - (self.rho * amount)
            else:
                raise ValueError(f"Unknown action: {action}")

    def compute_cost(self, action: str, label: int, amount: float) -> float:
        """Cost is the negative of reward (lower is better)."""
        return -self.compute_reward(action, label, amount)

    def evaluate_batch(
        self,
        actions: list[str],
        labels: list[int],
        amounts: list[float],
    ) -> dict[str, float]:
        """
        Computes aggregate reward and cost metrics over a batch of transactions.

        Returns:
            Dict containing total_reward, total_cost, cost_per_10k, mean_cost.
        """
        n = len(actions)
        if n == 0:
            return {
                "total_reward": 0.0,
                "total_cost": 0.0,
                "cost_per_10k": 0.0,
                "mean_cost": 0.0,
            }

        rewards = [
            self.compute_reward(act, int(lbl), float(amt))
            for act, lbl, amt in zip(actions, labels, amounts)
        ]
        total_reward = float(sum(rewards))
        total_cost = -total_reward
        cost_per_10k = (total_cost / n) * 10000.0

        return {
            "total_reward": total_reward,
            "total_cost": total_cost,
            "cost_per_10k": cost_per_10k,
            "mean_cost": total_cost / n,
        }
