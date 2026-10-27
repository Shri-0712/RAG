"""
Deep Q-Network (DQN) action policy for FraudSense.
Lightweight PyTorch implementation (< 300 lines) of a single-step contextual bandit (gamma=0),
optimizing operational fraud containment payoffs filtered by regulatory policy masks.
"""

import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config import load_config
from ..feedback.reward import RewardCalculator

ACTION_LIST = ["ALLOW", "CHALLENGE", "HOLD_FOR_REVIEW"]
ACTION_TO_IDX = {act: i for i, act in enumerate(ACTION_LIST)}
IDX_TO_ACTION = {i: act for i, act in enumerate(ACTION_LIST)}


class QNetwork(nn.Module):
    """Deep Q-Network mapping state vector to action values Q(s, a)."""

    def __init__(self, state_dim: int = 7, hidden_dim: int = 128, action_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(state))
        h = F.relu(self.fc2(h))
        return self.out(h)


class ReplayBuffer:
    """Experience replay buffer for storing state-action transitions."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNActionPolicy:
    """
    DQN Action Policy selecting optimal operational decisions:
    {ALLOW, CHALLENGE, HOLD_FOR_REVIEW} subject to policy masks.
    """

    def __init__(self, state_dim: int = 7, config: dict[str, Any] | None = None):
        cfg = config or load_config()
        dqn_cfg = cfg.get("models", {}).get("dqn", {})
        self.hidden_dim: int = int(dqn_cfg.get("hidden_units", 128))
        self.lr: float = float(dqn_cfg.get("learning_rate", 0.001))
        self.gamma: float = float(dqn_cfg.get("gamma_discount", 0.0))  # 0.0 for 1-step decisions
        self.buffer_size: int = int(dqn_cfg.get("replay_buffer_size", 50000))

        self.q_net = QNetwork(state_dim=state_dim, hidden_dim=self.hidden_dim, action_dim=3)
        self.target_net = QNetwork(state_dim=state_dim, hidden_dim=self.hidden_dim, action_dim=3)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(capacity=self.buffer_size)
        self.reward_calc = RewardCalculator(cfg)
        self.epsilon: float = 0.2

    @staticmethod
    def encode_state(
        conformal_set: set[str],
        amount: float,
        p_fraud: float,
        tabular_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encodes context into 7-dimensional continuous state vector."""
        # 1-3: Conformal set binary encoding
        has_allow = 1.0 if "ALLOW" in conformal_set else 0.0
        has_chal = 1.0 if "CHALLENGE" in conformal_set else 0.0
        has_fraud = 1.0 if "FRAUD" in conformal_set else 0.0

        # 4: Log amount bucket
        norm_amt = float(np.log1p(max(0.0, amount)) / 12.0)

        # 5: Calibrated fraud risk score
        p = float(np.clip(p_fraud, 0.0, 1.0))

        # 6-7: Account velocity / ratios from tabular features if provided
        ratio = 1.0
        time_last = 0.5
        if tabular_features is not None and len(tabular_features) >= 9:
            ratio = float(np.clip(tabular_features[7] / 5.0, 0.0, 2.0))
            time_last = float(np.clip(tabular_features[8] / 86400.0, 0.0, 1.0))

        return np.array(
            [has_allow, has_chal, has_fraud, norm_amt, p, ratio, time_last],
            dtype=np.float32,
        )

    def select_action(
        self,
        state: np.ndarray,
        allowed_actions: set[str],
        explore: bool = False,
    ) -> str:
        """
        Selects an action filtered by the regulatory policy mask.
        Invalid actions are masked with -infinity Q-values.
        """
        valid_indices = [ACTION_TO_IDX[a] for a in allowed_actions if a in ACTION_TO_IDX]
        if not valid_indices:
            valid_indices = [ACTION_TO_IDX["CHALLENGE"]]

        if explore and random.random() < self.epsilon:
            chosen_idx = random.choice(valid_indices)
            return IDX_TO_ACTION[chosen_idx]

        self.q_net.eval()
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_vals = self.q_net(s_t).squeeze(0).numpy()

            # Mask unallowed actions with large negative value
            masked_q = np.full_like(q_vals, -1e9)
            for idx in valid_indices:
                masked_q[idx] = q_vals[idx]

            chosen_idx = int(np.argmax(masked_q))
            return IDX_TO_ACTION[chosen_idx]

    def update(self, batch_size: int = 64) -> float:
        """Performs a Q-learning gradient descent step (gamma = 0)."""
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        self.q_net.train()
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 1-step contextual bandit: Target is simply the actualized reward R
        if self.gamma == 0.0:
            targets = rewards
        else:
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                targets = rewards + (1.0 - dones.float()) * self.gamma * next_q

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def train_on_history(
        self,
        states: list[np.ndarray],
        labels: list[int],
        amounts: list[float],
        allowed_actions_list: list[set[str]],
        epochs: int = 5,
        batch_size: int = 64,
    ) -> None:
        """Trains DQN policy offline on historical transactions."""
        for s, lbl, amt, allowed in zip(states, labels, amounts, allowed_actions_list):
            # Select action
            act = self.select_action(s, allowed, explore=True)
            act_idx = ACTION_TO_IDX[act]
            reward = self.reward_calc.compute_reward(act, lbl, amt)
            self.buffer.push(s, act_idx, reward, s, True)

        for _ in range(epochs):
            for _ in range(max(1, len(states) // batch_size)):
                self.update(batch_size)

        self.target_net.load_state_dict(self.q_net.state_dict())
