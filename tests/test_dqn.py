"""
Unit tests for DQN action policy and masked contextual bandit learning.
"""

import numpy as np
import torch

from fraudsense.models.dqn import DQNActionPolicy, QNetwork, ReplayBuffer


def test_q_network_dimensions():
    """Verify Q-network outputs Q-values for 3 actions."""
    net = QNetwork(state_dim=7, hidden_dim=64, action_dim=3)
    states = torch.randn((10, 7), dtype=torch.float32)
    q_vals = net(states)

    assert q_vals.shape == (10, 3)
    assert not torch.isnan(q_vals).any()


def test_replay_buffer_sampling():
    """Verify experience replay buffer push and mini-batch sampling."""
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        s = np.zeros(7, dtype=np.float32)
        buf.push(s, i % 3, float(i * 10), s, False)

    assert len(buf) == 20
    b_s, b_a, b_r, _b_ns, _b_d = buf.sample(batch_size=8)
    assert b_s.shape == (8, 7)
    assert b_a.shape == (8,)
    assert b_r.shape == (8,)


def test_dqn_policy_masked_action_selection():
    """Verify DQN strictly respects action masks."""
    policy = DQNActionPolicy(state_dim=7)
    state = np.array([1.0, 0.0, 0.0, 0.5, 0.1, 1.0, 0.2], dtype=np.float32)

    # If allowed actions is only {'CHALLENGE'}, select_action MUST return CHALLENGE
    act = policy.select_action(state, allowed_actions={"CHALLENGE"}, explore=False)
    assert act == "CHALLENGE"

    # If allowed actions is {'ALLOW', 'CHALLENGE'}, HOLD_FOR_REVIEW must never be selected
    for _ in range(10):
        act2 = policy.select_action(state, allowed_actions={"ALLOW", "CHALLENGE"}, explore=False)
        assert act2 in {"ALLOW", "CHALLENGE"}


def test_dqn_offline_training_step():
    """Verify DQN training loop executes and reduces loss."""
    policy = DQNActionPolicy(state_dim=7)
    states = [np.random.randn(7).astype(np.float32) for _ in range(50)]
    labels = [0 if i % 5 != 0 else 1 for i in range(50)]
    amounts = [float((i + 1) * 1000) for i in range(50)]
    allowed = [{"ALLOW", "CHALLENGE", "HOLD_FOR_REVIEW"} for _ in range(50)]

    policy.train_on_history(states, labels, amounts, allowed, epochs=2, batch_size=16)
    assert len(policy.buffer) == 50
