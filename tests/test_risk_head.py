"""
Unit tests for Calibrated Risk Head with Temperature Scaling.
"""

import numpy as np
import torch

from fraudsense.models.risk_head import CalibratedRiskHead, TemperatureScaler


def test_temperature_scaler():
    """Verify temperature scaling monotonically adjusts logits without flipping ranks."""
    scaler = TemperatureScaler()
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.8, 0.2]], dtype=torch.float)
    scaled = scaler(logits)

    assert scaled.shape == logits.shape
    # Ranking within each row must be preserved
    assert torch.argmax(logits, dim=1).tolist() == torch.argmax(scaled, dim=1).tolist()


def test_calibrated_risk_head_fit_predict():
    """Verify risk head MLP training, calibration, and probability outputs."""
    np.random.seed(42)
    torch.manual_seed(42)

    N = 300
    D = 82  # 64 (GAT) + 18 (Tabular)
    X = np.random.randn(N, D).astype(np.float32)
    y = (np.random.rand(N) < 0.15).astype(int)
    amounts = np.random.uniform(100.0, 75000.0, size=N)

    risk_head = CalibratedRiskHead(in_features=D)
    risk_head.fit(
        X_train=X[:200],
        y_train=y[:200],
        X_val=X[200:],
        y_val=y[200:],
        amounts_train=amounts[:200],
        amounts_val=amounts[200:],
        epochs=3,
    )

    probs_multi = risk_head.predict_multi_proba(X[200:])
    assert probs_multi.shape == (100, 3)
    # Probabilities must sum to 1
    assert np.allclose(probs_multi.sum(axis=1), 1.0, atol=1e-5)

    p_fraud = risk_head.predict_proba(X[200:])
    assert len(p_fraud) == 100
    assert (p_fraud >= 0.0).all() and (p_fraud <= 1.0).all()
