"""
Calibrated Risk Head for FraudSense.
Takes concatenated GAT embedding (64-dim) and tabular features (18-dim).
Outputs calibrated probability P(fraud) with Temperature Scaling on validation split.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config import load_config


class TemperatureScaler(nn.Module):
    """
    Post-hoc probability calibrator using Temperature Scaling.
    Scales logits by a learned positive scalar T: logits / T.
    Guarantees monotonic transformation preserving ranking while rectifying overconfidence.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Clamp temperature to avoid division by zero or extreme scaling
        temp = torch.clamp(self.temperature, min=0.05, max=10.0)
        return logits / temp

    def fit(
        self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50
    ) -> "TemperatureScaler":
        """Optimizes temperature parameter using L-BFGS on validation NLL."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self


class RiskHeadMLP(nn.Module):
    """
    Multilayer Perceptron mapping (GAT embedding + tabular features) to class logits.
    Outputs 3 classes: [0: ALLOW, 1: CHALLENGE, 2: FRAUD].
    """

    def __init__(
        self,
        in_features: int = 82,  # 64 (GAT) + 18 (Tabular)
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        if h.shape[0] > 1:
            h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        logits = self.out(h)
        return logits


class CalibratedRiskHead:
    """
    Full Risk Head service combining MLP classifier with Temperature Scaling.
    """

    def __init__(
        self,
        in_features: int = 82,
        config: dict[str, Any] | None = None,
    ):
        self.config = config or load_config()
        self.model = RiskHeadMLP(in_features=in_features)
        self.scaler = TemperatureScaler()
        self.is_calibrated: bool = False
        self.is_fitted: bool = False

    def _prepare_labels(
        self, y_binary: np.ndarray, amounts: np.ndarray | None = None
    ) -> torch.Tensor:
        """
        Maps binary fraud labels (0 or 1) and amounts to 3-class targets:
        - Class 0: ALLOW (legitimate low value)
        - Class 1: CHALLENGE (legitimate high value >= 50,000)
        - Class 2: FRAUD (fraudulent)
        """
        high_val = float(self.config.get("decision", {}).get("high_value_threshold", 50000))
        y_multi = []
        for i, lbl in enumerate(y_binary):
            if lbl == 1:
                y_multi.append(2)  # FRAUD
            elif amounts is not None and amounts[i] >= high_val:
                y_multi.append(1)  # CHALLENGE
            else:
                y_multi.append(0)  # ALLOW
        return torch.tensor(y_multi, dtype=torch.long)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        amounts_train: np.ndarray | None = None,
        amounts_val: np.ndarray | None = None,
        epochs: int = 15,
        batch_size: int = 256,
        lr: float = 0.002,
    ) -> "CalibratedRiskHead":
        """Trains MLP on concatenated features and calibrates on validation set."""
        self.model.train()
        device = torch.device("cpu")
        self.model.to(device)

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = self._prepare_labels(y_train, amounts_train)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        # Class weighting for class imbalance
        counts = torch.bincount(y_tensor, minlength=3).float()
        weights = 1.0 / (counts + 1.0)
        weights = weights / weights.sum()
        criterion = nn.CrossEntropyLoss(weight=weights)

        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        self.model.eval()

        if X_val is not None and y_val is not None:
            self.calibrate(X_val, y_val, amounts_val)

        return self

    def calibrate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        amounts_val: np.ndarray | None = None,
    ) -> "CalibratedRiskHead":
        """Fits temperature scaling parameter on held-out validation split."""
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_val, dtype=torch.float32)
            val_logits = self.model(x_t)
            y_t = self._prepare_labels(y_val, amounts_val)

        self.scaler.fit(val_logits, y_t)
        self.is_calibrated = True
        return self

    def predict_multi_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns calibrated 3-class probability distribution:
        shape (N, 3) over [p_allow, p_challenge, p_fraud].
        """
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model(x_t)
            if self.is_calibrated:
                logits = self.scaler(logits)
            probs = F.softmax(logits, dim=1).numpy()
        return probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns scalar calibrated P(fraud) probability, shape (N,).
        P(fraud) = p_class_2.
        """
        probs = self.predict_multi_proba(X)
        return probs[:, 2]
