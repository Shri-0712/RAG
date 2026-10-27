"""
Real-time feature service for FraudSense.
Maintains stateful historical statistics per account to extract causal tabular features
with strictly zero lookahead leakage.
"""

from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from ..config import load_config
from ..ingest.schema import TransactionEvent

FEATURE_NAMES: list[str] = [
    "amount",
    "log_amount",
    "payer_tx_count_1h",
    "payer_tx_count_24h",
    "payer_tx_count_72h",
    "payer_sum_amount_24h",
    "payer_avg_amount_24h",
    "amount_to_avg_ratio",
    "time_since_last_tx_sec",
    "distinct_payees_24h",
    "distinct_devices_24h",
    "hour_of_day",
    "day_of_week",
    "is_transfer",
    "is_cash_out",
    "orig_balance_before",
    "orig_balance_after",
    "balance_error_orig",
]


class FeatureService:
    """
    Computes streaming tabular features for payment events.
    Maintains a rolling window of past transactions per payer account.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        # Per-payer rolling history: deque of (timestamp, amount, payee_id, device_id)
        self._payer_history: dict[str, deque] = {}
        # Per-payer last transaction timestamp
        self._payer_last_tx: dict[str, datetime] = {}

    def extract_features(
        self,
        event: TransactionEvent,
        update_state: bool = True,
    ) -> np.ndarray:
        """
        Extracts causal feature vector for the given transaction.

        Args:
            event: The incoming TransactionEvent.
            update_state: If True, registers the transaction in payer history
                          AFTER feature extraction (strict causal order).

        Returns:
            1D numpy float32 array of shape (len(FEATURE_NAMES),).
        """
        ts = event.ts
        amt = float(event.amount)
        payer = event.payer_id

        # Retrieve or initialize history
        history = self._payer_history.setdefault(payer, deque())

        # Purge transactions older than 72 hours relative to current event ts
        cutoff_72h = ts.timestamp() - (72 * 3600)
        while history and history[0][0].timestamp() < cutoff_72h:
            history.popleft()

        # Compute historical aggregates
        ts_now = ts.timestamp()
        cutoff_1h = ts_now - 3600
        cutoff_24h = ts_now - 86400

        count_1h = 0
        count_24h = 0
        count_72h = len(history)
        sum_amt_24h = 0.0
        payees_24h = set()
        devices_24h = set()

        for past_ts, past_amt, past_payee, past_dev in history:
            p_time = past_ts.timestamp()
            if p_time >= cutoff_1h:
                count_1h += 1
            if p_time >= cutoff_24h:
                count_24h += 1
                sum_amt_24h += past_amt
                payees_24h.add(past_payee)
                devices_24h.add(past_dev)

        avg_amt_24h = (sum_amt_24h / count_24h) if count_24h > 0 else amt
        ratio = amt / max(1.0, avg_amt_24h)

        last_time = self._payer_last_tx.get(payer)
        time_since_last = (ts - last_time).total_seconds() if last_time else 86400.0
        time_since_last = max(0.0, time_since_last)

        # Channel indicator features
        ch = str(event.channel).upper()
        is_transfer = 1.0 if "TRANSFER" in ch else 0.0
        is_cash_out = 1.0 if "CASH_OUT" in ch else 0.0

        # Balance dynamics from metadata if available
        meta = event.meta or {}
        orig_before = float(meta.get("oldbalanceOrg", 0.0))
        orig_after = float(meta.get("newbalanceOrig", 0.0))
        bal_error = float((orig_before - orig_after) - amt)

        feature_vector = np.array(
            [
                amt,
                np.log1p(max(0.0, amt)),
                float(count_1h),
                float(count_24h),
                float(count_72h),
                sum_amt_24h,
                avg_amt_24h,
                ratio,
                time_since_last,
                float(len(payees_24h)),
                float(len(devices_24h)),
                float(ts.hour),
                float(ts.weekday()),
                is_transfer,
                is_cash_out,
                orig_before,
                orig_after,
                bal_error,
            ],
            dtype=np.float32,
        )

        if update_state:
            history.append((ts, amt, event.payee_id, event.device_id))
            self._payer_last_tx[payer] = ts

        return feature_vector

    def extract_batch(
        self,
        events: list[TransactionEvent],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extracts features sequentially across a list of chronological events.

        Returns:
            Tuple of (X_matrix of shape (N, D), FEATURE_NAMES).
        """
        features_list = [self.extract_features(e, update_state=True) for e in events]
        if not features_list:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), FEATURE_NAMES
        return np.vstack(features_list), FEATURE_NAMES
