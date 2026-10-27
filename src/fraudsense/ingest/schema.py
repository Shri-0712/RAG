"""
Normalized transaction event schema for FraudSense.
Defines the canonical event representation required by all downstream pipelines.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

UNIFIED_COLUMNS: list[str] = [
    "event_id",
    "ts",
    "amount",
    "payer_id",
    "payee_id",
    "channel",
    "device_id",
    "label",
]


@dataclass(frozen=True)
class TransactionEvent:
    """
    Normalized payment transaction event representation.

    Attributes:
        event_id: Unique transaction identifier.
        ts: Event timestamp in UTC.
        amount: Transaction amount (in INR or normalized currency equivalent).
        payer_id: Originating account/card/customer identifier.
        payee_id: Destination account/merchant identifier.
        channel: Transaction channel (e.g., PAYMENT, TRANSFER, ONLINE, POS).
        device_id: Unique device fingerprint or surrogate device cluster ID.
        label: Ground truth label (0 = legitimate, 1 = fraudulent, -1/None = unobserved/masked).
        meta: Optional dataset-specific attributes preserved for feature engineering.
    """

    event_id: str
    ts: datetime
    amount: float
    payer_id: str
    payee_id: str
    channel: str
    device_id: str
    label: int | None = None
    meta: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("event_id must be a non-empty string")
        if not isinstance(self.ts, datetime):
            raise TypeError("ts must be a valid datetime instance")
        if self.amount < 0:
            raise ValueError(f"amount must be non-negative, got {self.amount}")
        if self.label is not None and self.label not in (0, 1, -1):
            raise ValueError(f"label must be 0, 1, -1, or None, got {self.label}")

    def to_dict(self, include_meta: bool = False) -> dict[str, Any]:
        """Convert event to dictionary."""
        d = {
            "event_id": self.event_id,
            "ts": self.ts,
            "amount": self.amount,
            "payer_id": self.payer_id,
            "payee_id": self.payee_id,
            "channel": self.channel,
            "device_id": self.device_id,
            "label": self.label,
        }
        if include_meta:
            d["meta"] = self.meta
        return d


def events_to_dataframe(events: list[TransactionEvent], include_meta: bool = False) -> pd.DataFrame:
    """
    Utility function converting a list of TransactionEvents into a Pandas DataFrame
    with standard column dtypes and strictly matching UNIFIED_COLUMNS.
    """
    if not events:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    records = [e.to_dict(include_meta=include_meta) for e in events]
    df = pd.DataFrame.from_records(records)
    df["ts"] = pd.to_datetime(df["ts"])
    df["amount"] = df["amount"].astype(float)
    df["label"] = df["label"].astype(int)
    return df
