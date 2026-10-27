"""
Unit tests for normalized transaction event schema.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from fraudsense.ingest.schema import UNIFIED_COLUMNS, TransactionEvent, events_to_dataframe


def test_transaction_event_valid_instantiation():
    """Verify valid TransactionEvent instantiation and property access."""
    now = datetime.now(UTC)
    event = TransactionEvent(
        event_id="EVT_001",
        ts=now,
        amount=1500.50,
        payer_id="ACC_PAYER_123",
        payee_id="ACC_PAYEE_456",
        channel="PAYMENT",
        device_id="DEV_789",
        label=0,
        meta={"source": "test"},
    )
    assert event.event_id == "EVT_001"
    assert event.ts == now
    assert event.amount == 1500.50
    assert event.payer_id == "ACC_PAYER_123"
    assert event.payee_id == "ACC_PAYEE_456"
    assert event.channel == "PAYMENT"
    assert event.device_id == "DEV_789"
    assert event.label == 0
    assert event.meta == {"source": "test"}


def test_transaction_event_validation_errors():
    """Verify validation constraints on invalid attributes."""
    now = datetime.now(UTC)

    # Empty event_id
    with pytest.raises(ValueError):
        TransactionEvent(
            event_id="",
            ts=now,
            amount=100.0,
            payer_id="P1",
            payee_id="P2",
            channel="TRANSFER",
            device_id="D1",
            label=0,
        )

    # Negative amount
    with pytest.raises(ValueError):
        TransactionEvent(
            event_id="EVT_NEG",
            ts=now,
            amount=-50.0,
            payer_id="P1",
            payee_id="P2",
            channel="TRANSFER",
            device_id="D1",
            label=0,
        )

    # Invalid label
    with pytest.raises(ValueError):
        TransactionEvent(
            event_id="EVT_BAD_LABEL",
            ts=now,
            amount=100.0,
            payer_id="P1",
            payee_id="P2",
            channel="TRANSFER",
            device_id="D1",
            label=5,
        )

    # Invalid ts type
    with pytest.raises(TypeError):
        TransactionEvent(
            event_id="EVT_BAD_TS",
            ts="2025-01-01T00:00:00",  # string instead of datetime
            amount=100.0,
            payer_id="P1",
            payee_id="P2",
            channel="TRANSFER",
            device_id="D1",
            label=0,
        )


def test_events_to_dataframe():
    """Verify conversion of event sequence to canonical pandas DataFrame."""
    now = datetime.now(UTC)
    events = [
        TransactionEvent(
            event_id=f"EVT_{i}",
            ts=now,
            amount=float(i * 100),
            payer_id=f"P_{i}",
            payee_id=f"M_{i}",
            channel="TRANSFER",
            device_id=f"D_{i}",
            label=i % 2,
            meta={"custom_attr": i},
        )
        for i in range(5)
    ]

    df = events_to_dataframe(events, include_meta=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    for col in UNIFIED_COLUMNS:
        assert col in df.columns
    assert "meta" in df.columns
    assert df["amount"].dtype == float
    assert df["label"].dtype == int
