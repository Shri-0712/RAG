"""
Unit tests for streaming causal feature service.
"""

from datetime import UTC, datetime, timedelta

import numpy as np

from fraudsense.features.service import FEATURE_NAMES, FeatureService
from fraudsense.ingest.schema import TransactionEvent


def test_feature_service_dimensions_and_causality():
    """Verify feature vector length, name alignment, and causal state progression."""
    service = FeatureService()
    base_time = datetime(2025, 1, 1, 8, 0, 0, tzinfo=UTC)

    # First event for ACC_1
    e1 = TransactionEvent(
        event_id="E1",
        ts=base_time,
        amount=1000.0,
        payer_id="ACC_1",
        payee_id="MERCH_1",
        channel="PAYMENT",
        device_id="DEV_1",
        label=0,
    )
    f1 = service.extract_features(e1, update_state=True)

    assert len(f1) == len(FEATURE_NAMES)
    # On first transaction, past counts in 1h/24h must be 0 (no lookahead)
    idx_1h = FEATURE_NAMES.index("payer_tx_count_1h")
    idx_24h = FEATURE_NAMES.index("payer_tx_count_24h")
    idx_amt = FEATURE_NAMES.index("amount")

    assert f1[idx_amt] == 1000.0
    assert f1[idx_1h] == 0.0
    assert f1[idx_24h] == 0.0

    # Second event 10 minutes later for same payer
    e2 = TransactionEvent(
        event_id="E2",
        ts=base_time + timedelta(minutes=10),
        amount=2500.0,
        payer_id="ACC_1",
        payee_id="MERCH_2",
        channel="PAYMENT",
        device_id="DEV_2",
        label=0,
    )
    f2 = service.extract_features(e2, update_state=True)

    # Now past 1h count must be 1, past 24h sum amount must be 1000.0
    assert f2[idx_1h] == 1.0
    assert f2[idx_24h] == 1.0
    idx_sum = FEATURE_NAMES.index("payer_sum_amount_24h")
    assert f2[idx_sum] == 1000.0

    # Distinct payees in 24h before e2 was 1 (MERCH_1)
    idx_payees = FEATURE_NAMES.index("distinct_payees_24h")
    assert f2[idx_payees] == 1.0


def test_feature_service_batch_extraction():
    """Verify batch extraction preserves event order and dimensions."""
    service = FeatureService()
    base_time = datetime(2025, 1, 1, 8, 0, 0, tzinfo=UTC)

    events = [
        TransactionEvent(
            event_id=f"EVT_{i}",
            ts=base_time + timedelta(minutes=i * 5),
            amount=float((i + 1) * 200),
            payer_id=f"ACC_{i % 3}",
            payee_id=f"MERCH_{i}",
            channel="PAYMENT",
            device_id=f"DEV_{i % 2}",
            label=0,
        )
        for i in range(10)
    ]

    X, names = service.extract_batch(events)
    assert X.shape == (10, len(FEATURE_NAMES))
    assert names == FEATURE_NAMES
    assert not np.isnan(X).any()
