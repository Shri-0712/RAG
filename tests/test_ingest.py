"""
Unit tests for data loaders (PaySim, IEEE-CIS, European) and chronological replay.
"""

from datetime import UTC, datetime

import pandas as pd

from fraudsense.ingest.european_loader import load_european
from fraudsense.ingest.ieee_loader import load_ieee
from fraudsense.ingest.paysim_loader import load_paysim
from fraudsense.ingest.replay import EventReplayEngine
from fraudsense.ingest.schema import UNIFIED_COLUMNS, TransactionEvent


def test_load_paysim_sample():
    """Verify PaySim dataset loader produces valid canonical schema."""
    events = load_paysim(n_rows=50)
    assert len(events) == 50
    assert all(isinstance(e, TransactionEvent) for e in events)

    first = events[0]
    assert first.event_id.startswith("PAYSIM_")
    assert isinstance(first.ts, datetime)
    assert first.amount > 0
    assert first.payer_id.startswith("C") or first.payer_id.startswith("M")
    assert first.payee_id.startswith("C") or first.payee_id.startswith("M")
    assert first.channel in ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    assert first.device_id.startswith("DEV_")
    assert first.label in (0, 1)

    # Test DataFrame return
    df = load_paysim(n_rows=25, as_dataframe=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 25
    for col in UNIFIED_COLUMNS:
        assert col in df.columns


def test_load_ieee_sample():
    """Verify IEEE-CIS dataset loader joins tables and matches schema."""
    events = load_ieee(n_rows=50)
    assert len(events) == 50
    assert all(isinstance(e, TransactionEvent) for e in events)

    first = events[0]
    assert first.event_id.startswith("IEEE_")
    assert isinstance(first.ts, datetime)
    assert first.amount > 0
    assert first.payer_id.startswith("CARD_")
    assert first.payee_id.startswith("MERCH_")
    assert first.channel != ""
    assert first.device_id.startswith("DEV_")
    assert first.label in (0, 1)

    # Test DataFrame return
    df = load_ieee(n_rows=25, as_dataframe=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 25
    for col in UNIFIED_COLUMNS:
        assert col in df.columns


def test_load_european_sample():
    """Verify European Credit Card dataset loader matches schema and preserves PCA meta."""
    events = load_european(n_rows=50)
    assert len(events) == 50
    assert all(isinstance(e, TransactionEvent) for e in events)

    first = events[0]
    assert first.event_id.startswith("EU_")
    assert isinstance(first.ts, datetime)
    assert first.amount >= 0
    assert first.payer_id.startswith("EU_CARD_")
    assert first.payee_id.startswith("EU_MERCH_")
    assert first.channel == "ONLINE_POS"
    assert first.label in (0, 1)
    assert "V1" in first.meta
    assert "V28" in first.meta

    # Test DataFrame return
    df = load_european(n_rows=25, as_dataframe=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 25
    for col in UNIFIED_COLUMNS:
        assert col in df.columns


def test_event_replay_engine_ordering_and_masking():
    """Verify EventReplayEngine enforces chronological order and label concealment."""
    paysim_events = load_paysim(n_rows=40)
    engine = EventReplayEngine(events=paysim_events)

    assert len(engine) == 40

    last_ts = None
    stream_count = 0
    for scoring_event, true_label in engine.stream(mask_label=True):
        stream_count += 1
        # Check label is masked for scoring
        assert scoring_event.label == -1
        assert true_label in (0, 1)

        # Check chronological monotonicity
        if last_ts is not None:
            assert scoring_event.ts >= last_ts
        last_ts = scoring_event.ts

    assert stream_count == 40


def test_event_replay_label_maturity_delay():
    """Verify labels become available only after label_maturity_sim_days have passed."""
    # Create two synthetic events spaced 8 days apart
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    from datetime import timedelta

    e1 = TransactionEvent(
        event_id="E_DAY_0",
        ts=base,
        amount=100.0,
        payer_id="P1",
        payee_id="M1",
        channel="TRANSFER",
        device_id="D1",
        label=1,
    )
    e2 = TransactionEvent(
        event_id="E_DAY_8",
        ts=base + timedelta(days=8),
        amount=200.0,
        payer_id="P2",
        payee_id="M2",
        channel="TRANSFER",
        device_id="D2",
        label=0,
    )

    engine = EventReplayEngine(events=[e1, e2])

    stream_iter = engine.stream(mask_label=True)

    # Yield first event at day 0
    ev1, _ = next(stream_iter)
    assert ev1.event_id == "E_DAY_0"
    # At day 0, label for E_DAY_0 should NOT be matured (requires 7 days)
    assert engine.get_matured_label("E_DAY_0") is None

    # Yield second event at day 8
    ev2, _ = next(stream_iter)
    assert ev2.event_id == "E_DAY_8"
    # At day 8, label for E_DAY_0 should now be matured
    assert engine.get_matured_label("E_DAY_0") == 1
    # But E_DAY_8's label is NOT matured yet
    assert engine.get_matured_label("E_DAY_8") is None
