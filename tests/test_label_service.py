"""Tests for final-label maturity, selection-bias correction, and reward attribution."""

from datetime import UTC, datetime, timedelta

import pytest

from fraudsense.audit.log import DatabaseManager, DecisionRecord, PaymentEventRecord, RewardRecord
from fraudsense.feedback.label_service import LabelService, LabelSource


def _seed_decision(db, decision_id="DEC_1", action="ALLOW", amount=1000.0, created_at=None):
    created_at = created_at or datetime(2025, 1, 1, tzinfo=UTC)
    session = db.get_session()
    try:
        session.add(
            PaymentEventRecord(
                event_id=f"EVT_{decision_id}",
                ts=created_at,
                amount=amount,
                payer_id="PAYER",
                payee_id="PAYEE",
                channel="UPI",
                device_id="DEVICE",
                meta_json="{}",
            )
        )
        session.add(
            DecisionRecord(
                decision_id=decision_id,
                event_id=f"EVT_{decision_id}",
                action=action,
                reason="test",
                policy_clauses_json="[]",
                created_at=created_at,
            )
        )
        session.commit()
    finally:
        session.close()


def _service(tmp_path):
    return DatabaseManager(tmp_path / "feedback.db"), {
        "labels": {"label_maturity_sim_days": 7, "provisional_labels_in_training": False},
        "feedback": {"random_audit_rate": 0.5},
        "reward": {
            "alpha": 1.0,
            "lambda": 0.2,
            "u": 5,
            "gamma": 0.9,
            "c_a": 150,
            "h": 300,
            "rho": 0.01,
        },
    }


def test_provisional_analyst_label_never_enters_training_buffer(tmp_path):
    db, config = _service(tmp_path)
    _seed_decision(db)
    service = LabelService(db, config=config)
    service.ingest_analyst_decision("DEC_1", "CONFIRM_FRAUD", datetime(2025, 1, 1, tzinfo=UTC))
    assert service.mature_labels(datetime(2025, 2, 1, tzinfo=UTC)) == []
    assert service.training_buffer() == []


def test_final_label_matures_after_window_and_creates_reward(tmp_path):
    db, config = _service(tmp_path)
    _seed_decision(db, action="HOLD_FOR_REVIEW", amount=10000.0)
    service = LabelService(db, config=config)
    received = datetime(2025, 1, 1, tzinfo=UTC)
    service.ingest_label("DEC_1", 1, LabelSource.CASE_CLOSURE, received)
    assert service.mature_labels(received + timedelta(days=6, seconds=1)) == []
    matured = service.mature_labels(received + timedelta(days=7))
    assert len(matured) == 1
    assert matured[0].reward_amount == 8850.0
    session = db.get_session()
    try:
        assert session.query(RewardRecord).filter_by(decision_id="DEC_1").count() == 1
    finally:
        session.close()


def test_allow_random_audit_receives_inverse_propensity_weight(tmp_path):
    db, config = _service(tmp_path)
    _seed_decision(db)
    service = LabelService(db, config=config)
    received = datetime(2025, 1, 1, tzinfo=UTC)
    service.ingest_label("DEC_1", 0, "DISPUTE_OUTCOME", received, is_random_audit=True)
    example = service.mature_labels(received + timedelta(days=7))[0]
    assert example.importance_weight == 2.0


def test_unknown_decision_is_rejected(tmp_path):
    db, config = _service(tmp_path)
    service = LabelService(db, config=config)
    with pytest.raises(ValueError, match="Unknown decision_id"):
        service.ingest_label("DEC_UNKNOWN", 1, "CASE_CLOSURE")


def test_weekly_allow_audits_only_select_allow_decisions(tmp_path):
    db, config = _service(tmp_path)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    _seed_decision(db, "DEC_ALLOW", "ALLOW", created_at=start)
    _seed_decision(db, "DEC_HOLD", "HOLD_FOR_REVIEW", created_at=start)
    selected = LabelService(db, config=config).select_weekly_allow_audits(start, seed=1)
    assert set(selected).issubset({"DEC_ALLOW"})
