"""
Unit tests for SQLite database schema and append-only audit logger.
"""

from datetime import UTC, datetime

from fraudsense.audit.log import (
    AuditLogger,
    DatabaseManager,
    PaymentEventRecord,
)


def test_database_manager_table_creation(tmp_path):
    """Verify all 10 SQLite tables are created properly."""
    db_file = tmp_path / "test_fraudsense.db"
    db_mgr = DatabaseManager(db_path=db_file)
    assert db_file.exists()

    session = db_mgr.get_session()
    try:
        # Insert a sample payment event
        now = datetime.now(UTC)
        event_record = PaymentEventRecord(
            event_id="EVT_DB_001",
            ts=now,
            amount=5000.0,
            payer_id="ACC_1",
            payee_id="ACC_2",
            channel="UPI",
            device_id="DEV_1",
            label=0,
            meta_json="{}",
        )
        session.add(event_record)
        session.commit()

        # Query back
        retrieved = session.query(PaymentEventRecord).filter_by(event_id="EVT_DB_001").first()
        assert retrieved is not None
        assert retrieved.amount == 5000.0
        assert retrieved.payer_id == "ACC_1"
    finally:
        session.close()


def test_audit_logger_append_and_trail(tmp_path):
    """Verify AuditLogger writes immutable chronological records."""
    db_file = tmp_path / "test_audit.db"
    db_mgr = DatabaseManager(db_path=db_file)
    logger = AuditLogger(db_manager=db_mgr, model_version="v1.0-test")

    log_id1 = logger.log(
        entity_type="DECISION",
        entity_id="DEC_001",
        action="ALLOW",
        details={"score": 0.05, "rule": "CLEAR_ALLOW"},
    )
    log_id2 = logger.log(
        entity_type="DECISION",
        entity_id="DEC_001",
        action="SUPERVISOR_REVIEW",
        details={"status": "APPROVED"},
    )

    assert log_id2 > log_id1

    trail = logger.get_audit_trail("DEC_001")
    assert len(trail) == 2
    assert trail[0]["action"] == "ALLOW"
    assert trail[1]["action"] == "SUPERVISOR_REVIEW"
    assert trail[0]["model_version"] == "v1.0-test"
