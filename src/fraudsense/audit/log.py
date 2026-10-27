"""
Database models and append-only audit logger for FraudSense.
Persists all transactions, scores, decisions, cases, and regulatory drafts to SQLite.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ..config import get_project_root

Base = declarative_base()


class PaymentEventRecord(Base):
    __tablename__ = "payment_event"
    event_id = Column(String, primary_key=True)
    ts = Column(DateTime, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    payer_id = Column(String, nullable=False, index=True)
    payee_id = Column(String, nullable=False, index=True)
    channel = Column(String, nullable=False)
    device_id = Column(String, nullable=False)
    label = Column(Integer, nullable=True)
    meta_json = Column(Text, nullable=True)


class FeatureSnapshotRecord(Base):
    __tablename__ = "feature_snapshot"
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, nullable=False, index=True)
    features_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class ScoreRecord(Base):
    __tablename__ = "score_record"
    score_id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, nullable=False, index=True)
    p_fraud = Column(Float, nullable=False)
    conformal_set_json = Column(Text, nullable=False)
    model_version = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class DecisionRecord(Base):
    __tablename__ = "decision_record"
    decision_id = Column(String, primary_key=True)
    event_id = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False)
    reason = Column(Text, nullable=False)
    policy_clauses_json = Column(Text, nullable=False)
    is_fail_safe = Column(Boolean, default=False)
    was_overridden = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class InteractionRecord(Base):
    __tablename__ = "interaction_record"
    interaction_id = Column(String, primary_key=True)
    decision_id = Column(String, nullable=False, index=True)
    event_id = Column(String, nullable=False)
    channel_type = Column(String, nullable=False)  # SMS / IN_APP
    template_id = Column(String, nullable=False)
    status = Column(String, nullable=False)  # SENT, CONFIRMED, DENIED, TIMED_OUT
    simulated_flag = Column(String, default="SIMULATED")
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class CaseRecord(Base):
    __tablename__ = "case_record"
    case_id = Column(String, primary_key=True)
    event_id = Column(String, nullable=False, index=True)
    decision_id = Column(String, nullable=False)
    state = Column(String, nullable=False, index=True)  # OPENED, ENRICHED, etc.
    amount = Column(Float, nullable=False)
    assigned_analyst = Column(String, nullable=True)
    requires_supervisor = Column(Boolean, default=False)
    supervisor_approved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC))


class OutcomeRecord(Base):
    __tablename__ = "outcome_record"
    outcome_id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String, nullable=False, index=True)
    decision_id = Column(String, nullable=False)
    analyst_decision = Column(String, nullable=False)
    reason_code = Column(String, nullable=False)
    clause_ids_json = Column(Text, nullable=False)
    simulated_flag = Column(String, default="SIMULATED")
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class ReportingRecord(Base):
    __tablename__ = "reporting_record"
    report_id = Column(String, primary_key=True)
    case_id = Column(String, nullable=False, index=True)
    report_type = Column(String, nullable=False)  # CPFIR, FMR
    draft_json = Column(Text, nullable=False)
    simulated_flag = Column(String, default="SIMULATED")
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class RewardRecord(Base):
    __tablename__ = "reward_record"
    reward_id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(String, nullable=False, index=True)
    event_id = Column(String, nullable=False)
    action = Column(String, nullable=False)
    true_label = Column(Integer, nullable=False)
    reward_amount = Column(Float, nullable=False)
    cost_amount = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class AuditLogRecord(Base):
    __tablename__ = "audit_log"
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String, nullable=False)
    entity_id = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False)
    details_json = Column(Text, nullable=False)
    model_version = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC), index=True)


class DatabaseManager:
    """Manages SQLite database connections and table migrations."""

    def __init__(self, db_path: (str or Path) | None = None):
        if db_path is None:
            processed_dir = get_project_root() / "data" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            db_path = processed_dir / "fraudsense.db"
        else:
            db_path = Path(db_path)

        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.create_tables()

    def create_tables(self) -> None:
        """Creates all required SQLite tables."""
        Base.metadata.create_all(self.engine)

    def get_session(self):
        """Yields an active database session."""
        return self.SessionLocal()


class AuditLogger:
    """
    Append-only audit logger.
    Guarantees every model decision, prediction, and analyst action is permanently recorded.
    """

    def __init__(self, db_manager: DatabaseManager | None = None, model_version: str = "v1.0.0"):
        self.db = db_manager or DatabaseManager()
        self.model_version = model_version

    def log(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        details: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> int:
        """
        Appends an immutable audit record.

        Returns:
            log_id primary key of inserted audit record.
        """
        session = self.db.get_session()
        try:
            entry = AuditLogRecord(
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                details_json=json.dumps(details),
                model_version=self.model_version,
                timestamp=timestamp or datetime.now(UTC),
            )
            session.add(entry)
            session.commit()
            return entry.log_id
        finally:
            session.close()

    def get_audit_trail(self, entity_id: str) -> list[dict[str, Any]]:
        """Retrieves chronological audit trail for a specific entity."""
        session = self.db.get_session()
        try:
            records = (
                session.query(AuditLogRecord)
                .filter(AuditLogRecord.entity_id == entity_id)
                .order_by(AuditLogRecord.timestamp.asc())
                .all()
            )
            return [
                {
                    "log_id": r.log_id,
                    "entity_type": r.entity_type,
                    "entity_id": r.entity_id,
                    "action": r.action,
                    "details": json.loads(r.details_json),
                    "model_version": r.model_version,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in records
            ]
        finally:
            session.close()
