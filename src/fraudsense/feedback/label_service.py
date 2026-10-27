"""Delayed final-label management and reward attribution for FraudSense.

This module only handles simulated feedback.  Provisional analyst conclusions
are recorded for traceability but can never enter the retraining buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from sqlalchemy import select

from ..audit.log import (
    AuditLogger,
    DatabaseManager,
    DecisionRecord,
    PaymentEventRecord,
    RewardRecord,
)
from ..config import load_config
from .reward import RewardCalculator
from .weights import PropensityWeightCalculator


class LabelSource(StrEnum):
    """Permitted sources of simulated feedback."""

    ANALYST_DECISION = "ANALYST_DECISION"
    DISPUTE_OUTCOME = "DISPUTE_OUTCOME"
    CASE_CLOSURE = "CASE_CLOSURE"


FINAL_SOURCES = frozenset({LabelSource.DISPUTE_OUTCOME, LabelSource.CASE_CLOSURE})
ANALYST_FRAUD_DECISIONS = frozenset({"CONFIRM_FRAUD", "ATTEMPTED_FRAUD"})


@dataclass(frozen=True)
class TrainingExample:
    """One final, matured, rewarded observation eligible for retraining."""

    decision_id: str
    event_id: str
    label: int
    action: str
    amount: float
    source: LabelSource
    matured_at: datetime
    importance_weight: float
    reward_amount: float


@dataclass
class _LabelEntry:
    decision_id: str
    event_id: str
    label: int
    source: LabelSource
    received_at: datetime
    maturity_at: datetime | None
    is_random_audit: bool
    matured: bool = False


class LabelService:
    """Accepts simulated labels, matures final feedback, and attributes rewards.

    The in-memory queue deliberately accepts labels only through this service;
    final labels are also written to the immutable audit trail.  A fresh service
    instance can reconstruct its training buffer from persisted rewards.
    """

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        config: dict[str, Any] | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.config = config or load_config()
        self.db = db_manager or DatabaseManager()
        self.audit_logger = audit_logger or AuditLogger(db_manager=self.db)
        self.reward_calculator = RewardCalculator(self.config)
        self.weight_calculator = PropensityWeightCalculator(self.config)
        self.maturity_days = int(self.config.get("labels", {}).get("label_maturity_sim_days", 7))
        self._pending: dict[str, _LabelEntry] = {}
        self._training_buffer: dict[str, TrainingExample] = {}

    @staticmethod
    def analyst_decision_to_label(analyst_decision: str) -> int:
        """Maps the permitted analyst decisions to a provisional binary label."""
        decision = analyst_decision.upper()
        if decision in ANALYST_FRAUD_DECISIONS:
            return 1
        if decision == "LEGITIMATE":
            return 0
        if decision == "NEED_INFO":
            raise ValueError("NEED_INFO does not provide a label")
        raise ValueError(f"Unknown analyst decision: {analyst_decision}")

    def ingest_analyst_decision(
        self, decision_id: str, analyst_decision: str, received_at: datetime | None = None
    ) -> None:
        """Records a provisional analyst conclusion, never as a training label."""
        self.ingest_label(
            decision_id=decision_id,
            label=self.analyst_decision_to_label(analyst_decision),
            source=LabelSource.ANALYST_DECISION,
            received_at=received_at,
        )

    def ingest_label(
        self,
        decision_id: str,
        label: int,
        source: LabelSource | str,
        received_at: datetime | None = None,
        is_random_audit: bool = False,
    ) -> None:
        """Ingests feedback; only dispute and closure labels enter the maturity queue."""
        if label not in (0, 1):
            raise ValueError("label must be 0 (legitimate) or 1 (fraud)")
        source = LabelSource(source)
        received_at = self._as_utc(received_at or datetime.now(UTC))
        event_id, action, amount = self._lookup_decision(decision_id)
        del action, amount
        maturity_at = (
            received_at + timedelta(days=self.maturity_days) if source in FINAL_SOURCES else None
        )
        entry = _LabelEntry(
            decision_id=decision_id,
            event_id=event_id,
            label=label,
            source=source,
            received_at=received_at,
            maturity_at=maturity_at,
            is_random_audit=is_random_audit,
        )
        if source in FINAL_SOURCES:
            # A final source supersedes any earlier feedback for that decision.
            self._pending[decision_id] = entry
        self.audit_logger.log(
            entity_type="LABEL",
            entity_id=decision_id,
            action="LABEL_RECEIVED",
            details={
                "event_id": event_id,
                "label": label,
                "source": source.value,
                "provisional": source == LabelSource.ANALYST_DECISION,
                "maturity_at": maturity_at.isoformat() if maturity_at else None,
                "simulated_flag": "SIMULATED",
            },
            timestamp=received_at,
        )

    def mature_labels(self, simulated_now: datetime) -> list[TrainingExample]:
        """Matures final labels due at ``simulated_now`` and writes reward records."""
        now = self._as_utc(simulated_now)
        matured: list[TrainingExample] = []
        for decision_id, entry in list(self._pending.items()):
            if entry.matured or entry.maturity_at is None or entry.maturity_at > now:
                continue
            example = self._attribute_reward(entry, now)
            entry.matured = True
            self._training_buffer[decision_id] = example
            del self._pending[decision_id]
            matured.append(example)
        return matured

    def training_buffer(self) -> list[TrainingExample]:
        """Returns only matured, final, rewarded samples, ordered by maturity."""
        return sorted(self._training_buffer.values(), key=lambda item: item.matured_at)

    def select_weekly_allow_audits(
        self, week_start: datetime, seed: int | None = None
    ) -> list[str]:
        """Selects a deterministic simulated random audit sample of ALLOW decisions."""
        import random

        start = self._as_utc(week_start)
        end = start + timedelta(days=7)
        session = self.db.get_session()
        try:
            rows = (
                session.execute(
                    select(DecisionRecord.decision_id)
                    .where(DecisionRecord.action == "ALLOW")
                    .where(DecisionRecord.created_at >= start)
                    .where(DecisionRecord.created_at < end)
                )
                .scalars()
                .all()
            )
        finally:
            session.close()
        rng = random.Random(seed)
        selected = [
            decision_id
            for decision_id in rows
            if rng.random() < self.weight_calculator.random_audit_rate
        ]
        for decision_id in selected:
            self.audit_logger.log(
                entity_type="DECISION",
                entity_id=decision_id,
                action="RANDOM_AUDIT_SELECTED",
                details={
                    "audit_rate": self.weight_calculator.random_audit_rate,
                    "simulated_flag": "SIMULATED",
                },
                timestamp=start,
            )
        return selected

    def _attribute_reward(self, entry: _LabelEntry, matured_at: datetime) -> TrainingExample:
        event_id, action, amount = self._lookup_decision(entry.decision_id)
        reward = self.reward_calculator.compute_reward(action, entry.label, amount)
        weight = self.weight_calculator.compute_weight(action, entry.is_random_audit)
        session = self.db.get_session()
        try:
            existing = session.execute(
                select(RewardRecord).where(RewardRecord.decision_id == entry.decision_id)
            ).scalar_one_or_none()
            if existing is None:
                session.add(
                    RewardRecord(
                        decision_id=entry.decision_id,
                        event_id=event_id,
                        action=action,
                        true_label=entry.label,
                        reward_amount=reward,
                        cost_amount=-reward,
                    )
                )
                session.commit()
        finally:
            session.close()
        self.audit_logger.log(
            entity_type="DECISION",
            entity_id=entry.decision_id,
            action="LABEL_MATURED_AND_REWARDED",
            details={
                "event_id": event_id,
                "label": entry.label,
                "source": entry.source.value,
                "reward_amount": reward,
                "importance_weight": weight,
                "simulated_flag": "SIMULATED",
            },
            timestamp=matured_at,
        )
        return TrainingExample(
            decision_id=entry.decision_id,
            event_id=event_id,
            label=entry.label,
            action=action,
            amount=amount,
            source=entry.source,
            matured_at=matured_at,
            importance_weight=weight,
            reward_amount=reward,
        )

    def _lookup_decision(self, decision_id: str) -> tuple[str, str, float]:
        session = self.db.get_session()
        try:
            row = session.execute(
                select(DecisionRecord, PaymentEventRecord)
                .join(PaymentEventRecord, DecisionRecord.event_id == PaymentEventRecord.event_id)
                .where(DecisionRecord.decision_id == decision_id)
            ).one_or_none()
            if row is None:
                raise ValueError(f"Unknown decision_id: {decision_id}")
            decision, payment = row
            return decision.event_id, decision.action, float(payment.amount)
        finally:
            session.close()

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
