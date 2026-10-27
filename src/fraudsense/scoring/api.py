"""
FastAPI Fast-Path Scoring Service for FraudSense.
Provides sub-200ms synchronous transaction risk scoring and routing.
Guaranteed zero LLM invocations on the fast path.
"""

import time
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..audit.log import (
    AuditLogger,
    CaseRecord,
    DatabaseManager,
    DecisionRecord,
    InteractionRecord,
    PaymentEventRecord,
    ScoreRecord,
)
from ..config import load_config
from ..decision.router import DecisionResult, DecisionRouter
from ..features.service import FeatureService
from ..ingest.schema import TransactionEvent


class TransactionScoreRequest(BaseModel):
    """Transaction scoring payload."""

    event_id: str = Field(..., description="Unique transaction ID")
    amount: float = Field(..., ge=0, description="Amount in INR")
    payer_id: str = Field(..., description="Payer/cardholder identifier")
    payee_id: str = Field(..., description="Payee/merchant identifier")
    channel: str = Field(default="TRANSFER", description="Transaction channel")
    device_id: str = Field(default="DEV_DEFAULT", description="Device identifier")
    ts: datetime | None = Field(default=None, description="Event timestamp (UTC)")
    meta: dict[str, Any] | None = Field(default_factory=dict, description="Dataset metadata")


class TransactionScoreResponse(BaseModel):
    """Scoring decision response."""

    event_id: str
    decision_id: str
    action: str
    p_fraud: float
    conformal_set: list[str]
    policy_clauses: list[str]
    is_fail_safe: bool
    was_overridden: bool
    latency_ms: float
    case_id: str | None = None
    reason: str
    simulated_flag: str = "SIMULATED"


def create_app(
    decision_router: DecisionRouter | None = None,
    feature_service: FeatureService | None = None,
    db_manager: DatabaseManager | None = None,
) -> FastAPI:
    """Application factory for FraudSense scoring service."""
    app = FastAPI(
        title="FraudSense Fast-Path Scoring API",
        description="Sub-200ms synchronous transaction scoring service. SIMULATED DATA ONLY.",
        version="0.1.0",
    )

    cfg = load_config()
    db = db_manager or DatabaseManager()
    audit_logger = AuditLogger(db_manager=db)
    feat_service = feature_service or FeatureService(cfg)
    router = decision_router or DecisionRouter(config=cfg)

    latencies: list[float] = []

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "prototype_mode": "SIMULATED",
            "fast_path_llm_calls": 0,
        }

    @app.get("/metrics")
    def get_metrics():
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
        return {
            "total_scored": len(latencies),
            "p50_latency_ms": round(p50, 3),
            "p99_latency_ms": round(p99, 3),
            "latency_budget_ms": router.timeout_ms,
        }

    @app.post("/score", response_model=TransactionScoreResponse)
    def score_transaction(req: TransactionScoreRequest):
        t_start = time.perf_counter()
        now_utc = req.ts or datetime.now(UTC)
        decision_id = f"DEC_{uuid.uuid4().hex[:12]}"
        case_id = None

        # Build TransactionEvent
        event = TransactionEvent(
            event_id=req.event_id,
            ts=now_utc,
            amount=req.amount,
            payer_id=req.payer_id,
            payee_id=req.payee_id,
            channel=req.channel,
            device_id=req.device_id,
            label=-1,  # Masked for scoring
            meta=req.meta or {},
        )

        # 1. Extract streaming features (causal, no future leakage)
        features = feat_service.extract_features(event, update_state=True)

        # 2. Route transaction through fast path (zero LLM calls)
        res: DecisionResult = router.route(
            event=event,
            tabular_features=features,
            gat_embedding=None,
            can_open_case=True,
        )

        # 3. Persist to SQLite
        session = db.get_session()
        try:
            # Payment event
            session.merge(
                PaymentEventRecord(
                    event_id=req.event_id,
                    ts=now_utc,
                    amount=req.amount,
                    payer_id=req.payer_id,
                    payee_id=req.payee_id,
                    channel=req.channel,
                    device_id=req.device_id,
                    label=None,
                    meta_json=__import__("json").dumps(req.meta or {}),
                )
            )

            # Score record
            session.add(
                ScoreRecord(
                    event_id=req.event_id,
                    p_fraud=res.p_fraud,
                    conformal_set_json=__import__("json").dumps(list(res.conformal_set)),
                    model_version="fraudsense_v1.0",
                    latency_ms=res.latency_ms,
                )
            )

            # Decision record
            session.add(
                DecisionRecord(
                    decision_id=decision_id,
                    event_id=req.event_id,
                    action=res.action,
                    reason=res.reason,
                    policy_clauses_json=__import__("json").dumps(res.policy_clauses),
                    is_fail_safe=res.is_fail_safe,
                    was_overridden=res.was_overridden,
                )
            )

            # State escalation triggers
            if res.action == "CHALLENGE":
                # Create simulated customer interaction record
                session.add(
                    InteractionRecord(
                        interaction_id=f"INT_{uuid.uuid4().hex[:10]}",
                        decision_id=decision_id,
                        event_id=req.event_id,
                        channel_type="SMS",
                        template_id="TMPL_CHALLENGE_UPI_01",
                        status="SENT",
                        simulated_flag="SIMULATED",
                    )
                )

            elif res.action == "HOLD_FOR_REVIEW":
                # Open formal case record
                case_id = f"CASE_{uuid.uuid4().hex[:10]}"
                session.add(
                    CaseRecord(
                        case_id=case_id,
                        event_id=req.event_id,
                        decision_id=decision_id,
                        state="OPENED",
                        amount=req.amount,
                        requires_supervisor=req.amount >= router.high_val_thresh,
                        supervisor_approved=False,
                    )
                )

            session.commit()
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"Database commit error: {e!s}") from e
        finally:
            session.close()

        # 4. Audit Log
        audit_logger.log(
            entity_type="DECISION",
            entity_id=decision_id,
            action=res.action,
            details={
                "event_id": req.event_id,
                "amount": req.amount,
                "p_fraud": res.p_fraud,
                "conformal_set": list(res.conformal_set),
                "is_fail_safe": res.is_fail_safe,
                "was_overridden": res.was_overridden,
                "case_id": case_id,
            },
        )

        total_latency = (time.perf_counter() - t_start) * 1000.0
        latencies.append(total_latency)

        return TransactionScoreResponse(
            event_id=req.event_id,
            decision_id=decision_id,
            action=res.action,
            p_fraud=res.p_fraud,
            conformal_set=sorted(res.conformal_set),
            policy_clauses=res.policy_clauses,
            is_fail_safe=res.is_fail_safe,
            was_overridden=res.was_overridden,
            latency_ms=round(total_latency, 3),
            case_id=case_id,
            reason=res.reason,
            simulated_flag="SIMULATED",
        )

    return app


# Default app instance
app = create_app()
