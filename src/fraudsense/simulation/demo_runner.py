"""Repeatable, explicitly simulated end-to-end scenarios for the analyst demo.

The scenarios exercise the real feature, policy, routing, challenge, storage,
audit, feedback, and reward components.  Their calibrated probabilities are
fixed synthetic fixtures because this prototype does not ship a model checkpoint.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from ..audit.log import (
    AuditLogger,
    CaseRecord,
    DatabaseManager,
    DecisionRecord,
    FeatureSnapshotRecord,
    InteractionRecord,
    PaymentEventRecord,
    ScoreRecord,
)
from ..challenge.handler import ChallengeHandler
from ..decision.router import DecisionRouter
from ..features.service import FeatureService
from ..feedback.label_service import LabelService, LabelSource
from ..graph.builder import DynamicGraphBuilder
from ..ingest.schema import TransactionEvent


@dataclass(frozen=True)
class DemoScenario:
    """One named synthetic public-data-like transaction fixture."""

    name: str
    description: str
    amount: float
    true_label: int
    probabilities: tuple[float, float, float]
    conformal_set: frozenset[str]
    is_novel: bool = False
    timeout_challenge: bool = False


SCENARIOS: dict[str, DemoScenario] = {
    "Clear legitimate purchase": DemoScenario(
        "Clear legitimate purchase",
        "Low-value established-device payment; expected to pass through.",
        1200.0,
        0,
        (0.96, 0.03, 0.01),
        frozenset({"ALLOW"}),
    ),
    "Customer-confirmed challenge": DemoScenario(
        "Customer-confirmed challenge",
        "Ambiguous mid-value payment that receives a simulated confirmation.",
        18000.0,
        0,
        (0.20, 0.65, 0.15),
        frozenset({"CHALLENGE"}),
    ),
    "High-value mule risk": DemoScenario(
        "High-value mule risk",
        "High-value fraud fixture; policy and conformal evidence should open a case.",
        125000.0,
        1,
        (0.03, 0.05, 0.92),
        frozenset({"FRAUD"}),
        is_novel=True,
    ),
    "Challenge timeout containment": DemoScenario(
        "Challenge timeout containment",
        "High-value ambiguous event. The simulated timeout must escalate to a hold.",
        75000.0,
        1,
        (0.20, 0.55, 0.25),
        frozenset({"CHALLENGE"}),
        timeout_challenge=True,
    ),
}


class _FixtureRiskHead:
    """Small deterministic adapter that feeds a scenario's labelled test probability."""

    def __init__(self, probabilities: tuple[float, float, float]):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict_multi_proba(self, features: np.ndarray) -> np.ndarray:
        return np.repeat(self.probabilities[None, :], len(features), axis=0)


class _FixtureConformal:
    """Returns the scenario's transparent, fixed conformal set."""

    def __init__(self, prediction_set: frozenset[str]):
        self.prediction_set = set(prediction_set)

    def predict_set(self, probabilities: np.ndarray) -> set[str]:
        return set(self.prediction_set)


class DemoScenarioRunner:
    """Runs scenario fixtures end to end and persists only SIMULATED records."""

    def __init__(self, db_manager: DatabaseManager | None = None):
        self.db = db_manager or DatabaseManager()
        self.audit = AuditLogger(self.db, model_version="demo-fixture-v1")

    def run(self, scenario_name: str) -> dict[str, Any]:
        """Runs a selected fixture and returns stage results for the dashboard."""
        scenario = SCENARIOS[scenario_name]
        run_id = uuid.uuid4().hex[:8]
        event_id = f"SIM_EVT_{run_id}"
        decision_id = f"SIM_DEC_{run_id}"
        started_at = datetime.now(UTC)
        event = TransactionEvent(
            event_id=event_id,
            ts=started_at,
            amount=scenario.amount,
            payer_id=f"SIM_PAYER_{scenario_name[:3].upper()}",
            payee_id="SIM_MERCHANT_001",
            channel="UPI",
            device_id="SIM_DEVICE_001",
            label=-1,
            meta={"scenario": scenario.name, "simulated_flag": "SIMULATED"},
        )
        stages: list[dict[str, Any]] = []

        t0 = time.perf_counter()
        features = FeatureService().extract_features(event, update_state=True)
        stages.append(self._stage("Causal feature service", "PASS", t0, f"{len(features)} features"))

        t0 = time.perf_counter()
        graph = DynamicGraphBuilder()
        graph.add_event(event)
        graph_data = graph.build_pyg_data()
        stages.append(self._stage("Dynamic graph", "PASS", t0, f"{graph_data.num_nodes} nodes / {graph_data.num_edges} edges"))

        t0 = time.perf_counter()
        router = DecisionRouter(
            risk_head=_FixtureRiskHead(scenario.probabilities),
            conformal_calibrator=_FixtureConformal(scenario.conformal_set),
        )
        decision = router.route(event, features, can_open_case=True, is_novel=scenario.is_novel)
        stages.append(self._stage("Risk, conformal & policy router", "PASS", t0, decision.action))

        challenge_status = None
        resolved_action = decision.action
        challenge_reason = "Not required"
        if decision.action == "CHALLENGE":
            t0 = time.perf_counter()
            handler = ChallengeHandler()
            elapsed = handler.timeout_min if scenario.timeout_challenge else 0.0
            challenge_status, resolved_action, challenge_reason = handler.simulate_response(
                scenario.true_label, elapsed_minutes=elapsed
            )
            stages.append(self._stage("Simulated customer challenge", "PASS", t0, challenge_status.value))

        t0 = time.perf_counter()
        case_id = self._persist(event, decision_id, decision, features, resolved_action, challenge_status)
        stages.append(self._stage("SQLite records & audit log", "PASS", t0, "Case opened" if case_id else "Decision logged"))

        t0 = time.perf_counter()
        feedback = LabelService(self.db)
        feedback.ingest_label(decision_id, scenario.true_label, LabelSource.CASE_CLOSURE, started_at)
        examples = feedback.mature_labels(started_at + timedelta(days=7))
        reward = examples[0].reward_amount if examples else 0.0
        stages.append(self._stage("Delayed feedback & reward", "PASS", t0, f"₹{reward:,.2f}"))

        return {
            "scenario": scenario.name,
            "description": scenario.description,
            "simulated_flag": "SIMULATED",
            "event_id": event_id,
            "decision_id": decision_id,
            "case_id": case_id,
            "true_label": "FRAUD" if scenario.true_label else "LEGITIMATE",
            "fixture_probabilities": {"ALLOW": scenario.probabilities[0], "CHALLENGE": scenario.probabilities[1], "FRAUD": scenario.probabilities[2]},
            "conformal_set": sorted(decision.conformal_set),
            "policy_clauses": decision.policy_clauses,
            "initial_action": decision.action,
            "final_action": resolved_action,
            "challenge_status": challenge_status.value if challenge_status else "NOT_REQUIRED",
            "challenge_reason": challenge_reason,
            "router_latency_ms": round(decision.latency_ms, 3),
            "stages": stages,
        }

    @staticmethod
    def _stage(module: str, status: str, started: float, outcome: str) -> dict[str, Any]:
        return {"module": module, "status": status, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "outcome": outcome}

    def _persist(self, event, decision_id, decision, features, action, challenge_status):
        case_id = f"SIM_CASE_{decision_id[-8:]}" if action == "HOLD_FOR_REVIEW" else None
        session = self.db.get_session()
        try:
            session.add(PaymentEventRecord(event_id=event.event_id, ts=event.ts, amount=event.amount, payer_id=event.payer_id, payee_id=event.payee_id, channel=event.channel, device_id=event.device_id, label=None, meta_json=json.dumps(event.meta)))
            session.add(FeatureSnapshotRecord(event_id=event.event_id, features_json=json.dumps(features.tolist())))
            session.add(ScoreRecord(event_id=event.event_id, p_fraud=decision.p_fraud, conformal_set_json=json.dumps(sorted(decision.conformal_set)), model_version="demo-fixture-v1", latency_ms=decision.latency_ms))
            session.add(DecisionRecord(decision_id=decision_id, event_id=event.event_id, action=action, reason=decision.reason, policy_clauses_json=json.dumps(decision.policy_clauses), is_fail_safe=False, was_overridden=decision.was_overridden))
            if challenge_status:
                session.add(InteractionRecord(interaction_id=f"SIM_INT_{decision_id[-8:]}", decision_id=decision_id, event_id=event.event_id, channel_type="IN_APP", template_id="TMPL_CHALLENGE_UPI_01", status=challenge_status.value, simulated_flag="SIMULATED"))
            if case_id:
                session.add(CaseRecord(case_id=case_id, event_id=event.event_id, decision_id=decision_id, state="OPENED", amount=event.amount, requires_supervisor=event.amount >= 50000, supervisor_approved=False))
            session.commit()
        finally:
            session.close()
        self.audit.log("DECISION", decision_id, action, {"event_id": event.event_id, "scenario": event.meta["scenario"], "simulated_flag": "SIMULATED"})
        return case_id
