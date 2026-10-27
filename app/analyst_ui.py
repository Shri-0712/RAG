"""Streamlit console for the FraudSense simulated analyst workflow.

Run with: ``streamlit run app/analyst_ui.py``.  This interface is local only
and never sends customer messages or files reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fraudsense.audit.log import AuditLogger, DatabaseManager, OutcomeRecord
from fraudsense.enrichment.rag import PolicyRAG
from fraudsense.enrichment.text2sql import Text2SQLService, UnsafeQueryError

st.set_page_config(page_title="FraudSense — SIMULATED", layout="wide")
st.error("PROTOTYPE. SIMULATED DATA. NO REAL CUSTOMERS OR REPORTS.")
st.title("FraudSense Analyst Console")


@st.cache_resource
def services():
    db = DatabaseManager()
    return db, Text2SQLService(db), PolicyRAG()


db, sql_service, rag = services()
session = db.get_session()
try:
    cases = (
        session.execute(
            text(
                "SELECT case_id, state, amount, assigned_analyst, requires_supervisor, created_at FROM case_record ORDER BY created_at DESC"
            )
        )
        .mappings()
        .all()
    )
finally:
    session.close()

if not cases:
    st.info(
        "No simulated cases yet. Score a transaction that routes to HOLD_FOR_REVIEW, then refresh."
    )
    st.stop()

queue = [dict(row) for row in cases]
states = ["All"] + sorted({row["state"] for row in queue if row["state"]})
selected_state = st.sidebar.selectbox("Case state", states)
minimum_amount = st.sidebar.number_input("Minimum amount (₹)", min_value=0.0, value=0.0)
filtered = [row for row in queue if row["amount"] >= minimum_amount]
if selected_state != "All":
    filtered = [row for row in filtered if row["state"] == selected_state]
st.subheader("Open simulated case queue")
st.json(filtered)

case_id = st.selectbox(
    "Inspect case", [row["case_id"] for row in filtered] or [row["case_id"] for row in queue]
)
session = db.get_session()
try:
    case = (
        session.execute(
            text(
                "SELECT c.*, p.payer_id, p.payee_id, p.device_id, p.channel, d.action, d.reason, d.policy_clauses_json FROM case_record c JOIN payment_event p ON c.event_id=p.event_id JOIN decision_record d ON c.decision_id=d.decision_id WHERE c.case_id=:case_id"
            ),
            {"case_id": case_id},
        )
        .mappings()
        .one()
    )
finally:
    session.close()
st.subheader(f"Case {case_id}")
left, right = st.columns(2)
left.json({key: value for key, value in dict(case).items() if key not in {"policy_clauses_json"}})
right.write("Policy clauses", json.loads(case["policy_clauses_json"]))

question = st.text_input("Ask an approved case question", placeholder="Show payment history")
if question:
    try:
        result = sql_service.answer_question(question, case_id)
        st.caption(f"Source SQL ({result.elapsed_ms:.2f} ms): {result.sql}")
        st.json(result.rows)
    except (UnsafeQueryError, ValueError) as exc:
        st.warning(str(exc))

policy_question = st.text_input(
    "Find applicable policy clause", placeholder="What applies to real-time containment?"
)
if policy_question:
    st.write(rag.answer(policy_question))

st.subheader("Simulated analyst decision")
with st.form("decision"):
    outcome = st.selectbox(
        "Suggested outcome", ["CONFIRM_FRAUD", "ATTEMPTED_FRAUD", "LEGITIMATE", "NEED_INFO"]
    )
    reason = st.text_input("Reason code (required)")
    clause = st.text_input("At least one known clause ID (required)")
    submitted = st.form_submit_button("Record SIMULATED decision")
if submitted:
    if not reason or not clause:
        st.warning("A reason code and at least one clause ID are required.")
    elif clause not in {c["id"] for c in rag.clauses}:
        st.warning("Clause ID is not in the supplied policy corpus.")
    else:
        session = db.get_session()
        try:
            session.add(
                OutcomeRecord(
                    case_id=case_id,
                    decision_id=case["decision_id"],
                    analyst_decision=outcome,
                    reason_code=reason,
                    clause_ids_json=json.dumps([clause]),
                    simulated_flag="SIMULATED",
                )
            )
            session.commit()
        finally:
            session.close()
        AuditLogger(db).log(
            "CASE",
            case_id,
            "ANALYST_DECISION",
            {
                "decision_id": case["decision_id"],
                "analyst_decision": outcome,
                "reason_code": reason,
                "clause_ids": [clause],
                "simulated_flag": "SIMULATED",
            },
        )
        st.success("SIMULATED decision captured locally. No external action was taken.")
