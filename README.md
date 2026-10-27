# FraudSense Prototype

> ⚠️ **DISCLAIMER: RESEARCH CAPSTONE PROTOTYPE ONLY**
> **SIMULATED DATA AND POLICY RETRIEVAL. NEVER PRESENT AS DEPLOYABLE.**
> This system is built for research and demonstration of relational fraud detection and conversational compliance in Indian digital payment systems. It runs strictly on public/simulated datasets with synthetic regulatory text. Every external action, regulatory report, customer prompt, and analyst decision is simulated and labelled **`SIMULATED`**.

---

## 1. Overview & Constraints

FraudSense combines:
1. **Relational Graph Learning (GAT)**: Dynamic temporal transaction graph modeling account connectivity and mule rings.
2. **Split Conformal Prediction**: Guaranteed finite-sample prediction coverage sets ($C(x) \in 2^{\{\text{ALLOW}, \text{CHALLENGE}, \text{FRAUD}\}}$) with calibrated uncertainty.
3. **Reinforcement Learning (DQN)**: Action policy optimizing operational cost payoffs ($\gamma = 0$ contextual formulation).
4. **Fast Path / Slow Path Separation**: Sub-200ms transaction scoring with zero LLM dependencies; asynchronous conversational enrichment (CRAG RAG, guarded Text2SQL) for case management.
5. **Regulatory Alignment**: Deterministic policy masks and citations grounded strictly in `data/policy/clauses.json` (RBI Master Direction on Fraud Risk Management).
6. **Zero Silent Payment Releases**: Challenge timeouts strictly escalate to `HOLD_FOR_REVIEW`.

---

## 2. Repository Structure

```
fraudsense/
  config/params.yaml          # All proposed thresholds, reward weights, timers
  data/
    raw/                      # PaySim, IEEE-CIS, European Credit Card CSVs
    processed/                # Processed caches
    policy/clauses.json       # Paraphrased RBI Master Direction clauses
  src/fraudsense/
    ingest/                   # Dataset loaders, schema normalisation, chronological replay
    graph/                    # PyG dynamic graph builder with recency decay
    features/                 # Account feature store (velocities, ratios)
    models/                   # GAT, calibrated risk head, split conformal, DQN, baselines
    policy/                   # Regulatory action mask rules and high-value override
    decision/router.py        # Decision arbitrator & sub-200ms fail-safe fallback
    scoring/api.py            # FastAPI fast path
    cases/                    # Case record lifecycle and strict state machine
    challenge/handler.py      # Simulated challenge handler with timeout escalation
    feedback/                 # Label maturity window and importance weighting
    enrichment/               # Guarded Text2SQL, CRAG RAG, customer templates, CPFIR/FMR drafts
    audit/log.py              # Append-only SQLite audit log
    monitoring/               # PSI, hold rate, latency, and conformal coverage
    simulation/               # Simulated customers, analysts, and virtual replay clock
  app/analyst_ui.py           # Streamlit analyst workstation
  scripts/                    # run_replay.py, train.py, evaluate.py, red_team.py
  tests/                      # pytest test suite
  docs/
    DECISIONS.md              # Every assumption and deviation with rationale
    results.md                # Comparative benchmark metrics
  requirements.txt            # Pinned dependencies
  README.md
```

---

## 3. Normalized Schema

All datasets (PaySim, IEEE-CIS, European) are normalized into the canonical schema:
$$\langle \text{event\_id}, \text{ts}, \text{amount}, \text{payer\_id}, \text{payee\_id}, \text{channel}, \text{device\_id}, \text{label} \rangle$$

- **PaySim**: `step` mapped to hourly timestamps; `nameOrig` / `nameDest` accounts become graph nodes; transactions become edges.
- **IEEE-CIS**: Joined transaction and identity tables; graph constructed from cards, addresses, emails, and devices.
- **European Credit Card**: Normalized for tabular risk head and DQN baseline sanity checks.

---

## 4. Quickstart & Testing

```bash
# 1. Activate Python 3.11 virtual environment
source .venv/bin/activate

# 2. Run unit and integration tests
pytest tests/ -v

# 3. Check code formatting
ruff check src tests
```
