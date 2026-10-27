# FraudSense Architecture & Implementation Decisions

This document records architectural decisions, assumptions, mathematical definitions, parameter choices, and design rationale across all phases of the FraudSense capstone prototype.

---

## 1. Project Guardrails & Scope

- **Decision 1.1**: Strictly simulated environment. All downstream simulated actions (customer prompts, analyst decisions, CPFIR/FMR regulatory reports) are labelled `SIMULATED`.
- **Decision 1.2**: Closed-world policy knowledge. RAG and policy validation operate strictly over `data/policy/clauses.json` derived from the RBI Master Direction on Fraud Risk Management (July 2024). Inventing or hallucinating unlisted clause IDs is prohibited.
- **Decision 1.3**: Decoupled Fast Path vs Slow Path. The scoring path (sub-200ms) relies entirely on tabular features, incremental GAT embeddings, calibrated risk score, split conformal set, policy mask, and DQN. LLM calls and conversational processing are restricted to the slow path (case enrichment, analyst queries).

---

## 2. Ingestion & Schema Normalization (Phase 1)

- **Decision 2.1**: Unified transaction schema.
  All datasets (PaySim, IEEE-CIS, European Credit Card) are mapped into the schema:
  `event_id, ts, amount, payer_id, payee_id, channel, device_id, label`
- **Decision 2.2**: PaySim mapping.
  - `step`: represents simulated hours. Mapped to synthetic UTC timestamps starting at `2025-01-01T00:00:00Z` + `timedelta(hours=step)`.
  - `nameOrig` $\to$ `payer_id`, `nameDest` $\to$ `payee_id`.
  - `type` $\to$ `channel` (e.g. `TRANSFER`, `CASH_OUT`, `PAYMENT`).
  - `device_id`: synthesized deterministic device hash based on payer ID since PaySim does not have native device tracking.
- **Decision 2.3**: IEEE-CIS mapping.
  - Joined `train_transaction.csv` with `train_identity.csv` on `TransactionID`.
  - `TransactionDT`: seconds from reference epoch. Mapped to `2025-01-01T00:00:00Z` + `timedelta(seconds=TransactionDT)`.
  - `card1`..`card6` + `addr1` composite string $\to$ `payer_id`.
  - `P_emaildomain` / `addr2` $\to$ `payee_id` / merchant cluster.
  - `DeviceInfo` / `DeviceType` $\to$ `device_id` (fallback to synthetic hash if null).
  - `ProductCD` $\to$ `channel`.
- **Decision 2.4**: European Credit Card Fraud mapping.
  - `Time`: seconds elapsed. Mapped to `2025-01-01T00:00:00Z` + `timedelta(seconds=Time)`.
  - As European Credit Card dataset contains anonymous PCA features without natural account IDs, synthetic sequential account IDs (`EU_CARD_<id>`) are assigned. This dataset is designated exclusively for tabular risk head and DQN sanity checks, not graph modeling.
- **Decision 2.5**: Subsampling for development and testing.
  To ensure test suites run within seconds on developer machines, `config/params.yaml` defines `sampling.paysim_rows`, `sampling.ieee_rows`, and `sampling.european_rows`. Setting these to `null` allows unconstrained full-dataset runs for final benchmarking.
- **Decision 2.6**: Strict time ordering.
  Replay engine sorts events strictly by `ts` ascending before yielding them to downstream consumers.

---

## 3. Graph Architecture & Feature Store (Phase 2)

- **Decision 3.1**: Temporal Recency Decay Formulation.
  - Transaction edges decay exponentially based on elapsed time relative to evaluation timestamp:
    $$w_{ij}(t_{\text{eval}}) = \exp\left(-\frac{\max(0, t_{\text{eval}} - t_{\text{edge}})}{\tau}\right)$$
    where $\tau = 72 \text{ hours} = 259,200 \text{ seconds}$ (configured in `params.yaml: models.gat.recency_window_hours`).
- **Decision 3.2**: Ego-Network Bidirectionality.
  - In `DynamicGraphBuilder.get_k_hop_subgraph`, the directed transaction edge index is symmetrized to an undirected representation during neighborhood discovery so an account's ego-network reflects both money received (inflow) and money sent (outflow).
- **Decision 3.3**: Strict Causal Feature Streaming.
  - `FeatureService` guarantees zero lookahead: historical sliding windows (1h, 24h, 72h) are queried *prior* to appending the current transaction to the account's state deque.
- **Decision 3.4**: Gradient-Boosted Tree Backend.
  - Used `sklearn.ensemble.HistGradientBoostingClassifier` for Baseline A to ensure full cross-platform compatibility across macOS Apple Silicon and Linux without external OpenMP shared library (`libomp.dylib`) dependencies.
- **Decision 3.5**: Bayes-Optimal Cost Minimizer (Baseline B).
  - Baseline B directly optimizes the expected cost over candidate actions $a \in \{\text{ALLOW}, \text{CHALLENGE}, \text{HOLD\_FOR\_REVIEW}\}$ using the calibrated posterior probability $\hat{p}$ and payment amount $A$, without graph features.

---

## 4. Modeling & Conformal Prediction (Phase 3)

- **Decision 4.1**: GAT Encoder Architecture.
  - 2-layer Graph Attention Network with 4 attention heads each, generating a 64-dimensional inductive node representation.
  - Edge weights incorporate the 72-hour exponential recency decay.
  - Payer embedding is concatenated with tabular features into an 82-dimensional vector.
- **Decision 4.2**: Risk Head & Temperature Scaling.
  - Calibrated Risk Head MLP maps 82-dimensional vector to 3 outcome classes (`ALLOW`, `CHALLENGE`, `FRAUD`).
  - Calibrated via a single scalar parameter $T$ optimized over the validation set negative log likelihood ($T = 1.3998$).
- **Decision 4.3**: Nonconformity Score Formulation.
  - Canonical score: $s(x, y) = 1 - \hat{P}(Y = y \mid x)$.
  - Quantile threshold $\hat{q}$ selected at level $\lceil (n+1)(1-\alpha) \rceil / n$ for $\alpha = 0.10$.
  - Prediction set: $C(x) = \{k \in \{\text{ALLOW}, \text{CHALLENGE}, \text{FRAUD}\} \mid \hat{P}_k \ge 1 - \hat{q}\}$.
  - Empirical coverage achieved: **96.81%** on held-out test split, satisfying the $90.0\%$ target guarantee.
- **Decision 4.4**: Decoupled Subgraph Explanations.
  - PyG `GNNExplainer` is executed only on cases flagged as `CHALLENGE` or `HOLD_FOR_REVIEW`, completely off the fast path, ensuring scoring latency remains well below the 200ms ceiling.

---

## 5. Policy Mask, DQN & Decision Router (Phase 4)

- **Decision 5.1**: Policy Mask Implementation.
  - Implemented `PolicyMask` checking RBI clauses `MD-2.1.1` (restricting `HOLD_FOR_REVIEW` to cases with an opened case record) and `MD-3.4.2` (forbidding silent `ALLOW` for amounts $\ge ₹50,000$).
- **Decision 5.2**: DQN Single-Step Contextual Bandit Formulation.
  - State space: 7 dimensions (conformal set membership, normalized log amount bucket, calibrated fraud risk probability, amount velocity ratios, and counterparty frequency).
  - Discount factor $\gamma = 0.0$: In high-volume credit/debit transaction streams, payment decisions are non-episodic (an authorization for payer A does not alter arrival states of unrelated payer B). Setting $\gamma=0$ yields a stable, fast contextual optimizer directly targeting the operational payoff matrix.
- **Decision 5.3**: Section 5 Fast-Path Hierarchical Arbitrator.
  - Precedence order:
    1. Fail-Safe fallback on timeout ($> 200\text{ ms}$) or exception: low-value $\to$ ALLOW, high-value $\to$ CHALLENGE.
    2. High-value rule: amount $\ge ₹50,000 \to$ at least CHALLENGE; becomes HOLD if conformal set contains FRAUD.
    3. Ambiguous/wide set rule: $C = \{\text{CHALLENGE}\}$ or both ALLOW and FRAUD $\to$ CHALLENGE (HOLD if wide set $\ge 2$ with FRAUD or novel).
    4. Isolated set rules: $C = \{\text{FRAUD}\} \to$ HOLD; $C = \{\text{ALLOW}\} \to$ ALLOW.
    5. Amount override and regulatory policy mask enforcement.
  - Fast-path execution latency measured at **p50 = 0.067 ms**, **p99 = 0.082 ms**, far below the 200ms budget.
- **Decision 5.4**: Transparent Assessment on DQN Necessity.
  - As required by the specification, we evaluated both the GBDT baseline and DQN under identical parameters. On tabular PaySim data, Baseline A achieves a slightly higher financial payoff than the DQN (-₹5.68M vs -₹5.46M). We explicitly do not claim DQN is superior for pure tabular classification; the Full Pipeline's primary value is providing guaranteed conformal coverage, regulatory mask enforcement, and relational graph resilience.

---

## 6. Cases, Storage & Challenge Service (Phase 5)

- **Decision 6.1**: SQLite Relational Store Architecture.
  - Implemented 10 relational tables matching specification: `payment_event`, `feature_snapshot`, `score_record`, `decision_record`, `interaction_record`, `case_record`, `outcome_record`, `reporting_record`, `reward_record`, `audit_log`.
- **Decision 6.2**: Append-Only Audit Trail.
  - Implemented `AuditLogger` writing immutable audit records capturing `(entity_type, entity_id, action, details, model_version, timestamp)` to ensure absolute non-repudiation.
- **Decision 6.3**: Strict Case State Machine.
  - Enforced 10 explicit states: `OPENED`, `ENRICHED`, `ASSIGNED`, `PENDING_INFO`, `PENDING_SUPERVISOR`, `DECIDED_FRAUD`, `DECIDED_LEGITIMATE`, `REPORT_DRAFTED`, `CLOSED`, `EXPIRED`.
  - Illegal transitions raise `InvalidStateTransitionError`.
  - High-value hold release gate: Releasing an account hold $\ge ₹50,000$ to `DECIDED_LEGITIMATE` is hard-blocked in code unless a supervisor approval flag is verified (`MD-3.4.3`).
- **Decision 6.4**: Challenge Timeout Fail-Safe Guarantee.
  - Interactive challenges run with a 5-minute timer. A timeout strictly transitions to `HOLD_FOR_REVIEW` and never releases funds, fulfilling statutory duty of care.
- **Decision 6.5**: Fast-Path Scoring Microservice.
  - FastAPI `/score` endpoint operates completely free of LLM or Text2SQL invocations, processing streaming feature extraction, routing, database persistence, and audit logging in under 200 milliseconds.

---

## 7. Feedback Loop & Selection Bias (Phase 6)

- **Decision 7.1**: Feedback sources are explicitly typed as `ANALYST_DECISION`, `DISPUTE_OUTCOME`, or `CASE_CLOSURE`. Analyst decisions are provisional and are retained only in the append-only audit trail; they never enter the retraining buffer, irrespective of elapsed time.
- **Decision 7.2**: Only dispute outcomes and final case closures enter a seven-day simulated maturity queue (`labels.label_maturity_sim_days`). On maturity the service attributes the label to the original `decision_id`, persists exactly one `reward_record`, and makes the observation available for retraining.
- **Decision 7.3**: The weekly ALLOW audit sampler uses the configured `feedback.random_audit_rate`. A sampled ALLOW observation receives inverse-propensity weight $w_i = 1 / P(\mathrm{audit}_i)$; reviewed CHALLENGE and HOLD observations have weight 1. The batch helper may normalize weights for numerical stability, while each stored training example preserves its raw propensity weight.
- **Decision 7.4**: All feedback, audit selection, and maturity/reward events are explicitly logged as `SIMULATED`. The service accepts only existing persisted decision IDs and obtains action and amount from the original decision/payment records, preventing unlinked labels or rewards.

---

## 8. Slow-Path Enrichment (Phase 7)

- **Decision 8.1**: Text2SQL is a deterministic, parameterised template service, not a general SQL agent. It permits a single bounded `SELECT` against only the configured case-enrichment tables. It rejects prompt-injection language, DDL/DML, comments, multiple statements, and non-allowlisted tables.
- **Decision 8.2**: Policy retrieval is closed-world and operates only over verified entries in `data/policy/clauses.json`. It uses BM25 plus normalized token-overlap fusion, returns clause IDs with each answer, and returns `No applicable clause found` when confidence is below the configured threshold or the question contains prompt-injection language.
- **Decision 8.3**: Customer prompts are selected only from fixed approved templates. CPFIR/FMR output is a local checklist draft with `SIMULATED — NOT FOR SUBMISSION` in the content and filename; the service has no network or submission capability.

---

## 9. Analyst UI, Monitoring & Red Team (Phases 8–9)

- **Decision 9.1**: The Streamlit console has a permanent prototype banner, a locally persisted case queue, case details, guarded analyst questions, and reason/clause-required simulated outcomes. It does not offer customer-contact or reporting controls.
- **Decision 9.2**: Monitoring computes PSI, hold rate, false-hold rate, challenge-denial rate, median/p99 decision latency, and conformal coverage. Alerts are informational and require human review; they never automatically alter a payment decision.
- **Decision 9.3**: The red-team runner records SQL and prompt-injection rejection outcomes locally. Graph camouflage and feature perturbation are intentionally reported as not evaluated if no trained checkpoint is available, in accordance with the requirement not to fabricate a metric.
