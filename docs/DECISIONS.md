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

## 4. Modeling & Decision Logic (Phases 3–4 Preview)

- **Decision 4.1**: Split Conformal Prediction formulation.
  - Nonconformity score: $s(x, y) = 1 - \hat{P}(Y = y \mid x)$.
  - Calibration quantile $\hat{q}$ computed at coverage target $1 - \alpha = 0.90$.
  - Prediction set: $C(x) = \{y \in \mathcal{Y} \mid 1 - \hat{P}(Y = y \mid x) \le \hat{q}\}$.
- **Decision 4.2**: Single-step DQN formulation.
  - In credit transaction streaming, a single payment authorization does not dynamically alter global arrival rates or payment system environment transitions.
  - Therefore, discount factor $\gamma = 0$, transforming the DQN policy into a contextual bandit directly optimizing the operational cost payoff matrix specified in Section 6 of the project brief.
- **Decision 4.3**: Challenge Timeout Fail-Safe.
  - In compliance with regulatory duty of care, challenge timeouts strictly transition to `HOLD_FOR_REVIEW`, never to payment release.
