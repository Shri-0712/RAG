# FraudSense Model Evaluation Benchmarks

> **NOTICE**: Research capstone benchmark comparison. All evaluations are performed under identical time-ordered splits, operational payoff parameters ($\alpha=1.0, \lambda=0.2, \gamma=0.9, u=5, c_a=150, h=300, \rho=0.01$), and simulated financial conditions.

---

## 1. Summary of Model Progression

| Model | Graph Features | Conformal Sets | Policy Mechanism | PR-AUC | Cost per 10,000 (INR) | Total Test Reward (INR) | Primary Operating Action Distribution | Fast-Path Latency (p99) |
|---|---|---|---|---|---|---|---|---|
| **Baseline A (GBDT Tabular)** | None | No | GBDT + Tuned Thresholds | **1.000** | **-₹5,687,833.57** | **+₹5,687,833.57** | ALLOW: 99.62%, HOLD: 0.38% | < 0.1 ms |
| **Baseline B (Cost-Sensitive)** | None | No | Bayes-Optimal Cost Minimizer | — | -₹5,558,269.92 | +₹5,558,269.92 | ALLOW: 66.82%, CHAL: 32.59%, HOLD: 0.59% | < 0.1 ms |
| **Risk Head Alone (GAT + MLP)** | 64-dim GAT | No | Calibrated Temperature ($T=1.40$) | 0.403 | -₹5,574,204.84 | +₹5,574,204.84 | CHAL: 96.42%, ALLOW: 3.12%, HOLD: 0.46% | < 0.1 ms |
| **Full Pipeline (GAT + Conformal + DQN + Mask)** | 64-dim GAT | **Yes (Coverage 96.8%)** | Policy Mask + DQN + Override | 0.403 | -₹5,463,307.63 | +₹5,463,307.63 | CHAL: 97.06%, ALLOW: 1.94%, HOLD: 1.00% | **0.082 ms** |

*(Negative cost indicates positive net financial payoff / prevented fraud losses under Section 6 parameters).*

> **Critical Empirical Finding on DQN Necessity**:
> Under the PaySim tabular distribution, **Baseline A (GBDT with tuned thresholds) achieves the lowest operational cost (-₹5,687,833.57)**, outperforming both the Risk Head alone and the DQN-driven Full Pipeline.
> We do **not** claim that the DQN is superior for static tabular fraud classification. The primary role of the Full Pipeline is not raw unconstrained profit maximization, but rather:
> 1. Providing **distribution-free finite-sample coverage guarantees** via split conformal prediction.
> 2. Enforcing statutory compliance constraints (RBI Master Direction natural justice gates, case opening prerequisites, and supervisor sign-offs for high-value releases).
> 3. Sub-0.1ms fast-path arbitration without LLM latency.

*(Negative cost indicates positive net financial payoff / prevented fraud losses).*

---

## 2. Phase 2 Baseline Analysis

### Baseline A: Tabular Gradient-Boosted Trees (LightGBM/HistGBDT)
- **Feature Set**: 18 causal streaming account features (amounts, velocities in 1h/24h/72h windows, payee diversity, channel indicators, balance transition error).
- **Threshold Tuning**: Tuned via grid search on held-out validation split ($\theta_{\text{challenge}} = 0.50, \theta_{\text{hold}} = 0.55$).
- **Performance on Test Split**:
  - Test set size: 10,000 transactions (6 actual frauds).
  - PR-AUC: 1.000
  - Recall at 0.5% False-Hold Rate: 100.0%
  - Recall at 1.0% False-Hold Rate: 100.0%
  - Action Distribution: ALLOW 99.62%, HOLD 0.38%, CHALLENGE 0.00%.
  - Total Reward on Test Split: +₹5,687,833.57 (Net prevented fraud loss).

### Baseline B: Bayes-Optimal Cost-Sensitive Decision Model
- **Mechanism**: Calculates expected cost across candidate actions $a \in \{\text{ALLOW}, \text{CHALLENGE}, \text{HOLD\_FOR\_REVIEW}\}$ directly using predicted fraud probability $\hat{p}$ and transaction amount $A$:
  $$\mathbb{E}[\text{Cost}(\text{ALLOW})] = \hat{p} \cdot \alpha \cdot A$$
  $$\mathbb{E}[\text{Cost}(\text{CHALLENGE})] = \hat{p} \cdot \lambda \cdot A + (1 - \hat{p}) \cdot u$$
  $$\mathbb{E}[\text{Cost}(\text{HOLD})] = \hat{p} \cdot (c_a - \gamma \cdot A) + (1 - \hat{p}) \cdot (h + \rho \cdot A)$$
- **Performance on Test Split**:
  - Cost per 10,000 payments: -₹5,558,269.92
  - Action Distribution: ALLOW 66.82%, CHALLENGE 32.59%, HOLD 0.59%.
- **Key Observation**:
  Baseline B proactively utilizes the low-friction **`CHALLENGE`** action ($u = ₹5$) for borderline risk payments, creating a softer customer experience while protecting against false holds.

---

## 3. Threshold Sensitivity Table (Hold Rate vs. Missed Fraud Rate)

*Extracted from `reports/threshold_sensitivity.csv` on the test split:*

| Threshold ($\theta$) | Hold Rate (%) | Missed Fraud Rate (%) |
|---|---|---|
| **0.05** | 0.46% | 0.0% |
| **0.10** | 0.42% | 0.0% |
| **0.20** | 0.42% | 0.0% |
| **0.30** | 0.41% | 0.0% |
| **0.40** | 0.39% | 0.0% |
| **0.50** | 0.38% | 0.0% |
| **0.55** | 0.38% | 0.0% |
| **0.60** | 0.38% | 0.0% |
| **0.70** | 0.38% | 0.0% |
| **0.80** | 0.35% | 0.0% |
| **0.90** | 0.28% | 0.0% |
| **0.95** | 0.19% | 16.7% |

At the tuned threshold $\theta = 0.55$, the model achieves zero missed frauds while maintaining a strict hold rate of 0.38%, well within the regulatory false-hold ceiling of 1.0%.

---

## 4. Phase 3: GAT Encoder, Calibrated Risk Head & Conformal Coverage

- **GAT Encoder**: 2 layers, 4 heads, output dimension 64. Encodes 78,499 account nodes and 50,000 temporal transaction edges.
- **Risk Head Calibration**:
  - Input: Concatenation of 64-dim GAT representation and 18-dim tabular streaming features (82 dimensions).
  - Learned Temperature: $T = 1.3998$ (calibrated via L-BFGS on validation negative log-likelihood).
- **Split Conformal Prediction Metrics (Target Coverage = 90.0%)**:
  - **Empirical Coverage on Test Split**: **96.81%** (Exceeds nominal 90% finite-sample target; coverage gap = +6.81%).
  - **Quantile Threshold $\hat{q}$**: $0.4433$
  - **Mean Set Size**: $1.00$
  - **Singleton Rate**: $100.0\%$
  - **Cost per 10,000 Payments**: -₹5,189,532.59 (Total net prevented fraud payoff = +₹5.19M per 10k payments).
  - **Action Distribution under Conformal Set Rules**:
    - `ALLOW`: 44.25%
    - `CHALLENGE`: 53.84% (soft verification protects customer trust)
    - `HOLD_FOR_REVIEW`: 1.91% (strict fraud containment)

