# FraudSense Model Evaluation Benchmarks

> **NOTICE**: Research capstone benchmark comparison. All evaluations are performed under identical time-ordered splits, operational payoff parameters ($\alpha=1.0, \lambda=0.2, \gamma=0.9, u=5, c_a=150, h=300, \rho=0.01$), and simulated financial conditions.

---

## 1. Summary of Model Progression

| Model | Graph Features | Conformal Sets | Policy Mechanism | PR-AUC | Recall @ 0.5% FPR | Recall @ 1.0% FPR | Cost per 10,000 (INR) | Primary Operating Action Distribution |
|---|---|---|---|---|---|---|---|---|
| **Baseline A** | None | No | GBDT + Tuned Thresholds | **1.000** | **1.000** | **1.000** | -₹5,687,833.57 | ALLOW: 99.62%, HOLD: 0.38% |
| **Baseline B** | None | No | Bayes-Optimal Cost-Sensitive | — | — | — | -₹5,558,269.92 | ALLOW: 66.82%, CHALLENGE: 32.59%, HOLD: 0.59% |
| **Risk Head Alone** | Planned (Ph 3) | Planned (Ph 3) | Calibrated Temperature | *Phase 3* | *Phase 3* | *Phase 3* | *Phase 3* | *Phase 3* |
| **Full Pipeline (GAT + Conformal + DQN)** | Planned (Ph 3/4)| Planned (Ph 3) | DQN Contextual Policy + Mask | *Phase 4* | *Phase 4* | *Phase 4* | *Phase 4* | *Phase 4* |

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
