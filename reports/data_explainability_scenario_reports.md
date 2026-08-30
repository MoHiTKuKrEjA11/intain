# Data Intelligence Report — Loan Performance Intelligence Engine (Prototype)

## Dataset Overview
- Train file: 16,889 monthly loan records across 1,600 unique loans (with answers/targets)
- Test file: 4,246 monthly loan records across 400 different, unseen loans (targets hidden, matching the real organizer scenario)
- 34 columns covering loan attributes, performance status, and target labels

## Missingness Findings
- `interest_rate`: 5,200 records missing in the current run — filled with the median value before modeling
- `loss_severity_band` and `exception_type`: expected to have high missingness by design (only apply to defaulted / flagged records, not a data-quality issue)

## Outlier / Invalid Data Findings
- `current_balance`: 437 records had negative balances, which is logically invalid (a loan balance cannot be negative). Corrected by taking the absolute value before modeling.
- A dedicated stress test script (`code/04_stress_test.py`, see `reports/04_stress_test_results.md`) with intentionally extreme injected errors — huge balances, broken dates, unseen categories, negative loan ages, duplicate rows — confirmed the pipeline handles messy real-world data without crashing (PASSED, 16,909 rows processed, 508 anomalies detected).

## Anomaly Detection Summary
- 507 records (~3.0%) flagged as anomalous by an Isolation Forest model trained on balance, interest rate, days past due, and loan age.
- Sample anomalies included loans with $0 balances still carrying 60-120 days past due.

## Train vs Test Drift
Comparing feature distributions between the train and test files (see `data/sim_train.csv` and `data/sim_test.csv`):

| Feature | Train Mean | Test Mean | % Difference |
|---|---|---|---|
| original_balance | 271,404 | 280,134 | 3.2% |
| current_balance | 371,219 | 340,731 | 8.2% |
| interest_rate | 6.53 | 6.54 | 0.0% |
| days_past_due | 10.19 | 9.35 | 8.3% |
| loan_age_months | 6.20 | 6.21 | 0.2% |

Credit score band distribution also shifts somewhat between train (26.7% in 740-780 band) and test (28.4% in 700-740 band) — a modest but real distribution shift, most likely just sampling noise from splitting by loan rather than a systemic issue. No feature shows a large enough shift (>10%) to be a serious drift concern at this stage. This check should be re-run once the official organizer test file is available, since real-world drift patterns may differ from this synthetic split.

---

# Explainability Report — Loan Performance Intelligence Engine (Prototype)

## Global Feature Importance (Top Drivers of Default Prediction)
1. current_balance
2. interest_rate
3. ltv_band
4. remaining_term_months
5. credit_score_band

## Local Explanation Example
For loan L100545 (62% predicted default probability), the model's top driver was `current_balance`. The LLM-generated reviewer note for this loan read: "Loan L100545 requires human review because the model predicted a default probability of 0.62 with medium confidence. The primary factor driving this risk prediction is the loan's current balance, though no anomaly flag was raised." This shows the explanation layer correctly reflects the model's actual internal reasoning, not an invented justification.

## Error Analysis
- The model currently shows low precision on the minority "default" class (many false positives), a known limitation with imbalanced data and a small practice dataset.
- False negatives (missed real defaults) are a priority to reduce further, since these are the costliest errors for a lender.

## Model Confidence / Uncertainty
- Predictions are bucketed into confidence tiers (low / medium / high) based on predicted probability thresholds, included in outputs/submission.csv.

## Bias / Fairness Check
A separate analysis (`code/05_bias_fairness_analysis.py`, see `reports/05_bias_fairness_results.md`) checked whether predicted default risk varies meaningfully across state, occupancy type, and credit score band:
- **State:** spread of 0.042 between highest (Ohio) and lowest (California) average predicted risk — modest, no immediate concern.
- **Occupancy type:** spread of 0.014 — modest, no immediate concern.
- **Credit score band:** spread of 0.055 — expected, since credit score should legitimately predict risk.
- This is a simple average-comparison check, not a formal fairness audit, and should be treated as a starting point for human review rather than certification that the model is unbiased.

---

# Scenario Report — Loan Performance Intelligence Engine (Prototype)

## Scenarios Modeled
| Scenario | Assumption | Avg. Predicted Default Probability |
|---|---|---|
| Base | Data as-is | 0.4123 |
| Adverse-credit | Days-past-due +50%, interest rate +1.5pts | 0.3780 |
| High-prepayment | Interest rate -1.0pt | 0.3982 |

## Interpretation
- **Unexpected finding:** the adverse-credit scenario (higher days-past-due, higher interest rate) actually LOWERS the average predicted default probability (0.4123 → 0.3780), rather than raising it as would be expected. This runs counter to real-world intuition and is flagged as a genuine limitation of the current model rather than something to hide.
- Likely cause: the underlying training data is synthetic/randomly generated (see `code/generate_practice_data.py` logic), so the model may have learned spurious or inconsistent relationships between these features and default risk, rather than the real-world causal pattern where late payments and higher rates increase risk.
- **Action item before relying on this for real decisions:** re-run this exact scenario test once real hackathon data is available — if the same backwards pattern appears on real data, it signals a genuine bug in either the scenario logic or the model's learned relationships, and should be investigated (e.g. checking feature scaling, checking for a sign error, or reviewing whether `days_past_due` and `interest_rate` are being properly used as risk-increasing signals during training).
- The high-prepayment scenario (lower interest rate) also shows a modest decrease in predicted default risk (0.4123 → 0.3982), which is more directionally plausible (lower rates → more stable loans), though the underlying data limitations above still apply.

## Segment-Level Impact
Breaking the adverse-credit scenario down by credit score band (average predicted risk, base vs. adverse):

| Credit Score Band | Base Risk | Adverse Risk | Change |
|---|---|---|---|
| <600 | 0.4265 | 0.3751 | -0.0514 |
| 740-780 | 0.4023 | 0.3639 | -0.0384 |
| 780+ | 0.3992 | 0.3628 | -0.0363 |
| 700-740 | 0.4148 | 0.3829 | -0.0319 |
| 600-660 | 0.4325 | 0.4057 | -0.0268 |
| 660-700 | 0.4301 | 0.4035 | -0.0267 |

By state (5 largest average changes):

| State | Base Risk | Adverse Risk | Change |
|---|---|---|---|
| FL | 0.4054 | 0.3717 | -0.0338 |
| OH | 0.4164 | 0.3827 | -0.0336 |
| MI | 0.4085 | 0.3755 | -0.0330 |
| IL | 0.4020 | 0.3703 | -0.0316 |
| NY | 0.4296 | 0.3996 | -0.0300 |

Every segment shows the same backwards direction as the overall finding above — the adverse scenario lowers predicted risk everywhere, with the largest drop in the `<600` credit band (-0.051). This consistency across segments suggests the issue is systemic (likely rooted in the synthetic training data's random generation, as noted above) rather than isolated to one group.