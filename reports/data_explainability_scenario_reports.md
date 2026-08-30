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
- A separate stress test (with intentionally extreme injected errors — huge balances, broken dates, unseen categories, negative loan ages) confirmed the pipeline handles messy real-world data without crashing.

## Anomaly Detection Summary

- 507 records (~3.0%) flagged as anomalous by an Isolation Forest model trained on balance, interest rate, days past due, and loan age.
- Sample anomalies included loans with $0 balances still carrying 60-120 days past due.

## Train vs Test Drift

- [Fill in once the official organizer test file is available — compare feature distributions between train and test to check for drift]

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

---

# Scenario Report — Loan Performance Intelligence Engine (Prototype)

## Scenarios Modeled

| Scenario        | Assumption                                | Avg. Predicted Default Probability |
| --------------- | ----------------------------------------- | ---------------------------------- |
| Base            | Data as-is                                | 0.4123                             |
| Adverse-credit  | Days-past-due +50%, interest rate +1.5pts | 0.3780                             |
| High-prepayment | Interest rate -1.0pt                      | 0.3982                             |

## Interpretation

- [Fill in with your team's read of these numbers once run on real data — e.g. does the adverse scenario meaningfully raise predicted risk? Does the direction make sense?]
- Note: on this practice dataset, the scenario differences are modest, likely because the underlying data is synthetic/random. Expect clearer, more interpretable shifts once run on real hackathon data with genuine economic relationships.

## Segment-Level Impact

- [Add once your team runs the scenario breakdown by vintage, credit band, state, or servicer — currently only an aggregate/portfolio-level number is shown above]
