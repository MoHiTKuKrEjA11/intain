# Model Card — Loan Performance Intelligence Engine (Prototype)

## Objective

Predict whether a loan will default within the next 12 months, detect anomalous/suspicious loan records, and generate reviewer-facing explanations.

## Data

- Source: practice/synthetic data, split into a train file (with answers) and a test file (answers hidden), simulating the organizer's real train/test setup
- Train file: 16,889 rows / 1,600 loans
- Test file: 4,246 rows / 400 loans (never seen during training)
- Target variable: `next_12m_default_flag`

## Features Used

19 features: loan_age_months, remaining_term_months, original_balance, current_balance, interest_rate, days_past_due, modification_flag, prepayment_flag, plus encoded categorical fields (credit_score_band, ltv_band, dti_band, state, loan_purpose, occupancy_type, property_type, servicer_name, current_status, source_system, document_status)

## Model Type

Random Forest Classifier (scikit-learn), 200 trees, max depth 8, class-weight balanced (to handle rare default cases)

## Validation Method

**Time-aware split** — NOT random. Trained on the earliest ~80% of months, validated on the most recent ~20% of months within the train file, to avoid leaking future information into training. Final predictions are made on a completely separate test file containing different loans never seen during training.

## Metrics (latest run, on sim_train/sim_test)

- ROC-AUC: 0.599
- Note: This score is modest because the underlying data is synthetic/random by design. Expected to improve meaningfully once trained on real hackathon data with genuine patterns.

## Leakage Controls

- Time-based split prevents the same loan's future records from appearing in training.
- Target/future outcome columns excluded from input features.
- Encoders (for categorical fields) are fit only on the training file, then safely applied to the test file, handling any category the test file has that the training file never saw.

## Known Limitations

- Trained on synthetic practice data — results should not be interpreted as real loan risk.
- Class imbalance: defaults are rare in this data, which limits precision on the minority class.
- Categorical encoding is basic (label encoding) — could be improved with more principled encoding.

## Failure Modes

- Model currently shows low precision on the "default" class (many false positives) — flagged for future tuning (threshold adjustment, more features, more real data).
