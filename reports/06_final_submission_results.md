# Final Submission Build — Full Required Format

## Training Models

- Trained model for `next_3m_delinquency_flag` -> `predicted_delinquency_prob`
- Trained model for `next_12m_default_flag` -> `predicted_default_prob`
- Trained model for `next_12m_prepayment_flag` -> `predicted_prepayment_prob`

## Transition Matrix Built (for `next_state` predictions)

- States covered: ['current', 'delinquent_30_60', 'delinquent_90_plus']

## Scoring Test File: `data/sim_test.csv`

## Final submission.csv Columns

`['loan_id', 'predicted_delinquency_prob', 'predicted_default_prob', 'predicted_prepayment_prob', 'next_state', 'anomaly_score', 'exception_type', 'exception_required', 'top_drivers', 'action', 'confidence']`

Saved **outputs/submission.csv** with 4246 rows.

### Sample rows

| loan_id   |   predicted_delinquency_prob |   predicted_default_prob |   predicted_prepayment_prob | next_state   |   anomaly_score | exception_type   |   exception_required | top_drivers                    | action   | confidence   |
|:----------|-----------------------------:|-------------------------:|----------------------------:|:-------------|----------------:|:-----------------|---------------------:|:-------------------------------|:---------|:-------------|
| L100001   |                     0.321745 |                 0.248122 |                    0.135    | current      |               0 | none             |                    0 | interest_rate, current_balance | monitor  | low          |
| L100001   |                     0.333839 |                 0.271131 |                    0.131605 | current      |               0 | none             |                    0 | interest_rate, current_balance | monitor  | low          |
| L100001   |                     0.331872 |                 0.298528 |                    0.117807 | current      |               0 | none             |                    0 | interest_rate, current_balance | monitor  | low          |
| L100001   |                     0.325319 |                 0.287411 |                    0.110962 | current      |               0 | none             |                    0 | interest_rate, current_balance | monitor  | low          |
| L100001   |                     0.334235 |                 0.292683 |                    0.102807 | current      |               0 | none             |                    0 | interest_rate, current_balance | monitor  | low          |