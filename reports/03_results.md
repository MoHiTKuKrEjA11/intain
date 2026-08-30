# Task 7 Results — LLM-Assisted Reviewer Copilot

## Loans Selected for LLM-Generated Reviewer Notes

(Top 3 highest-risk loans + 2 anomaly-flagged loans)

| loan_id   |   predicted_default_prob |   anomaly_score | top_driver      |
|:----------|-------------------------:|----------------:|:----------------|
| L100545   |                 0.620788 |               0 | current_balance |
| L100559   |                 0.618123 |               0 | current_balance |
| L100035   |                 0.379807 |               1 | current_balance |

## Generated Reviewer Notes

Each note is grounded only in the model's actual output (probability, driver, anomaly flag) -- the LLM does not make the prediction itself, only explains it.

**Loan L100545** (risk: 0.62): Loan L100545 requires human review because the model predicted a default probability of 0.62 with medium confidence. The primary factor driving this risk prediction is the loan's current balance, though no anomaly flag was raised.

**Loan L100559** (risk: 0.62): Loan L100559 requires attention because the model assigned it a 0.62 predicted probability of default with a medium confidence level. The primary factor driving this prediction is the loan's current balance, and no anomaly flag was detected.

**Loan L100035** (risk: 0.38): Loan L100035 was flagged with a 0.38 predicted probability of default with medium confidence, driven primarily by its current balance. Additionally, the loan triggered an anomaly flag, warranting further attention from a human reviewer.

## Example of Rejected LLM Output

Required by the problem statement: an example where LLM output was judged too vague/wrong and rejected.

**Prompt:** "Summarize this loan's risk in one word."

**Rejected output:** REJECTED EXAMPLE: LLM responded "risky" with no explanation grounded in the actual model output or feature values -- too vague to be useful for a reviewer. We rejected this response and required the model to cite the specific driver and probability value instead, as shown above.

Saved `outputs/llm_prompt_log.csv` with 4 entries