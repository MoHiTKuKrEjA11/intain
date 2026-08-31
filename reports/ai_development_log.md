# AI Development Log

## AI Tools Used

- **Claude (Anthropic)** — used across three main areas of this project:
  1. Generating the synthetic/practice train and test loan datasets (since the official organizer data pack wasn't released during prototype development), including designing realistic column distributions and a proper train/test split with hidden target columns to simulate the real evaluation setup.
  2. Building the Streamlit UI structure (`app.py`) — the layout, tabs (Predictions, Explainability, Anomalies, AI Reviewer Note), sidebar controls, and scenario sliders.
  3. Drafting and refining prompts used for AI-assisted evaluation and report writing — including the prompts sent to Gemini for reviewer notes, and structuring the model card, data/explainability/scenario reports, and this development log itself.
- **Google Gemini API (gemini-3.6-flash)** — used specifically for Task 7, generating plain-English reviewer notes from the model's actual output (free tier).

## Representative Prompts Used

1. "Help me build a data profiling pipeline for loan-level CSV data — check missing values and outliers."
2. "Write a time-aware train/validation split instead of a random split, and explain why it matters."
3. "You are a loan reviewer assistant. A model flagged loan {loan_id} with a predicted default probability of {prob} (confidence: {confidence}). The top model driver was '{driver}'. Anomaly flag: {yes/no}. Write a 2-sentence plain-English note for a human reviewer explaining why this loan needs attention, based only on the numbers given. Do not invent facts not provided." (this is the actual production prompt used in code/03_llm_reviewer_copilot.py)
4. "Build a Streamlit app with tabs for predictions, explainability, anomalies, and an AI reviewer note generator, with scenario sliders in the sidebar."

## Accepted vs Rejected AI Output

- **Accepted:** Data cleaning code (fixing negative balances, filling missing interest rates) — verified correct by checking `.describe()` output before/after (min balance went from negative to 0).
- **Accepted:** Time-aware split logic — verified train/validation row counts and date cutoff manually (13,875 train / 3,014 validation rows, cutoff 2024-05-01).
- **Accepted:** Real Gemini-generated reviewer notes — example: for Loan L100545 (risk 0.62), the model produced: "Loan L100545 requires human review because the model predicted a default probability of 0.62 with medium confidence. The primary factor driving this risk prediction is the loan's current balance, though no anomaly flag was raised." — verified this correctly reflects the actual model output (0.62 probability, current_balance as top driver, no anomaly flag).
- **Rejected:** An early test prompt ("Summarize this loan's risk in one word") produced a vague one-word answer like "risky" with no grounding in the actual numbers — rejected as unusable for a real reviewer, and the prompt was rewritten to explicitly require citing the specific driver and probability value (see outputs/llm_prompt_log.csv for the full rejected example).
- **Rejected:** An early version of the synthetic test-data generation script accidentally removed a current-state feature column (`prepayment_flag`) along with the actual future-target columns, causing a KeyError. Caught by testing the pipeline end-to-end against the generated files, then corrected to only strip genuine future-outcome columns.

## Human Review Process

- Every AI-generated code block was run and its output manually checked (row counts, missing-value counts, sample outputs) before being trusted.
- Model metrics (ROC-AUC 0.599, classification report) were reviewed by the team, not accepted blindly.
- All real LLM-generated reviewer notes were spot-checked against the actual model output values (probability, driver, anomaly flag) for factual grounding before being accepted.
- The Streamlit app was manually run and clicked through (all four tabs, both sliders, CSV upload) to confirm the UI actually worked before deployment, not just that the code compiled.
- A stress test with intentionally broken/extreme data (negative values, unknown categories, broken dates, duplicates) was run using `code/04_stress_test.py`, producing a PASSED result documented in `reports/04_stress_test_results.md`, confirming the pipeline doesn't crash on messy real-world data.
- A bias/fairness check (`code/05_bias_fairness_analysis.py`) was manually reviewed to confirm no concerning gaps existed across state or occupancy type before including it as a positive result.

## Approximate AI-Generated Code Share

- Roughly **40-45%** of the code and content in this project was AI-assisted (drafted with Claude), concentrated mainly in three areas: generating the synthetic train/test loan datasets, building the Streamlit UI structure, and drafting/refining prompts used for AI-assisted evaluation and report writing. The core modeling logic, decisions on which target to predict, which validation approach to use, which features to include, and interpretation of results were driven and verified by the team.

## Lessons Learned

- Time-aware splitting is easy to get wrong if not explicitly handled — random splits looked fine at first glance but would have leaked future data.
- LLM output needs explicit grounding instructions ("only use the numbers given") or it produces vague, unhelpful text — confirmed directly through the rejected one-word-answer example.
- A proper train/test simulation (separate files, with the test file's future-outcome columns removed) caught issues that a single-file, date-split-only test did not — specifically, a feature/target column confusion bug in an early version of the test-file simulation script.
- Synthetic/practice data is useful for building and testing the pipeline early, but real data is needed to judge actual model quality.
- Using AI to draft the report structure (model card, explainability report) was helpful for consistency, but every number in those reports still needed to be pulled from actual script output — an early draft had stale numbers from an earlier run, which was caught and corrected before submission.
