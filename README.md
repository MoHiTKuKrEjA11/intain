# Loan Performance Intelligence Engine — Prototype

Prototype submission for Intain Campus FinTech Challenge 2026 (AI Track).

**Live Demo:** https://loan-intelligence-engine.streamlit.app/

## What this project does

Given loan-level monthly performance data, this project:

1. Profiles and cleans the data (finds missing values, invalid records)
2. Trains machine learning models to predict loan delinquency, default, and prepayment risk
3. Builds a monthly state-transition model (does a loan move to "default", "current", etc. next month)
4. Detects anomalous/suspicious loan records
5. Runs "what-if" scenario simulations (adverse credit conditions, high prepayment)
6. Explains what drives each prediction
7. Uses an LLM (Google Gemini, free tier) to turn model output into plain-English reviewer notes — with full prompt logging
8. Includes a live Streamlit demo app (`app.py`) for interactive exploration
9. Produces a final `submission.csv` matching the format described in the problem statement (probabilities, next state, exception type, anomaly score, top drivers, action, confidence)

## Folder structure

```
data/
  sim_train.csv                               -> Training data, has answer/target columns
  sim_test.csv                                -> Test data, target columns hidden (same loans excluded from train)
code/
  01_data_profiling_and_prediction_model.py   -> Tasks 1, 2, 4, 5, 6 + generates the FINAL outputs/submission.csv directly
  02_transition_model.py                      -> Task 3 (monthly transition model, standalone deep-dive report)
  03_llm_reviewer_copilot.py                  -> Task 7 (LLM reviewer notes)
  04_stress_test.py                           -> Advanced feature: synthetic-data stress testing (proves the pipeline survives extreme/broken data)
  05_bias_fairness_analysis.py                -> Advanced feature: bias/fairness analysis (checks predicted risk gaps across state, occupancy type, credit band)
reports/
  01_results.md                               -> Auto-generated: full output of script 01 (profiling, 3 model metrics, feature importance, anomalies, scenarios, final submission summary)
  02_results.md                               -> Auto-generated: full output of script 02 (transition matrix, baseline comparison)
  03_results.md                               -> Auto-generated: full output of script 03 (LLM-generated reviewer notes, rejected-output example)
  04_stress_test_results.md                   -> Auto-generated: full output of script 04 (stress test against deliberately broken data)
  05_bias_fairness_results.md                 -> Auto-generated: full output of script 05 (predicted risk by state, occupancy type, credit band)
  model_card.md                               -> Model objective, data, features, metrics, limitations, known failure modes
  data_explainability_scenario_reports.md     -> Data quality findings, explainability (top drivers, error analysis), and scenario simulation results
  ai_development_log.md                       -> How AI tools were used during development: prompts used, accepted/rejected AI output, human review process, lessons learned
outputs/                                      -> Generated automatically when you run the scripts
  submission.csv                              -> FINAL output, produced entirely by script 01 in the required format
  transition_matrix.csv                       -> Produced by script 02
  llm_prompt_log.csv                          -> Produced by script 03
app.py                                        -> Live interactive Streamlit demo (predictions, explainability, anomalies, AI reviewer notes)
requirements.txt                              -> Python packages needed
.env.example                                  -> Template for your API key (copy to .env)
```

## How to run this (from a terminal)

**1. Install the required packages:**

```
pip install -r requirements.txt
```

**2. Set up your API key (only needed for script 03 and the AI Reviewer tab in app.py):**
Copy `.env.example` to a new file named `.env`, and paste your free Gemini API key into it:

```
cp .env.example .env
```

Then open `.env` and replace `your_key_here` with your real key (get one free at aistudio.google.com).

**3. Run each script (any order after 01, since 02-05 don't depend on each other):**

```
python3 code/01_data_profiling_and_prediction_model.py
python3 code/02_transition_model.py
python3 code/03_llm_reviewer_copilot.py
python3 code/04_stress_test.py
python3 code/05_bias_fairness_analysis.py
```

Script 01 alone produces the complete, final `outputs/submission.csv` — scripts 02, 04, and 05 are standalone deep-dive analyses with their own reports, and don't modify `submission.csv`. Each script prints its results directly in the terminal, and also writes the same results into a matching report file in `reports/`.

**4. Run the live demo app (optional):**

```
streamlit run app.py
```

## Advanced Features (beyond minimum requirements)

- **Synthetic-data stress testing** (`code/04_stress_test.py`) — deliberately injects extreme/broken data (impossible values, unseen categories, corrupted dates, duplicates) into the pipeline to prove it handles messy real-world conditions, not just clean practice data.
- **Bias/fairness analysis** (`code/05_bias_fairness_analysis.py`) — checks whether predicted default risk varies meaningfully across state, occupancy type, and credit score band, flagging any large, unexplained gaps for human review.

## What's in each report file

- **`01_results.md` / `02_results.md` / `03_results.md` / `04_stress_test_results.md` / `05_bias_fairness_results.md`** — raw, auto-generated output from each script run: exact numbers, tables, and metrics produced. Regenerated every time the scripts are run.
- **`model_card.md`** — the model's objective, the data and features it uses, its type and validation method, current metrics, known limitations, and failure modes.
- **`data_explainability_scenario_reports.md`** — three write-ups in one file: data quality findings (missing values, outliers, anomalies), explainability (top prediction drivers, a local example, error analysis, bias/fairness check), and scenario simulation results with interpretation and segment-level breakdown.
- **`ai_development_log.md`** — which AI tools were used and how, representative prompts, examples of accepted vs. rejected AI output, the human review process, and lessons learned during development.

## Final submission.csv format

Produced entirely by script 01, matching the fields described in the problem statement's `submission_template.csv`: `loan_id, predicted_delinquency_prob, predicted_default_prob, predicted_prepayment_prob, next_state, anomaly_score, exception_type, exception_required, top_drivers, action, confidence`.

## Data

`data/sim_train.csv` and `data/sim_test.csv` are a practice train/test split (built from synthetic data) simulating the real organizer file structure — the test file's future-outcome columns are removed, and it contains loans never seen during training, matching how the official `loan_monthly_performance_train.csv` / `loan_monthly_performance_test.csv` will work. Once the official files are released, only the `TRAIN_FILE` / `TEST_FILE` paths at the top of each script need to change — no other code changes required.