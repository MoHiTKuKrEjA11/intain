# Loan Performance Intelligence Engine — Prototype

Prototype submission for Intain Campus FinTech Challenge 2026 (AI Track).

## What this project does

Given loan-level monthly performance data, this project:

1. Profiles and cleans the data (finds missing values, invalid records)
2. Trains a machine learning model to predict loan default risk
3. Builds a monthly state-transition model (does a loan move to "default", "current", etc. next month)
4. Detects anomalous/suspicious loan records
5. Runs "what-if" scenario simulations (adverse credit conditions, high prepayment)
6. Explains what drives each prediction
7. Uses an LLM (Google Gemini, free tier) to turn model output into plain-English reviewer notes — with full prompt logging
8. Includes a live Streamlit demo app (`app.py`) for interactive exploration

## Folder structure

```
data/
  sim_train.csv                               -> Training data, has answer/target columns
  sim_test.csv                                -> Test data, target columns hidden (same loans excluded from train)
code/
  01_data_profiling_and_prediction_model.py   -> Tasks 1, 2, 4, 5, 6 + generates submission.csv
  02_transition_model.py                      -> Task 3 (monthly transition model)
  03_llm_reviewer_copilot.py                  -> Task 7 (LLM reviewer notes)
reports/
  01_results.md                               -> Auto-generated: full output of script 01 (profiling, model metrics, feature importance, anomalies, scenarios)
  02_results.md                               -> Auto-generated: full output of script 02 (transition matrix, baseline comparison)
  03_results.md                               -> Auto-generated: full output of script 03 (LLM-generated reviewer notes, rejected-output example)
  model_card.md                               -> Model objective, data, features, metrics, limitations, known failure modes
  data_explainability_scenario_reports.md     -> Data quality findings, explainability (top drivers, error analysis), and scenario simulation results
  ai_development_log.md                       -> How AI tools were used during development: prompts used, accepted/rejected AI output, human review process, lessons learned
outputs/                                      -> Generated automatically when you run the scripts
  submission.csv
  transition_matrix.csv
  llm_prompt_log.csv
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

**3. Run each script in order:**

```
python3 code/01_data_profiling_and_prediction_model.py
python3 code/02_transition_model.py
python3 code/03_llm_reviewer_copilot.py
```

Each script prints its results directly in the terminal, **and also writes the same results into a matching report file** in `reports/` (`01_results.md`, `02_results.md`, `03_results.md`) — so results can be reviewed later without re-running anything.

**4. Run the live demo app (optional):**

```
streamlit run app.py
```

## What's in each report file

- **`01_results.md` / `02_results.md` / `03_results.md`** — raw, auto-generated output from each script run: exact numbers, tables, and metrics produced. Regenerated every time the scripts are run.
- **`model_card.md`** — the model's objective, the data and features it uses, its type and validation method, current metrics, known limitations, and failure modes.
- **`data_explainability_scenario_reports.md`** — three write-ups in one file: data quality findings (missing values, outliers, anomalies), explainability (top prediction drivers, a local example, error analysis), and scenario simulation results with interpretation.
- **`ai_development_log.md`** — which AI tools were used and how, representative prompts, examples of accepted vs. rejected AI output, the human review process, and lessons learned during development.

## Data

`data/sim_train.csv` and `data/sim_test.csv` are a practice train/test split (built from synthetic data) simulating the real organizer file structure — the test file's future-outcome columns are removed, and it contains loans never seen during training, matching how the official `loan_monthly_performance_train.csv` / `loan_monthly_performance_test.csv` will work. Once the official files are released, only the `TRAIN_FILE` / `TEST_FILE` paths at the top of `code/01_data_profiling_and_prediction_model.py` need to change — no other code changes required.
