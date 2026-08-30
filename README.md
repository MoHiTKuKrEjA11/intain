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

## Folder structure
```
data/                                         -> Put loan_monthly_performance_train.csv here
code/
  01_data_profiling_and_prediction_model.py   -> Tasks 1, 2, 4, 5, 6 + generates submission.csv
  02_transition_model.py                      -> Task 3 (monthly transition model)
  03_llm_reviewer_copilot.py                  -> Task 7 (LLM reviewer notes)
reports/
  model_card.md                               -> Model objective, data, metrics, limitations
  ai_development_log.md                       -> Task 8 (how AI tools were used to build this)
  data_explainability_scenario_reports.md     -> Data quality, explainability, and scenario write-ups
  demo_video_script.md                        -> What to say/show in the 5-minute demo video
outputs/                                      -> Generated automatically when you run the scripts
  submission.csv
  transition_matrix.csv
  llm_prompt_log.csv
requirements.txt                              -> Python packages needed
.env.example                                  -> Template for your API key (copy to .env)
```

## How to run this (from a terminal)

**1. Install the required packages:**
```
pip install -r requirements.txt
```

**2. Add your loan data:**
Place `practice_loan_data.csv` (or the official `loan_monthly_performance_train.csv` once released) inside the `data/` folder.

**3. Set up your API key (only needed for script 03):**
Copy `.env.example` to a new file named `.env`, and paste your free Gemini API key into it:
```
cp .env.example .env
```
Then open `.env` and replace `your_key_here` with your real key (get one free at aistudio.google.com).

**4. Run each script in order:**
```
python3 code/01_data_profiling_and_prediction_model.py
python3 code/02_transition_model.py
python3 code/03_llm_reviewer_copilot.py
```

Each script prints its results directly in the terminal and saves its output files into the `outputs/` folder.

## Running in Google Colab instead
If your team prefers Colab: upload the data CSV via the folder icon, and for script 03, save your key as a Colab secret named `GEMINI_API_KEY` instead of using a `.env` file (see reports for setup steps).

## Data
Currently built and tested against a practice/synthetic dataset since the official organizer data pack has not yet been released. Column names match the fields described in the official problem statement, so swapping in the real `loan_monthly_performance_train.csv` / `loan_monthly_performance_test.csv` requires no code changes beyond the filename in `data/`.

## Team
[Add your team name and member names here]
