import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CONFIG: change these two lines when the real files arrive
# ============================================================
TRAIN_FILE = 'data/sim_train.csv'
TEST_FILE = 'data/sim_test.csv'
REPORT_FILE = 'reports/01_results.md'

report_lines = []
def log(text=""):
    print(text)
    report_lines.append(str(text))

log("# Task 1, 2, 4, 5, 6 Results — Data Profiling, Prediction, Anomaly, Scenario, Explainability\n")

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
df = pd.read_csv(TRAIN_FILE)
log(f"## Data Loaded\n- Training file: `{TRAIN_FILE}`\n- Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

# ============================================================
# STEP 2: CLEAN DATA
# ============================================================
neg_balance_count = (df['current_balance'] < 0).sum()
missing_rate_count = df['interest_rate'].isnull().sum()

df['current_balance'] = df['current_balance'].abs()
df['interest_rate'] = df['interest_rate'].fillna(df['interest_rate'].median())
median_rate = df['interest_rate'].median()

log("## Data Cleaning (Task 1)\n")
log(f"- Found and fixed **{neg_balance_count} rows** with negative `current_balance` (invalid).")
log(f"- Found and filled **{missing_rate_count} rows** with missing `interest_rate` (filled with median).\n")

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
categorical_cols = ['credit_score_band', 'ltv_band', 'dti_band', 'state',
                     'loan_purpose', 'occupancy_type', 'property_type',
                     'servicer_name', 'current_status', 'source_system',
                     'document_status']

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

def safe_encode(series, encoder):
    known_classes = set(encoder.classes_)
    return series.astype(str).apply(
        lambda x: encoder.transform([x])[0] if x in known_classes else -1
    )

feature_cols = [
    'loan_age_months', 'remaining_term_months', 'original_balance',
    'current_balance', 'interest_rate', 'days_past_due',
    'modification_flag', 'prepayment_flag'
] + [c + '_enc' for c in categorical_cols]

log(f"## Feature Engineering\n- {len(feature_cols)} features used\n")

# ============================================================
# STEP 4: TIME-AWARE TRAIN/VALIDATION SPLIT
# ============================================================
df['reporting_month'] = pd.to_datetime(df['reporting_month'])
cutoff_date = df['reporting_month'].quantile(0.8)

train_df = df[df['reporting_month'] <= cutoff_date]
val_df = df[df['reporting_month'] > cutoff_date]

log("## Time-Aware Train/Validation Split (Task 2)\n")
log(f"- Train rows: {len(train_df)}")
log(f"- Validation rows: {len(val_df)}")
log(f"- Cutoff date: {cutoff_date.date()}")
log("- Split by date (not random) so future information never leaks into training.\n")

# ============================================================
# STEP 5: TRAIN THREE PREDICTION MODELS (delinquency, default, prepayment)
# ============================================================
targets = {
    'next_3m_delinquency_flag': 'predicted_delinquency_prob',
    'next_12m_default_flag': 'predicted_default_prob',
    'next_12m_prepayment_flag': 'predicted_prepayment_prob',
}

models = {}
importances_by_target = {}
log(f"## Model Results\n")
for target, prob_col in targets.items():
    m = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    m.fit(train_df[feature_cols], train_df[target])
    models[target] = m

    val_probs = m.predict_proba(val_df[feature_cols])[:, 1]
    val_preds = m.predict(val_df[feature_cols])
    auc = roc_auc_score(val_df[target], val_probs)
    importances_by_target[prob_col] = pd.Series(m.feature_importances_, index=feature_cols).sort_values(ascending=False)

    log(f"### Predicting `{target}`\n")
    log(f"- **ROC-AUC: {auc:.3f}**\n")
    log("```")
    log(classification_report(val_df[target], val_preds))
    log("```\n")

# Use the default model as the "primary" one for feature importance display and action/confidence logic
model = models['next_12m_default_flag']
importances = importances_by_target['predicted_default_prob']

log("## Top 10 Features Driving Default Predictions (Task 6)\n")
log(importances.head(10).round(4).to_markdown())
log("")

# ============================================================
# STEP 6: BUILD TRANSITION MATRIX (for next_state, ties to Task 3)
# ============================================================
df_sorted = df.sort_values(['loan_id', 'month_index'])
df_sorted['next_month_status'] = df_sorted.groupby('loan_id')['current_status'].shift(-1)
transitions = df_sorted.dropna(subset=['next_month_status'])
transition_matrix = pd.crosstab(transitions['current_status'], transitions['next_month_status'], normalize='index')
most_likely_next = transition_matrix.idxmax(axis=1)

# ============================================================
# STEP 7: ANOMALY DETECTION (Task 4)
# ============================================================
anomaly_features = ['original_balance', 'current_balance', 'interest_rate',
                     'days_past_due', 'loan_age_months']
iso = IsolationForest(contamination=0.03, random_state=42)
df['anomaly_score'] = iso.fit_predict(df[anomaly_features])
df['anomaly_score'] = df['anomaly_score'].map({1: 0, -1: 1})

log(f"## Anomaly Detection (Task 4)\n")
log(f"- Anomalies flagged: **{df['anomaly_score'].sum()}** out of {len(df)} rows ({df['anomaly_score'].mean()*100:.1f}%)\n")
log("### Sample anomalous records\n")
sample_anomalies = df[df['anomaly_score'] == 1][['loan_id', 'current_balance', 'interest_rate', 'days_past_due']].head(5)
log(sample_anomalies.to_markdown(index=False))
log("")

# ============================================================
# STEP 8: SCENARIO SIMULATION (Task 5)
# ============================================================
base_pred = model.predict_proba(df[feature_cols])[:, 1].mean()

adverse_df = df.copy()
adverse_df['days_past_due'] = adverse_df['days_past_due'] * 1.5
adverse_df['interest_rate'] = adverse_df['interest_rate'] + 1.5
adverse_pred = model.predict_proba(adverse_df[feature_cols])[:, 1].mean()

prepay_df = df.copy()
prepay_df['interest_rate'] = prepay_df['interest_rate'] - 1.0
prepay_pred = model.predict_proba(prepay_df[feature_cols])[:, 1].mean()

log("## Scenario Simulation (Task 5)\n")
scenario_table = pd.DataFrame({
    'Scenario': ['Base', 'Adverse-credit', 'High-prepayment'],
    'Avg Predicted Default Probability': [round(base_pred,4), round(adverse_pred,4), round(prepay_pred,4)]
})
log(scenario_table.to_markdown(index=False))
log("")

# ============================================================
# STEP 9: BUILD FINAL SUBMISSION FILE -- full required format
# ============================================================
log(f"## Final Predictions\n- Loading official test file: `{TEST_FILE}`\n")
test_df = pd.read_csv(TEST_FILE)
test_df['current_balance'] = test_df['current_balance'].abs()
test_df['interest_rate'] = test_df['interest_rate'].fillna(median_rate)

for col in categorical_cols:
    test_df[col + '_enc'] = safe_encode(test_df[col], encoders[col])

submission = test_df[['loan_id']].copy()

# -- 3 required probability columns
for target, prob_col in targets.items():
    submission[prob_col] = models[target].predict_proba(test_df[feature_cols])[:, 1]

# -- next_state (from the transition matrix)
submission['next_state'] = test_df['current_status'].map(most_likely_next).fillna('unknown')

# -- anomaly_score
anomaly_raw = iso.predict(test_df[anomaly_features])
submission['anomaly_score'] = pd.Series(anomaly_raw).map({1: 0, -1: 1}).values

# -- exception_type (rule-based)
def determine_exception_type(anomaly_flag, balance_ratio, dpd):
    if balance_ratio > 1.3:
        return 'balance_increase_anomaly'
    if anomaly_flag == 1:
        return 'flagged_anomaly'
    if dpd >= 90:
        return 'severe_delinquency'
    return 'none'

balance_ratios = test_df['current_balance'] / test_df['original_balance'].replace(0, np.nan)
submission['exception_type'] = [
    determine_exception_type(submission['anomaly_score'].iloc[i], balance_ratios.iloc[i], test_df['days_past_due'].iloc[i])
    for i in range(len(submission))
]
submission['exception_required'] = (submission['exception_type'] != 'none').astype(int)

# -- top_drivers (top 2 global features for the default model)
top_2_drivers = importances.head(2).index.tolist()
submission['top_drivers'] = ', '.join(top_2_drivers)

# -- action and confidence (based on default probability, the primary risk signal)
submission['action'] = np.where(submission['predicted_default_prob'] > 0.5, 'review', 'monitor')
submission['confidence'] = np.where(submission['predicted_default_prob'] > 0.7, 'high',
                             np.where(submission['predicted_default_prob'] > 0.3, 'medium', 'low'))

os.makedirs('outputs', exist_ok=True)
submission.to_csv('outputs/submission.csv', index=False)

log(f"- Saved `outputs/submission.csv` with **{len(submission)} rows** and **{len(submission.columns)} columns**, matching the required format (probabilities, next state, exception type, anomaly score, top drivers, action, confidence)\n")
log("### Sample predictions\n")
log(submission.head(5).to_markdown(index=False))

os.makedirs('reports', exist_ok=True)
with open(REPORT_FILE, 'w') as f:
    f.write("\n".join(report_lines))

print(f"\n\n=== DONE. Full report saved to {REPORT_FILE} ===")