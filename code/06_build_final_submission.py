import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder

# ============================================================
# Builds the FINAL submission.csv matching every field described
# in the problem statement's submission_template.csv:
# probabilities, next state, exception type, anomaly score,
# top drivers, action, confidence.
# ============================================================

TRAIN_FILE = 'data/sim_train.csv'
TEST_FILE = 'data/sim_test.csv'
REPORT_FILE = 'reports/06_final_submission_results.md'

report_lines = []
def log(text=""):
    print(text)
    report_lines.append(str(text))

log("# Final Submission Build — Full Required Format\n")

# ============================================================
# STEP 1: Load and clean train data
# ============================================================
df = pd.read_csv(TRAIN_FILE)
df['current_balance'] = df['current_balance'].abs()
df['interest_rate'] = df['interest_rate'].fillna(df['interest_rate'].median())
median_rate = df['interest_rate'].median()

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
    known = set(encoder.classes_)
    return series.astype(str).apply(lambda x: encoder.transform([x])[0] if x in known else -1)

feature_cols = [
    'loan_age_months', 'remaining_term_months', 'original_balance',
    'current_balance', 'interest_rate', 'days_past_due',
    'modification_flag', 'prepayment_flag'
] + [c + '_enc' for c in categorical_cols]

df['reporting_month'] = pd.to_datetime(df['reporting_month'])
cutoff_date = df['reporting_month'].quantile(0.8)
train_df = df[df['reporting_month'] <= cutoff_date]

# ============================================================
# STEP 2: Train THREE separate models -- one per probability required
# ============================================================
log("## Training Models\n")
targets = {
    'next_3m_delinquency_flag': 'predicted_delinquency_prob',
    'next_12m_default_flag': 'predicted_default_prob',
    'next_12m_prepayment_flag': 'predicted_prepayment_prob',
}

models = {}
importances_by_target = {}
for target, prob_col in targets.items():
    m = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    m.fit(train_df[feature_cols], train_df[target])
    models[target] = m
    importances_by_target[prob_col] = pd.Series(m.feature_importances_, index=feature_cols).sort_values(ascending=False)
    log(f"- Trained model for `{target}` -> `{prob_col}`")
log("")

# ============================================================
# STEP 3: Build the monthly transition matrix (for "next state")
# ============================================================
df_sorted = df.sort_values(['loan_id', 'month_index'])
df_sorted['next_month_status'] = df_sorted.groupby('loan_id')['current_status'].shift(-1)
transitions = df_sorted.dropna(subset=['next_month_status'])
transition_matrix = pd.crosstab(transitions['current_status'], transitions['next_month_status'], normalize='index')
most_likely_next = transition_matrix.idxmax(axis=1)
log("## Transition Matrix Built (for `next_state` predictions)\n")
log(f"- States covered: {list(most_likely_next.index)}\n")

# ============================================================
# STEP 4: Anomaly detection
# ============================================================
anomaly_features = ['original_balance', 'current_balance', 'interest_rate', 'days_past_due', 'loan_age_months']
iso = IsolationForest(contamination=0.03, random_state=42)
iso.fit(df[anomaly_features])

# ============================================================
# STEP 5: Load and prepare test data
# ============================================================
log(f"## Scoring Test File: `{TEST_FILE}`\n")
test_df = pd.read_csv(TEST_FILE)
test_df['current_balance'] = test_df['current_balance'].abs()
test_df['interest_rate'] = test_df['interest_rate'].fillna(median_rate)
for col in categorical_cols:
    test_df[col + '_enc'] = safe_encode(test_df[col], encoders[col])

# ============================================================
# STEP 6: Generate all required fields
# ============================================================
submission = test_df[['loan_id']].copy()

# -- Probabilities (3 separate ones)
for target, prob_col in targets.items():
    submission[prob_col] = models[target].predict_proba(test_df[feature_cols])[:, 1]

# -- Next state (from transition matrix, using current_status)
submission['next_state'] = test_df['current_status'].map(most_likely_next).fillna('unknown')

# -- Anomaly score
anomaly_raw = iso.predict(test_df[anomaly_features])
submission['anomaly_score'] = pd.Series(anomaly_raw).map({1: 0, -1: 1}).values

# -- Exception type (rule-based, combining anomaly + specific data issues)
def determine_exception_type(row, test_row):
    if test_row['current_balance'] > test_row['original_balance'] * 1.3:
        return 'balance_increase_anomaly'
    if row['anomaly_score'] == 1:
        return 'flagged_anomaly'
    if test_row['days_past_due'] >= 90:
        return 'severe_delinquency'
    return 'none'

submission['exception_type'] = [
    determine_exception_type(submission.iloc[i], test_df.iloc[i]) for i in range(len(submission))
]
submission['exception_required'] = (submission['exception_type'] != 'none').astype(int)

# -- Top drivers (top 2 features, per the model that matters most -- default risk)
top_2_drivers = importances_by_target['predicted_default_prob'].head(2).index.tolist()
submission['top_drivers'] = ', '.join(top_2_drivers)

# -- Action and confidence (based on default probability, the primary risk signal)
submission['action'] = np.where(submission['predicted_default_prob'] > 0.5, 'review', 'monitor')
submission['confidence'] = np.where(submission['predicted_default_prob'] > 0.7, 'high',
                              np.where(submission['predicted_default_prob'] > 0.3, 'medium', 'low'))

# ============================================================
# STEP 7: Save
# ============================================================
os.makedirs('outputs', exist_ok=True)
submission.to_csv('outputs/submission.csv', index=False)

log(f"## Final submission.csv Columns\n")
log(f"`{list(submission.columns)}`\n")
log(f"Saved **outputs/submission.csv** with {len(submission)} rows.\n")
log("### Sample rows\n")
log(submission.head(5).to_markdown(index=False))

os.makedirs('reports', exist_ok=True)
with open(REPORT_FILE, 'w') as f:
    f.write("\n".join(report_lines))

print(f"\n\n=== DONE. Full report saved to {REPORT_FILE} ===")