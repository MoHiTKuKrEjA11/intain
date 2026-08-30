import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================================
# ADVANCED FEATURE: Bias / Fairness Analysis (PDF section 10)
# ============================================================
# Checks whether the model's predicted risk differs meaningfully across
# groups (state, occupancy type, credit band) in ways that could signal
# unfair treatment -- not because these groups SHOULD predict risk
# differently, but to surface and disclose any large gaps for review.

REPORT_FILE = 'reports/05_bias_fairness_results.md'
report_lines = []
def log(text=""):
    print(text)
    report_lines.append(str(text))

log("# Advanced Feature: Bias / Fairness Analysis\n")
log("Checks whether predicted default risk varies meaningfully across groups (state, occupancy type, credit score band), to surface any large, unexplained gaps for human review.\n")

# ============================================================
# STEP 1: Train the model (same as script 01)
# ============================================================
df = pd.read_csv('data/sim_train.csv')
df['current_balance'] = df['current_balance'].abs()
df['interest_rate'] = df['interest_rate'].fillna(df['interest_rate'].median())

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

model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(train_df[feature_cols], train_df['next_12m_default_flag'])

# ============================================================
# STEP 2: Score the test set
# ============================================================
test_df = pd.read_csv('data/sim_test.csv')
test_df['current_balance'] = test_df['current_balance'].abs()
test_df['interest_rate'] = test_df['interest_rate'].fillna(df['interest_rate'].median())
for col in categorical_cols:
    test_df[col + '_enc'] = safe_encode(test_df[col], encoders[col])

test_df['predicted_default_prob'] = model.predict_proba(test_df[feature_cols])[:, 1]

# ============================================================
# STEP 3: Check average predicted risk by group
# ============================================================
def fairness_table(df, group_col, min_group_size=20):
    grouped = df.groupby(group_col)['predicted_default_prob'].agg(['mean', 'count']).reset_index()
    grouped = grouped[grouped['count'] >= min_group_size]  # ignore tiny groups (unreliable averages)
    grouped.columns = [group_col, 'avg_predicted_risk', 'num_loans']
    grouped['avg_predicted_risk'] = grouped['avg_predicted_risk'].round(4)
    return grouped.sort_values('avg_predicted_risk', ascending=False)

for group_col, label in [('state', 'State'), ('occupancy_type', 'Occupancy Type'), ('credit_score_band', 'Credit Score Band')]:
    table = fairness_table(test_df, group_col)
    log(f"## Average Predicted Risk by {label}\n")
    log(table.to_markdown(index=False))
    if len(table) > 1:
        spread = table['avg_predicted_risk'].max() - table['avg_predicted_risk'].min()
        log(f"\n**Spread across groups: {spread:.3f}** (highest minus lowest average predicted risk)")
        if group_col == 'credit_score_band':
            log("- This spread is EXPECTED here -- credit score band should legitimately predict different risk levels.")
        else:
            if spread > 0.15:
                log(f"- ⚠️ This is a notable spread for `{group_col}`, which should not obviously predict default risk on its own. Recommend human review to confirm this isn't an unfair bias, before relying on this model for `{group_col}`-sensitive decisions.")
            else:
                log(f"- This spread is modest and doesn't raise an immediate concern, but should still be monitored as more data becomes available.")
    log("")

log("## Limitations of This Check\n")
log("- This is a simple average-comparison check, not a formal fairness audit (e.g. no statistical significance testing, no demographic parity/equalized odds metrics).")
log("- Small group sizes (fewer than 20 loans) were excluded since their averages are unreliable.")
log("- This should be treated as a starting point for human review, not a certification that the model is unbiased.")

os.makedirs('reports', exist_ok=True)
with open(REPORT_FILE, 'w') as f:
    f.write("\n".join(report_lines))

print(f"\n\n=== DONE. Full report saved to {REPORT_FILE} ===")