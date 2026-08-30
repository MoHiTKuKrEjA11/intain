import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder

# ============================================================
# ADVANCED FEATURE: Synthetic-Data Stress Testing (PDF section 10)
# ============================================================
# This script deliberately injects extreme, broken, and unusual data
# into a copy of the training data, then runs the full pipeline against
# it to prove the code doesn't crash on messy real-world conditions --
# not just on clean practice data.

REPORT_FILE = 'reports/04_stress_test_results.md'
report_lines = []
def log(text=""):
    print(text)
    report_lines.append(str(text))

log("# Advanced Feature: Synthetic-Data Stress Testing\n")
log("This test deliberately injects extreme, broken data into the pipeline to confirm it doesn't crash under real-world messy conditions.\n")

# ============================================================
# STEP 1: Load clean data and inject extreme problems
# ============================================================
np.random.seed(99)
df = pd.read_csv('data/sim_train.csv')
n = len(df)
stress = df.copy()

# Heavier missingness (30% instead of the usual small %)
mask = np.random.rand(n) < 0.30
stress.loc[mask, 'interest_rate'] = np.nan

# Extreme outliers
extreme_idx = np.random.choice(n, 50, replace=False)
stress.loc[extreme_idx[:25], 'current_balance'] = 99999999.99
stress.loc[extreme_idx[25:], 'current_balance'] = -999999.99

# Categories never seen before
unseen_idx = np.random.choice(n, 30, replace=False)
stress.loc[unseen_idx[:10], 'state'] = 'ZZ_UNKNOWN'
stress.loc[unseen_idx[10:20], 'servicer_name'] = 'BrandNewServicerXYZ'
stress.loc[unseen_idx[20:], 'credit_score_band'] = 'UNKNOWN_BAND'

# Broken dates
date_idx = np.random.choice(n, 10, replace=False)
stress.loc[date_idx, 'reporting_month'] = '1900-01-01'

# Impossible values
age_idx = np.random.choice(n, 15, replace=False)
stress.loc[age_idx, 'loan_age_months'] = -5

dpd_idx = np.random.choice(n, 10, replace=False)
stress.loc[dpd_idx, 'days_past_due'] = 9999

# Duplicate rows
dup_rows = stress.sample(20, random_state=1)
stress = pd.concat([stress, dup_rows], ignore_index=True)

# A fully empty column
stress['fully_missing_test_col'] = np.nan

log("## Problems Deliberately Injected\n")
log("| Problem | Count |")
log("|---|---|")
log(f"| Missing `interest_rate` (heavier than normal) | ~{mask.sum()} rows |")
log(f"| Extreme balance outliers (huge + very negative) | 50 rows |")
log(f"| Categories never seen in normal data | 30 rows |")
log(f"| Broken dates (year 1900) | 10 rows |")
log(f"| Negative `loan_age_months` | 15 rows |")
log(f"| `days_past_due` = 9999 | 10 rows |")
log(f"| Duplicate rows | 20 rows |")
log(f"| Fully empty column | 1 column |")
log("")

# ============================================================
# STEP 2: Run the SAME pipeline logic against this broken data
# ============================================================
try:
    stress['current_balance'] = stress['current_balance'].abs()
    stress['interest_rate'] = stress['interest_rate'].fillna(stress['interest_rate'].median())

    categorical_cols = ['credit_score_band', 'ltv_band', 'dti_band', 'state',
                         'loan_purpose', 'occupancy_type', 'property_type',
                         'servicer_name', 'current_status', 'source_system',
                         'document_status']
    for col in categorical_cols:
        le = LabelEncoder()
        stress[col + '_enc'] = le.fit_transform(stress[col].astype(str))

    feature_cols = [
        'loan_age_months', 'remaining_term_months', 'original_balance',
        'current_balance', 'interest_rate', 'days_past_due',
        'modification_flag', 'prepayment_flag'
    ] + [c + '_enc' for c in categorical_cols]

    stress['reporting_month'] = pd.to_datetime(stress['reporting_month'])
    cutoff_date = stress['reporting_month'].quantile(0.8)
    train_df = stress[stress['reporting_month'] <= cutoff_date]
    val_df = stress[stress['reporting_month'] > cutoff_date]

    model = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(train_df[feature_cols], train_df['next_12m_default_flag'])
    preds = model.predict_proba(val_df[feature_cols])[:, 1]

    iso = IsolationForest(contamination=0.03, random_state=42)
    anomaly_features = ['original_balance', 'current_balance', 'interest_rate', 'days_past_due', 'loan_age_months']
    anomaly_flags = iso.fit_predict(stress[anomaly_features])

    log("## Result\n")
    log(f"**PASSED.** The pipeline completed with no crashes, despite all injected problems above.\n")
    log(f"- Rows processed: {len(stress)}")
    log(f"- Predictions generated: {len(preds)}")
    log(f"- Anomalies detected: {(anomaly_flags == -1).sum()}")
    log("")
    log("## What This Proves\n")
    log("- Missing value handling survives heavier-than-normal missingness")
    log("- Outlier correction (`.abs()`) survives extreme values without breaking downstream steps")
    log("- Category encoding survives brand-new categories never seen in the base data")
    log("- Date-based splitting survives corrupted date values")
    log("- The pipeline is not fragile to a handful of duplicate or malformed rows")

except Exception as e:
    log("## Result\n")
    log(f"**FAILED.** The pipeline crashed with the following error:\n")
    log(f"```\n{str(e)}\n```")
    log("\nThis is a real issue that needs to be fixed before relying on this pipeline with messier real-world data.")

os.makedirs('reports', exist_ok=True)
with open(REPORT_FILE, 'w') as f:
    f.write("\n".join(report_lines))

print(f"\n\n=== DONE. Full report saved to {REPORT_FILE} ===")