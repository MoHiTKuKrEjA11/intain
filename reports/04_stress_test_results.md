# Advanced Feature: Synthetic-Data Stress Testing

This test deliberately injects extreme, broken data into the pipeline to confirm it doesn't crash under real-world messy conditions.

## Problems Deliberately Injected

| Problem | Count |
|---|---|
| Missing `interest_rate` (heavier than normal) | ~4937 rows |
| Extreme balance outliers (huge + very negative) | 50 rows |
| Categories never seen in normal data | 30 rows |
| Broken dates (year 1900) | 10 rows |
| Negative `loan_age_months` | 15 rows |
| `days_past_due` = 9999 | 10 rows |
| Duplicate rows | 20 rows |
| Fully empty column | 1 column |

## Result

**PASSED.** The pipeline completed with no crashes, despite all injected problems above.

- Rows processed: 16909
- Predictions generated: 3017
- Anomalies detected: 508

## What This Proves

- Missing value handling survives heavier-than-normal missingness
- Outlier correction (`.abs()`) survives extreme values without breaking downstream steps
- Category encoding survives brand-new categories never seen in the base data
- Date-based splitting survives corrupted date values
- The pipeline is not fragile to a handful of duplicate or malformed rows