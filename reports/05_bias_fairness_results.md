# Advanced Feature: Bias / Fairness Analysis

Checks whether predicted default risk varies meaningfully across groups (state, occupancy type, credit score band), to surface any large, unexplained gaps for human review.

## Average Predicted Risk by State

| state   |   avg_predicted_risk |   num_loans |
|:--------|---------------------:|------------:|
| OH      |               0.4337 |         391 |
| NC      |               0.4322 |         456 |
| GA      |               0.4267 |         411 |
| PA      |               0.4257 |         336 |
| NY      |               0.4225 |         345 |
| MI      |               0.4215 |         487 |
| IL      |               0.4204 |         492 |
| FL      |               0.4195 |         413 |
| TX      |               0.4078 |         447 |
| CA      |               0.3912 |         468 |

**Spread across groups: 0.042** (highest minus lowest average predicted risk)
- This spread is modest and doesn't raise an immediate concern, but should still be monitored as more data becomes available.

## Average Predicted Risk by Occupancy Type

| occupancy_type   |   avg_predicted_risk |   num_loans |
|:-----------------|---------------------:|------------:|
| primary          |               0.4211 |        3450 |
| investment       |               0.4207 |         344 |
| second_home      |               0.4071 |         452 |

**Spread across groups: 0.014** (highest minus lowest average predicted risk)
- This spread is modest and doesn't raise an immediate concern, but should still be monitored as more data becomes available.

## Average Predicted Risk by Credit Score Band

| credit_score_band   |   avg_predicted_risk |   num_loans |
|:--------------------|---------------------:|------------:|
| 600-660             |               0.4544 |         342 |
| 660-700             |               0.4322 |         660 |
| 700-740             |               0.4243 |        1207 |
| <600                |               0.42   |         239 |
| 780+                |               0.4122 |         825 |
| 740-780             |               0.399  |         971 |

**Spread across groups: 0.055** (highest minus lowest average predicted risk)
- This spread is EXPECTED here -- credit score band should legitimately predict different risk levels.

## Limitations of This Check

- This is a simple average-comparison check, not a formal fairness audit (e.g. no statistical significance testing, no demographic parity/equalized odds metrics).
- Small group sizes (fewer than 20 loans) were excluded since their averages are unreliable.
- This should be treated as a starting point for human review, not a certification that the model is unbiased.