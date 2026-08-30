# Task 3 Results — Monthly Transition Model

## What This Model Does

Answers: *"if a loan is in state X this month, what's the probability it moves to state Y next month?"*

## Monthly State Transition Probabilities

(Rows = current state, Columns = next month's state, values = probability)

| current_status     |   current |   default |   delinquent_30_60 |   delinquent_90_plus |   prepaid |
|:-------------------|----------:|----------:|-------------------:|---------------------:|----------:|
| current            |     0.939 |     0.007 |              0.022 |                0.018 |     0.015 |
| delinquent_30_60   |     0.533 |     0.012 |              0.421 |                0.016 |     0.018 |
| delinquent_90_plus |     0     |     0.135 |              0.592 |                0.261 |     0.012 |

## Comparison Against Baseline

| Approach                                          |   Accuracy |
|:--------------------------------------------------|-----------:|
| Naive baseline (always guess "no change")         |      0.895 |
| Transition model (predict most likely next state) |      0.909 |

**Conclusion:** the transition model beats the naive baseline (90.9% vs 89.5% accuracy), showing it captures real patterns in how loans move between states month to month.

Saved `outputs/transition_matrix.csv`