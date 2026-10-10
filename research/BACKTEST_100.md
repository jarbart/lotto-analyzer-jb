# Lotto Analyzer JB vs Random — Backtest (100-Draw Window)

## Overview

This experiment compares the Lotto Analyzer JB heuristic generator against a uniformly random Lotto 6/49 baseline.

The generator uses a rolling history window of 100 previous draws. Performance was evaluated over the latest 300 historical draws.

## Configuration

| Parameter | Value |
|---|---|
| History window | 100 previous draws |
| Evaluation draws | 300 |
| Combinations per draw | 10 |
| Candidate pool | 1,500 |
| Refinement | 50 |
| Generator seed | 42 |
| Random seed | 123556 |
| Optional mask | OFF |

## Results

| Metric | Lotto Analyzer JB | Random | Difference |
|---|---:|---:|---:|
| TOP 1 — average matches | 0.753 | 0.750 | +0.003 |
| TOP 1 — at least 2 matches | 13.3% | 16.7% | -3.3 pp |
| TOP 1 — at least 3 matches | 1.7% | 3.0% | -1.3 pp |
| BEST TOP 10 — average matches | **2.087** | 2.000 | **+0.087** |
| BEST TOP 10 — at least 2 matches | **91.0%** | 83.7% | **+7.3 pp** |
| BEST TOP 10 — at least 3 matches | **16.3%** | 15.7% | **+0.7 pp** |
| BEST TOP 10 — at least 4 matches | **1.3%** | 0.7% | **+0.6 pp** |

*Differences use the rounded values printed by the backtest. Minor discrepancies may result from rounding.*

## Key Finding

In this backtest, Lotto Analyzer JB achieved a higher best-of-ten average match count (2.087 vs 2.000) and a higher proportion of draws with at least two matches among ten generated combinations (91.0% vs 83.7%).

The TOP 1 comparison did not demonstrate a consistent advantage over random selection.

## Generation Profiles

The generator produced a concentrated distribution of G1/G2/G3 profiles:

- 3/2/1: 53.1% of generated combinations.
- 2/2/2: 31.2% of generated combinations.

The random baseline produced a broader distribution, with 3/2/1 accounting for 7.2% and 2/2/2 for 15.0%.

This demonstrates a structural difference between the strategies, not proof of predictive power.

## Interpretation

The observed BEST TOP 10 advantage is specific to this historical evaluation and one random baseline seed.

The history window of 100 was selected after earlier comparisons of 30, 50, and 100 draws. Therefore, the result should be considered exploratory rather than an independent validation.

BEST TOP 10 measures the maximum number of matches among ten combinations after each draw. It should not be interpreted as the performance of a single ticket.

## Reproducibility

The experiment is implemented in `backtest.py` using `run_window_test()`.

The complete console output is available in [`results/backtest_100.txt`](results/backtest_100.txt).

## Disclaimer

This project is for educational and statistical research. In a fair Lotto 6/49 draw, every distinct six-number combination has the same probability of winning the jackpot. Historical backtest results do not guarantee future performance.