# Trend-context bucket backtest — trend_continuation vs. reversion_bounce (depth-only-priority ablation)

Phase 2 of core/trend_context.py's trend-context layer — see docs/strategy.md. Compare against research/portfolio_backtest.md (the un-bucketed baseline this was forked from) for the blended-portfolio effect of the split. This run reverts candidate-selection priority to depth-only, to isolate whether the setup_type reorder (not the exit/sizing split) drove the main run's drawdown.

- universe: 482 cached tickers (survivorship-biased), 2021-06-01 .. 2026-08-31
- 100,000 start, 4% risk/trade (reversion_bounce sized at 0.5x that), max 6 positions, 20% position cap, 3/sector, 5bps slip
- trend_continuation exit: trail +2R activate / give 1R (unchanged baseline); reversion_bounce exit: fixed stop/target, trailing OFF; both capped at 30-bar max hold
- regime filter (SPY > 200-SMA to open): ON
- candidate priority when signals compete for capacity: DEPTH-ONLY (ablation: matches baseline's own tie-break, setup_type priority OFF — isolates the exit/sizing split from the reorder effect)

## Blended portfolio result

| metric | strategy | SPY (same window) |
|---|---|---|
| total return | +9.6% | +82.7% |
| CAGR | +1.8% | +12.2% |
| max drawdown | -47.0% | -25.4% |
| Sharpe (daily, ann.) | 0.20 | — |

## Per-bucket trade stats (the actual A/B)

| setup_type | trades | win% | avg R | median R | profit factor | avg hold (bars) |
|---|---|---|---|---|---|---|
| trend_continuation | 2 | 50.0 | 0.93 | 0.93 | 2.86 | 4.5 |
| reversion_bounce | 269 | 26.0 | 0.057 | -1.0 | 1.08 | 7.9 |
| unclassified | 677 | 41.4 | 0.045 | -1.0 | 1.08 | 5.5 |
| **all (blended)** | 948 | 37.0 | 0.05 | -1.0 | 1.08 | 6.2 |

## By year and bucket

| year | setup_type | trades | win% | avg R |
|---|---|---|---|---|
| 2021 | reversion_bounce | 18 | 6% | -0.67 |
| 2021 | unclassified | 41 | 56% | +0.40 |
| 2022 | reversion_bounce | 33 | 9% | -0.51 |
| 2022 | unclassified | 34 | 15% | -0.76 |
| 2023 | reversion_bounce | 60 | 28% | +0.16 |
| 2023 | unclassified | 151 | 40% | -0.09 |
| 2024 | reversion_bounce | 61 | 34% | +0.35 |
| 2024 | trend_continuation | 1 | 0% | -1.00 |
| 2024 | unclassified | 202 | 43% | +0.06 |
| 2025 | reversion_bounce | 53 | 23% | +0.04 |
| 2025 | unclassified | 126 | 42% | +0.10 |
| 2026 | reversion_bounce | 44 | 36% | +0.26 |
| 2026 | trend_continuation | 1 | 100% | +2.86 |
| 2026 | unclassified | 123 | 43% | +0.23 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected. reversion_bounce sizing/exit constants (REVERSION_BOUNCE_SIZE_MULT) are a first cut, not independently calibrated — revisit once this backtest's per-bucket numbers are in.