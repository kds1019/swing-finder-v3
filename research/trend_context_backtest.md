# Trend-context bucket backtest — trend_continuation vs. reversion_bounce

Phase 2 of core/trend_context.py's trend-context layer — see docs/strategy.md. Compare against research/portfolio_backtest.md (the un-bucketed baseline this was forked from) for the blended-portfolio effect of the split.

- universe: 482 cached tickers (survivorship-biased), 2021-06-01 .. 2026-09-11
- 100,000 start, 4% risk/trade (reversion_bounce sized at 0.5x that), max 6 positions, 20% position cap, 3/sector, 5bps slip
- trend_continuation exit: trail +2R activate / give 1R (unchanged baseline); reversion_bounce exit: fixed stop/target, trailing OFF; both capped at 30-bar max hold
- regime filter (SPY > 200-SMA to open): ON
- candidate priority when signals compete for capacity: HARD PRIORITY: trend_continuation > reversion_bounce > unclassified, deepest pullback first within a tier

## Blended portfolio result

| metric | strategy | SPY (same window) |
|---|---|---|
| total return | +42.9% | +82.7% |
| CAGR | +7.0% | +12.1% |
| max drawdown | -19.4% | -25.4% |
| Sharpe (daily, ann.) | 0.60 | — |

## Per-bucket trade stats (the actual A/B)

| setup_type | trades | win% | avg R | median R | profit factor | avg hold (bars) |
|---|---|---|---|---|---|---|
| trend_continuation | 164 | 37.2 | 0.062 | -1.0 | 1.1 | 8.3 |
| reversion_bounce | 444 | 28.4 | 0.017 | -1.0 | 1.02 | 9.5 |
| unclassified | 35 | 60.0 | 0.387 | 0.2 | 2.04 | 8.7 |
| **all (blended)** | 643 | 32.3 | 0.049 | -1.0 | 1.07 | 9.1 |

## By year and bucket

| year | setup_type | trades | win% | avg R |
|---|---|---|---|---|
| 2021 | reversion_bounce | 22 | 32% | +0.24 |
| 2021 | trend_continuation | 18 | 22% | -0.26 |
| 2021 | unclassified | 2 | 100% | +1.22 |
| 2022 | reversion_bounce | 37 | 19% | -0.30 |
| 2022 | trend_continuation | 20 | 30% | -0.27 |
| 2022 | unclassified | 2 | 50% | -0.39 |
| 2023 | reversion_bounce | 97 | 25% | -0.19 |
| 2023 | trend_continuation | 25 | 36% | -0.04 |
| 2023 | unclassified | 5 | 100% | +0.98 |
| 2024 | reversion_bounce | 111 | 36% | +0.30 |
| 2024 | trend_continuation | 41 | 41% | +0.12 |
| 2024 | unclassified | 15 | 40% | +0.35 |
| 2025 | reversion_bounce | 105 | 25% | -0.04 |
| 2025 | trend_continuation | 39 | 41% | +0.16 |
| 2025 | unclassified | 11 | 64% | +0.15 |
| 2026 | reversion_bounce | 72 | 31% | +0.04 |
| 2026 | trend_continuation | 21 | 43% | +0.47 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected. reversion_bounce sizing/exit constants (REVERSION_BOUNCE_SIZE_MULT) are a first cut, not independently calibrated — revisit once this backtest's per-bucket numbers are in.