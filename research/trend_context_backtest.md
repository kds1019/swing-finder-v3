# Trend-context bucket backtest — trend_continuation vs. reversion_bounce

Phase 2 of core/trend_context.py's trend-context layer — see docs/strategy.md. Compare against research/portfolio_backtest.md (the un-bucketed baseline this was forked from) for the blended-portfolio effect of the split.

- universe: 483 cached tickers (survivorship-biased), 2021-06-01 .. 2026-08-31
- 100,000 start, 4% risk/trade (reversion_bounce sized at 0.5x that), max 6 positions, 20% position cap, 3/sector, 5bps slip
- trend_continuation exit: trail +2R activate / give 1R (unchanged baseline); reversion_bounce exit: fixed stop/target, trailing OFF; both capped at 30-bar max hold
- regime filter (SPY > 200-SMA to open): ON
- candidate priority when signals compete for capacity: trend_continuation > reversion_bounce > unclassified, deepest pullback first within a tier

## Blended portfolio result

| metric | strategy | SPY (same window) |
|---|---|---|
| total return | +43.0% | +82.7% |
| CAGR | +7.1% | +12.2% |
| max drawdown | -78.5% | -25.4% |
| Sharpe (daily, ann.) | 0.38 | — |

## Per-bucket trade stats (the actual A/B)

| setup_type | trades | win% | avg R | median R | profit factor | avg hold (bars) |
|---|---|---|---|---|---|---|
| trend_continuation | 272 | 39.3 | 0.19 | -1.0 | 1.32 | 8.9 |
| reversion_bounce | 398 | 28.1 | 0.084 | -1.0 | 1.12 | 8.0 |
| unclassified | 30 | 50.0 | 0.246 | -0.29 | 1.49 | 7.6 |
| **all (blended)** | 700 | 33.4 | 0.133 | -1.0 | 1.2 | 8.3 |

## By year and bucket

| year | setup_type | trades | win% | avg R |
|---|---|---|---|---|
| 2021 | reversion_bounce | 21 | 19% | +0.08 |
| 2021 | trend_continuation | 15 | 33% | -0.22 |
| 2021 | unclassified | 3 | 33% | -0.46 |
| 2022 | reversion_bounce | 35 | 14% | -0.51 |
| 2022 | trend_continuation | 21 | 33% | -0.08 |
| 2022 | unclassified | 1 | 0% | -1.00 |
| 2023 | reversion_bounce | 93 | 30% | +0.16 |
| 2023 | trend_continuation | 62 | 40% | +0.19 |
| 2023 | unclassified | 3 | 100% | +1.09 |
| 2024 | reversion_bounce | 95 | 29% | +0.10 |
| 2024 | trend_continuation | 89 | 42% | +0.36 |
| 2024 | unclassified | 18 | 50% | +0.44 |
| 2025 | reversion_bounce | 89 | 34% | +0.38 |
| 2025 | trend_continuation | 52 | 40% | +0.15 |
| 2025 | unclassified | 5 | 40% | -0.30 |
| 2026 | reversion_bounce | 65 | 26% | -0.12 |
| 2026 | trend_continuation | 33 | 36% | +0.14 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected. reversion_bounce sizing/exit constants (REVERSION_BOUNCE_SIZE_MULT) are a first cut, not independently calibrated — revisit once this backtest's per-bucket numbers are in.