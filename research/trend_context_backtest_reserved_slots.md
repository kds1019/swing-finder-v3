# Trend-context bucket backtest — trend_continuation vs. reversion_bounce (reserved-slots=2 variant)

Phase 2 of core/trend_context.py's trend-context layer — see docs/strategy.md. Compare against research/portfolio_backtest.md (the un-bucketed baseline this was forked from) for the blended-portfolio effect of the split. This run reserves 2 of 6 slots/day for trend_continuation candidates only, then fills the rest depth-only from the whole pool — a middle ground between the full-priority run (real trend_continuation sample, but it could claim every slot) and the depth-only ablation (sane drawdown, but trend_continuation almost never got a slot).

- universe: 483 cached tickers (survivorship-biased), 2021-06-01 .. 2026-08-31
- 100,000 start, 4% risk/trade (reversion_bounce sized at 0.5x that), max 6 positions, 20% position cap, 3/sector, 5bps slip
- trend_continuation exit: trail +2R activate / give 1R (unchanged baseline); reversion_bounce exit: fixed stop/target, trailing OFF; both capped at 30-bar max hold
- regime filter (SPY > 200-SMA to open): ON
- candidate priority when signals compete for capacity: RESERVED SLOTS: up to 2 of 6 slots reserved for trend_continuation (deepest first), remainder filled depth-only from the whole pool

## Blended portfolio result

| metric | strategy | SPY (same window) |
|---|---|---|
| total return | +9.0% | +82.7% |
| CAGR | +1.7% | +12.2% |
| max drawdown | -60.2% | -25.4% |
| Sharpe (daily, ann.) | 0.28 | — |

## Per-bucket trade stats (the actual A/B)

| setup_type | trades | win% | avg R | median R | profit factor | avg hold (bars) |
|---|---|---|---|---|---|---|
| trend_continuation | 301 | 37.9 | 0.158 | -1.0 | 1.26 | 8.6 |
| reversion_bounce | 173 | 25.4 | 0.051 | -1.0 | 1.07 | 7.2 |
| unclassified | 405 | 41.7 | 0.034 | -1.0 | 1.06 | 4.9 |
| **all (blended)** | 879 | 37.2 | 0.08 | -1.0 | 1.13 | 6.6 |

## By year and bucket

| year | setup_type | trades | win% | avg R |
|---|---|---|---|---|
| 2021 | reversion_bounce | 15 | 13% | -0.29 |
| 2021 | trend_continuation | 16 | 38% | -0.18 |
| 2021 | unclassified | 23 | 52% | +0.20 |
| 2022 | reversion_bounce | 17 | 6% | -0.68 |
| 2022 | trend_continuation | 22 | 18% | -0.57 |
| 2022 | unclassified | 26 | 8% | -0.91 |
| 2023 | reversion_bounce | 39 | 31% | +0.18 |
| 2023 | trend_continuation | 67 | 40% | +0.14 |
| 2023 | unclassified | 79 | 42% | -0.04 |
| 2024 | reversion_bounce | 36 | 33% | +0.37 |
| 2024 | trend_continuation | 87 | 45% | +0.38 |
| 2024 | unclassified | 115 | 42% | -0.05 |
| 2025 | reversion_bounce | 37 | 24% | +0.05 |
| 2025 | trend_continuation | 59 | 34% | +0.06 |
| 2025 | unclassified | 91 | 47% | +0.25 |
| 2026 | reversion_bounce | 29 | 28% | +0.08 |
| 2026 | trend_continuation | 50 | 36% | +0.33 |
| 2026 | unclassified | 71 | 44% | +0.26 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected. reversion_bounce sizing/exit constants (REVERSION_BOUNCE_SIZE_MULT) are a first cut, not independently calibrated — revisit once this backtest's per-bucket numbers are in.