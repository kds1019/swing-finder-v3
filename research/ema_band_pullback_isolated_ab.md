# EMA-band pullback (8+ day stabilization) — standalone isolated backtest (RESEARCH)

The live core.trend_context.EMA_BAND_STABILIZATION_MIN_DAYS threshold, tested on its OWN dedicated capital — no competition for capacity against reversion_bounce or any other bucket, unlike the blended-portfolio version where this pattern mostly lost capacity fights under the portfolio's depth-based ordering. Criteria: TrendState == uptrend, KnifeRiskTier == stabilising, NOT already trend_continuation, price <= 50-EMA, depth >= -20% vs EMA200, days_since_pullback_low >= 8. Same trade management as every other isolated test here (swing-low/EMA-anchored stop, Fibonacci-extension target, +2R/give-1R trailing exit) and the same engine/costs/sizing/sector cap as research/current_trend_gate_ab.py.
- 537 raw signals, 207 tickers, 2021-06-01 .. present
- engine: research/current_trend_gate_ab.py's run()/stat() — same costs/sizing/sector cap as every other isolated test in this repo

## Result

| | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| full | +29% | -10% | 0.63 | 287 | 36% | +0.26 | 1.41 | 12 |
| full SPY | +83% | | | | | | | |
| 2021-2024 | +6% | -10% | 0.25 | 173 | 34% | +0.26 | 1.39 | 10 |
| 2021-2024 SPY | +40% | | | | | | | |
| 2025-2026 | +22% | -8% | 1.27 | 114 | 40% | +0.26 | 1.44 | 12 |
| 2025-2026 SPY | +31% | | | | | | | |
| 2022 | -6% | -6% | -1.38 | 23 | 35% | +0.36 | 1.55 | 5 |
| 2022 SPY | -20% | | | | | | | |

## Exit-reason breakdown

| reason | count | avgR |
|---|---|---|
| expired | 12 | +1.12 |
| stop_hit | 180 | -1.00 |
| target_hit | 30 | +4.11 |
| trail_stop | 65 | +1.80 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected. This is a first isolated backtest of a brand-new bucket — treat with the same skepticism as any first result, not as validated the way trend_continuation/reversion_bounce are._