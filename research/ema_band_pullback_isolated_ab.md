# EMA-band pullback (10+ day stabilization) — standalone isolated backtest (RESEARCH)

The uptrend_pullback_ema_band_10d bucket from research/ema_band_pullback_v2_ab.py, tested on its OWN dedicated capital — no competition for capacity against reversion_bounce or any other bucket, unlike the blended-portfolio version where only 11 of 318 raw signals ever got traded. Criteria: TrendState == uptrend, KnifeRiskTier == stabilising, price <= 50-EMA, depth >= -20% vs EMA200, days_since_pullback_low >= 10. Same trade management as every other isolated test here (swing-low/EMA-anchored stop, Fibonacci-extension target, +2R/give-1R trailing exit) and the same engine/costs/sizing/sector cap as research/current_trend_gate_ab.py.
- 318 raw signals, 160 tickers, 2021-06-01 .. present
- engine: research/current_trend_gate_ab.py's run()/stat() — same costs/sizing/sector cap as every other isolated test in this repo

## Result

| | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| full | +15% | -9% | 0.45 | 175 | 37% | +0.25 | 1.39 | 9 |
| full SPY | +83% | | | | | | | |
| 2021-2024 | +5% | -6% | 0.26 | 108 | 36% | +0.32 | 1.50 | 6 |
| 2021-2024 SPY | +40% | | | | | | | |
| 2025-2026 | +9% | -5% | 0.81 | 67 | 37% | +0.13 | 1.21 | 9 |
| 2025-2026 SPY | +31% | | | | | | | |
| 2022 | -4% | -4% | -1.14 | 16 | 31% | +0.22 | 1.32 | 4 |
| 2022 SPY | -20% | | | | | | | |

## Exit-reason breakdown

| reason | count | avgR |
|---|---|---|
| expired | 5 | +1.05 |
| stop_hit | 111 | -1.00 |
| target_hit | 18 | +3.99 |
| trail_stop | 41 | +1.88 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected. This is a first isolated backtest of a brand-new bucket — treat with the same skepticism as any first result, not as validated the way trend_continuation/reversion_bounce are._