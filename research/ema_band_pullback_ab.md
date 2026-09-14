# EMA-band pullback — an alternative to the Fib-zone requirement for trend_continuation (RESEARCH)

Motivated by a live case: IRM, UAL, COLB, UNFI, SEZL, TTC, KEY, FNB, MIRM were all confirmed uptrend + stabilising but fell through to SetupType=null because none was in the 38.2-61.8% Fib-retracement zone of its own recent 60-session swing — all 9 sit below their 50-EMA (-1.5% to -10.1%) but still above their 200-EMA (+0.7% to +2.7%), a shallow dip the Fib-zone math misread as 'too deep' relative to a small recent swing. Tests price <= 50-EMA (down to the existing, already-live G2 floor of -20% vs EMA200) as an alternative zone, in a NEW bucket (`uptrend_pullback_ema_band`) checked separately from the existing Fib-zone-based `trend_continuation` (priority: trend_continuation > this new bucket > reversion_bounce > null, so this isolates candidates that ONLY qualify the new way).
- signal set: today's live gate held exactly as-is, 29945 signals
- bucket counts: {'reversion_bounce': 16254, nan: 11505, 'uptrend_pullback_ema_band': 1705, 'trend_continuation': 481}
- trade management: DEFAULT for every bucket here (no reversion_bounce-style sizing/exit overrides) — this tests whether the bucket has an edge at all, not how to size it

## Per-bucket trade stats (one portfolio run, default trade management)

| bucket | trades | win% | avgR | medianR | PF |
|---|---|---|---|---|---|
| reversion_bounce | 852 | 37% | +0.23 | -1.00 | 1.36 |
| trend_continuation | 6 | 67% | +1.02 | +1.27 | 4.07 |
| uptrend_pullback_ema_band | 31 | 32% | -0.03 | -1.00 | 0.95 |
| nan | 49 | 31% | +0.06 | -1.00 | 1.08 |

## By year and bucket

| year | bucket | trades | win% | avgR |
|---|---|---|---|---|
| 2021 | reversion_bounce | 58 | 41% | +0.28 |
| 2021 | uptrend_pullback_ema_band | 2 | 50% | +0.18 |
| 2021 | nan | 5 | 40% | +0.44 |
| 2022 | reversion_bounce | 77 | 22% | -0.24 |
| 2022 | uptrend_pullback_ema_band | 2 | 50% | +1.09 |
| 2022 | nan | 6 | 17% | -0.80 |
| 2023 | reversion_bounce | 186 | 37% | +0.25 |
| 2023 | trend_continuation | 1 | 100% | +1.39 |
| 2023 | uptrend_pullback_ema_band | 7 | 14% | -0.33 |
| 2023 | nan | 11 | 9% | -0.78 |
| 2024 | reversion_bounce | 193 | 40% | +0.39 |
| 2024 | trend_continuation | 4 | 50% | +0.39 |
| 2024 | uptrend_pullback_ema_band | 11 | 36% | +0.04 |
| 2024 | nan | 11 | 55% | +0.76 |
| 2025 | reversion_bounce | 195 | 38% | +0.26 |
| 2025 | trend_continuation | 1 | 100% | +3.21 |
| 2025 | uptrend_pullback_ema_band | 5 | 40% | -0.15 |
| 2025 | nan | 15 | 33% | +0.44 |
| 2026 | reversion_bounce | 143 | 36% | +0.16 |
| 2026 | uptrend_pullback_ema_band | 4 | 25% | -0.23 |
| 2026 | nan | 1 | 0% | -1.00 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected. This is a first backtest of a NEW bucket — treat with the same skepticism as any first result, not as validated the way trend_continuation/reversion_bounce are (those went through train/test + isolated-portfolio checks before being trusted)._