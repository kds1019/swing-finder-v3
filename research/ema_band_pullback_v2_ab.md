# EMA-band pullback v2 — does requiring 10+ days of stabilization fix the stop-out rate? (RESEARCH)

Follow-up to research/ema_band_pullback_ab.py, which found the EMA-band bucket (price <= 50-EMA, no worse than -20% vs 200-EMA, uptrend, stabilising) a net loser (PF 0.95) because 68% of trades were stopped out directly, NOT because the reward side was bad (winners averaged +1.5R to +3.7R). Tests whether requiring the stabilization to have HELD for >= 10 days (core.pullback_reversal.measure_stabilization's days_since_pullback_low, already computed live but not gated on) fixes this, by splitting the same zone into a >=10-day sub-bucket and a <10-day ('fresh') sub-bucket.
- signal set: today's live gate held exactly as-is, 29945 signals
- bucket counts: {'reversion_bounce': 16254, nan: 11505, 'uptrend_pullback_ema_band_fresh': 1387, 'trend_continuation': 481, 'uptrend_pullback_ema_band_10d': 318}
- trade management: DEFAULT for every bucket (no sizing/exit overrides)

## Per-bucket trade stats (one portfolio run, default trade management)

| bucket | trades | win% | avgR | medianR | PF | % stopped out |
|---|---|---|---|---|---|---|
| reversion_bounce | 852 | 37% | +0.23 | -1.00 | 1.36 | 62% |
| trend_continuation | 6 | 67% | +1.02 | +1.27 | 4.07 | 33% |
| uptrend_pullback_ema_band_10d | 11 | 36% | +0.20 | -1.00 | 1.31 | 64% |
| uptrend_pullback_ema_band_fresh | 20 | 30% | -0.16 | -1.00 | 0.77 | 70% |
| nan | 49 | 31% | +0.06 | -1.00 | 1.08 | 69% |

## By year and bucket

| year | bucket | trades | win% | avgR |
|---|---|---|---|---|
| 2021 | reversion_bounce | 58 | 41% | +0.28 |
| 2021 | uptrend_pullback_ema_band_10d | 2 | 50% | +0.18 |
| 2021 | nan | 5 | 40% | +0.44 |
| 2022 | reversion_bounce | 77 | 22% | -0.24 |
| 2022 | uptrend_pullback_ema_band_10d | 1 | 0% | -1.00 |
| 2022 | uptrend_pullback_ema_band_fresh | 1 | 100% | +3.18 |
| 2022 | nan | 6 | 17% | -0.80 |
| 2023 | reversion_bounce | 186 | 37% | +0.25 |
| 2023 | trend_continuation | 1 | 100% | +1.39 |
| 2023 | uptrend_pullback_ema_band_10d | 3 | 33% | +0.56 |
| 2023 | uptrend_pullback_ema_band_fresh | 4 | 0% | -1.00 |
| 2023 | nan | 11 | 9% | -0.78 |
| 2024 | reversion_bounce | 193 | 40% | +0.39 |
| 2024 | trend_continuation | 4 | 50% | +0.39 |
| 2024 | uptrend_pullback_ema_band_10d | 4 | 50% | +0.53 |
| 2024 | uptrend_pullback_ema_band_fresh | 7 | 29% | -0.25 |
| 2024 | nan | 11 | 55% | +0.76 |
| 2025 | reversion_bounce | 195 | 38% | +0.26 |
| 2025 | trend_continuation | 1 | 100% | +3.21 |
| 2025 | uptrend_pullback_ema_band_10d | 1 | 0% | -1.00 |
| 2025 | uptrend_pullback_ema_band_fresh | 4 | 50% | +0.06 |
| 2025 | nan | 15 | 33% | +0.44 |
| 2026 | reversion_bounce | 143 | 36% | +0.16 |
| 2026 | uptrend_pullback_ema_band_fresh | 4 | 25% | -0.23 |
| 2026 | nan | 1 | 0% | -1.00 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected. First backtest of the 10-day split — treat with the same skepticism as any first result._