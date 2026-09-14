# EMA-band pullback — stabilization-duration threshold sweep (RESEARCH)

Follow-up to research/ema_band_pullback_isolated_ab.py (validated at days_since_pullback_low >= 10). Sweeps specific day-count thresholds, each as an isolated backtest with its own dedicated capital (no competition for capacity) — checking whether a shorter minimum (catching candidates sooner) holds up nearly as well, or whether 10 days is doing real, load-bearing work.
- 1705 raw signals in the EMA-band zone (any duration), 341 tickers

- signal counts per threshold: no minimum=1705, >=5 days=1082, >=7 days=681, >=8 days=537, >=10 days (current)=318, >=12 days=208, >=15 days=122

## Full/train/test/2022 by threshold

### full

| threshold | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no minimum | +54% | -18% | 0.77 | 557 | 37% | +0.25 | 1.39 | 16 |
| >=5 days | +42% | -14% | 0.68 | 429 | 36% | +0.25 | 1.39 | 14 |
| >=7 days | +20% | -15% | 0.44 | 336 | 36% | +0.22 | 1.34 | 12 |
| >=8 days | +29% | -10% | 0.63 | 287 | 36% | +0.26 | 1.41 | 12 |
| >=10 days (current) | +15% | -9% | 0.45 | 175 | 37% | +0.25 | 1.39 | 9 |
| >=12 days | +19% | -5% | 0.76 | 117 | 38% | +0.33 | 1.53 | 7 |
| >=15 days | +15% | -5% | 0.74 | 75 | 40% | +0.32 | 1.53 | 6 |
| SPY | +83% | | | | | | | |

### 2021-2024

| threshold | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no minimum | +30% | -18% | 0.73 | 338 | 36% | +0.27 | 1.43 | 11 |
| >=5 days | +31% | -14% | 0.82 | 260 | 37% | +0.33 | 1.53 | 12 |
| >=7 days | +3% | -15% | 0.15 | 201 | 35% | +0.26 | 1.40 | 8 |
| >=8 days | +6% | -10% | 0.25 | 173 | 34% | +0.26 | 1.39 | 10 |
| >=10 days (current) | +5% | -6% | 0.26 | 108 | 36% | +0.32 | 1.50 | 6 |
| >=12 days | +14% | -5% | 0.85 | 77 | 39% | +0.51 | 1.84 | 5 |
| >=15 days | +5% | -5% | 0.39 | 48 | 38% | +0.29 | 1.46 | 5 |
| SPY | +40% | | | | | | | |

### 2025-2026

| threshold | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no minimum | +17% | -9% | 0.79 | 219 | 39% | +0.20 | 1.34 | 16 |
| >=5 days | +10% | -8% | 0.56 | 169 | 35% | +0.11 | 1.17 | 14 |
| >=7 days | +17% | -10% | 0.94 | 135 | 37% | +0.16 | 1.26 | 13 |
| >=8 days | +22% | -8% | 1.27 | 114 | 40% | +0.26 | 1.44 | 12 |
| >=10 days (current) | +9% | -5% | 0.81 | 67 | 37% | +0.13 | 1.21 | 9 |
| >=12 days | +4% | -5% | 0.58 | 40 | 35% | -0.01 | 0.98 | 7 |
| >=15 days | +9% | -2% | 1.48 | 27 | 44% | +0.38 | 1.69 | 6 |
| SPY | +31% | | | | | | | |

### 2022

| threshold | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no minimum | -7% | -9% | -1.27 | 38 | 26% | +0.02 | 1.02 | 7 |
| >=5 days | -7% | -8% | -1.40 | 33 | 33% | +0.38 | 1.57 | 7 |
| >=7 days | -9% | -9% | -1.89 | 29 | 28% | +0.08 | 1.11 | 8 |
| >=8 days | -6% | -6% | -1.38 | 23 | 35% | +0.36 | 1.55 | 5 |
| >=10 days (current) | -4% | -4% | -1.14 | 16 | 31% | +0.22 | 1.32 | 4 |
| >=12 days | -2% | -2% | -1.37 | 12 | 33% | +0.26 | 1.39 | 5 |
| >=15 days | -1% | -1% | -0.47 | 6 | 17% | -0.55 | 0.34 | 3 |
| SPY | -20% | | | | | | | |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected._