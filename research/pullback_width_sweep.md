# Pullback width-in-bars — finer sweep (RESEARCH)

Follow-up to research/pullback_shape_ab.py's tercile finding (long/grinding pullbacks underperformed in every window). This sweeps quintiles (finer shape of the relationship) and specific absolute width<=N thresholds (same style as research/current_trend_gate_ab.py's slope sweep) to find where the effect actually kicks in, so Decision Agent prompt guidance can cite a specific bar count.
- signal set: today's live gate held exactly as-is, 29945 signals with a computable width_bars

## Quintile shape (one portfolio run, today's actual gate, no additional filter)

| width_bars range | trades | win% | avgR | PF |
|---|---|---|---|---|
| (4.999, 8.0] | 202 | 38% | +0.29 | 1.46 |
| (8.0, 12.0] | 219 | 37% | +0.29 | 1.47 |
| (12.0, 16.0] | 163 | 35% | +0.00 | 1.00 |
| (16.0, 23.0] | 189 | 38% | +0.26 | 1.42 |
| (23.0, 59.0] | 165 | 33% | +0.19 | 1.29 |

- signal counts per variant: no gate (current)=29945, width<=10=10524, width<=15=17531, width<=20=22781, width<=25=26238, width<=30=28272

## Portfolio-level threshold sweep (require width_bars <= N)

### full

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +91% | -24% | 0.73 | 938 | 36% | +0.22 | 1.34 | 16 |
| width<=10 | +100% | -20% | 0.86 | 992 | 35% | +0.19 | 1.30 | 17 |
| width<=15 | +38% | -26% | 0.44 | 952 | 36% | +0.18 | 1.28 | 16 |
| width<=20 | +81% | -23% | 0.70 | 965 | 36% | +0.18 | 1.28 | 21 |
| width<=25 | +129% | -14% | 0.92 | 913 | 37% | +0.21 | 1.33 | 15 |
| width<=30 | +107% | -20% | 0.81 | 918 | 36% | +0.21 | 1.34 | 16 |
| SPY | +83% | | | | | | | |

### 2021-2024

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +66% | -24% | 0.85 | 574 | 36% | +0.21 | 1.33 | 17 |
| width<=10 | +51% | -20% | 0.80 | 624 | 35% | +0.22 | 1.33 | 19 |
| width<=15 | +46% | -16% | 0.72 | 598 | 36% | +0.20 | 1.31 | 16 |
| width<=20 | +50% | -23% | 0.70 | 596 | 34% | +0.12 | 1.18 | 21 |
| width<=25 | +71% | -14% | 0.90 | 550 | 37% | +0.18 | 1.28 | 15 |
| width<=30 | +60% | -20% | 0.79 | 577 | 36% | +0.23 | 1.36 | 16 |
| SPY | +40% | | | | | | | |

### 2025-2026

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +14% | -18% | 0.49 | 364 | 37% | +0.22 | 1.36 | 10 |
| width<=10 | +33% | -16% | 0.99 | 368 | 35% | +0.15 | 1.24 | 15 |
| width<=15 | -5% | -25% | -0.08 | 354 | 34% | +0.15 | 1.23 | 15 |
| width<=20 | +20% | -16% | 0.68 | 369 | 38% | +0.28 | 1.47 | 11 |
| width<=25 | +33% | -12% | 0.95 | 363 | 38% | +0.25 | 1.42 | 15 |
| width<=30 | +29% | -13% | 0.84 | 341 | 36% | +0.19 | 1.30 | 11 |
| SPY | +31% | | | | | | | |

### 2022

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | -14% | -15% | -1.19 | 85 | 22% | -0.25 | 0.68 | 17 |
| width<=10 | -14% | -18% | -1.58 | 66 | 26% | -0.19 | 0.74 | 17 |
| width<=15 | -6% | -13% | -0.59 | 58 | 26% | -0.04 | 0.94 | 16 |
| width<=20 | -12% | -16% | -1.13 | 76 | 22% | -0.24 | 0.69 | 21 |
| width<=25 | -7% | -12% | -0.62 | 73 | 23% | -0.20 | 0.74 | 14 |
| width<=30 | -12% | -16% | -0.95 | 78 | 23% | -0.17 | 0.78 | 16 |
| SPY | -20% | | | | | | | |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected._