# Current-trend gate (20-session EMA200 slope) — isolated portfolio A/B (RESEARCH)

Tests whether requiring the CURRENT 20-session EMA200 slope not be sharply negative, ON TOP OF the existing 126-session G1 gate, helps, hurts, or washes — motivated by a live finding (2026-09-11) that 24 of 26 real screener matches had G1 reading "uptrend" while the current slope had already rolled over.
- signal set: current live gate (G1>=5, G2[-20,3], G3<=20, G4<=-4, weak-RR dropped) held exactly as-is; 29945 signals total.
- engine: portfolio_backtest.py / g1_ab.py conventions; pool ordered knife-tier then depth

- signal counts per variant: no gate (current)=29945, slope>=-2%=27005, slope>=-0.5% (trend_context flat-band)=17937, slope>=0%=12821, slope>=2%=612

## full

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +89% | -40% | 0.48 | 937 | 36% | +0.22 | 1.34 | 16 |
| slope>=-2% | +122% | -40% | 0.57 | 858 | 38% | +0.23 | 1.37 | 14 |
| slope>=-0.5% (trend_context flat-band) | +57% | -32% | 0.47 | 831 | 38% | +0.22 | 1.36 | 14 |
| slope>=0% | +45% | -50% | 0.36 | 941 | 37% | +0.26 | 1.42 | 22 |
| slope>=2% | +13% | -11% | 0.34 | 299 | 30% | +0.13 | 1.19 | 14 |
| SPY | +83% | | | | | | | |

## 2021-2024

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +66% | -40% | 0.51 | 574 | 36% | +0.21 | 1.33 | 17 |
| slope>=-2% | +83% | -40% | 0.58 | 512 | 37% | +0.22 | 1.35 | 12 |
| slope>=-0.5% (trend_context flat-band) | +29% | -32% | 0.39 | 533 | 37% | +0.20 | 1.32 | 16 |
| slope>=0% | +25% | -50% | 0.37 | 615 | 35% | +0.22 | 1.35 | 25 |
| slope>=2% | +8% | -9% | 0.34 | 157 | 29% | +0.05 | 1.07 | 14 |
| SPY | +40% | | | | | | | |

## 2025-2026

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +13% | -18% | 0.47 | 363 | 37% | +0.23 | 1.37 | 12 |
| slope>=-2% | +21% | -14% | 0.78 | 346 | 38% | +0.24 | 1.40 | 14 |
| slope>=-0.5% (trend_context flat-band) | +22% | -13% | 0.83 | 298 | 39% | +0.26 | 1.43 | 13 |
| slope>=0% | +16% | -12% | 0.65 | 326 | 40% | +0.34 | 1.58 | 10 |
| slope>=2% | +7% | -8% | 0.46 | 142 | 31% | +0.22 | 1.32 | 10 |
| SPY | +31% | | | | | | | |

## 2022

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | -14% | -15% | -1.19 | 85 | 22% | -0.25 | 0.68 | 17 |
| slope>=-2% | -9% | -13% | -0.79 | 59 | 31% | -0.04 | 0.95 | 10 |
| slope>=-0.5% (trend_context flat-band) | -15% | -30% | -0.31 | 57 | 21% | -0.33 | 0.58 | 9 |
| slope>=0% | -16% | -46% | 0.11 | 68 | 16% | -0.45 | 0.46 | 25 |
| slope>=2% | -3% | -4% | -1.77 | 7 | 0% | -1.00 | 0.00 | 7 |
| SPY | -20% | | | | | | | |
