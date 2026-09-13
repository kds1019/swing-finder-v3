# Current-trend gate (20-session EMA200 slope) — isolated portfolio A/B (RESEARCH)

Tests whether requiring the CURRENT 20-session EMA200 slope not be sharply negative, ON TOP OF the existing 126-session G1 gate, helps, hurts, or washes — motivated by a live finding (2026-09-11) that 24 of 26 real screener matches had G1 reading "uptrend" while the current slope had already rolled over.
- signal set: current live gate (G1>=5, G2[-20,3], G3<=20, G4<=-4, weak-RR dropped) held exactly as-is; 29945 signals total.
- engine: portfolio_backtest.py / g1_ab.py conventions; pool ordered knife-tier then depth

- signal counts per variant: no gate (current)=29945, slope>=-2%=27005, slope>=-0.5% (trend_context flat-band)=17937, slope>=0%=12821, slope>=2%=612

## full

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +91% | -24% | 0.73 | 938 | 36% | +0.22 | 1.34 | 16 |
| slope>=-2% | +146% | -19% | 1.10 | 868 | 38% | +0.23 | 1.38 | 16 |
| slope>=-0.5% (trend_context flat-band) | +49% | -25% | 0.59 | 826 | 38% | +0.23 | 1.38 | 15 |
| slope>=0% | +45% | -31% | 0.57 | 942 | 37% | +0.26 | 1.42 | 22 |
| slope>=2% | +13% | -11% | 0.34 | 299 | 30% | +0.13 | 1.19 | 14 |
| SPY | +83% | | | | | | | |

## 2021-2024

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +66% | -24% | 0.85 | 574 | 36% | +0.21 | 1.33 | 17 |
| slope>=-2% | +103% | -19% | 1.26 | 517 | 38% | +0.24 | 1.39 | 14 |
| slope>=-0.5% (trend_context flat-band) | +29% | -25% | 0.58 | 528 | 37% | +0.22 | 1.34 | 14 |
| slope>=0% | +25% | -31% | 0.53 | 615 | 35% | +0.22 | 1.35 | 25 |
| slope>=2% | +8% | -9% | 0.34 | 157 | 29% | +0.05 | 1.07 | 14 |
| SPY | +40% | | | | | | | |

## 2025-2026

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +14% | -18% | 0.49 | 364 | 37% | +0.22 | 1.36 | 10 |
| slope>=-2% | +21% | -15% | 0.77 | 351 | 38% | +0.22 | 1.36 | 14 |
| slope>=-0.5% (trend_context flat-band) | +15% | -13% | 0.61 | 298 | 40% | +0.26 | 1.44 | 15 |
| slope>=0% | +16% | -12% | 0.64 | 327 | 40% | +0.34 | 1.57 | 10 |
| slope>=2% | +7% | -8% | 0.46 | 142 | 31% | +0.22 | 1.32 | 10 |
| SPY | +31% | | | | | | | |

## 2022

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | -14% | -15% | -1.19 | 85 | 22% | -0.25 | 0.68 | 17 |
| slope>=-2% | -9% | -14% | -0.82 | 59 | 27% | -0.18 | 0.76 | 16 |
| slope>=-0.5% (trend_context flat-band) | -16% | -17% | -2.00 | 58 | 22% | -0.32 | 0.59 | 10 |
| slope>=0% | -16% | -16% | -1.82 | 68 | 16% | -0.45 | 0.46 | 25 |
| slope>=2% | -3% | -4% | -1.77 | 7 | 0% | -1.00 | 0.00 | 7 |
| SPY | -20% | | | | | | | |
