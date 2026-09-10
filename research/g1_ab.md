# G1 (EMA200 slope) — isolated portfolio A/B (RESEARCH)

- signal set: G2[-20,3] + G3<=20 + G4<=-4 held, weak-RR dropped, G1 relaxed. 63903 signals total; G1>=5 covers 29799, >=3 37685, >=0 50515.
- engine: portfolio_backtest.py conventions; pool ordered knife-tier then depth

## full

| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | +89% | -40% | 0.48 | 935 | 36% | +0.19 | 1.30 | 16 |
| G1>=3 | +66% | -40% | 0.43 | 921 | 35% | +0.18 | 1.28 | 17 |
| G1>=0 | +10% | -45% | 0.23 | 915 | 32% | +0.07 | 1.10 | 19 |
| SPY | +83% | | | | | | | |

## 2021-2024

| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | +65% | -40% | 0.51 | 578 | 36% | +0.20 | 1.32 | 16 |
| G1>=3 | +30% | -40% | 0.37 | 555 | 34% | +0.18 | 1.27 | 18 |
| G1>=0 | +11% | -45% | 0.28 | 557 | 32% | +0.08 | 1.13 | 19 |
| SPY | +40% | | | | | | | |

## 2025-2026

| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | +14% | -20% | 0.47 | 357 | 35% | +0.17 | 1.27 | 13 |
| G1>=3 | +27% | -16% | 0.79 | 366 | 36% | +0.18 | 1.28 | 17 |
| G1>=0 | -1% | -20% | 0.05 | 358 | 33% | +0.05 | 1.07 | 15 |
| SPY | +31% | | | | | | | |

## 2022

| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | -14% | -15% | -1.19 | 85 | 22% | -0.25 | 0.68 | 17 |
| G1>=3 | -15% | -16% | -1.27 | 78 | 23% | -0.20 | 0.73 | 18 |
| G1>=0 | -22% | -23% | -1.84 | 57 | 18% | -0.39 | 0.52 | 16 |
| SPY | -20% | | | | | | | |
