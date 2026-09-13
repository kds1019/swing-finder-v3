# G1 (EMA200 slope) — isolated portfolio A/B (RESEARCH)

- signal set: G2[-20,3] + G3<=20 + G4<=-4 held, weak-RR dropped, G1 relaxed. 63903 signals total; G1>=5 covers 29799, >=3 37685, >=0 50515.
- engine: portfolio_backtest.py conventions; pool ordered knife-tier then depth

## full

| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | +91% | -24% | 0.73 | 936 | 35% | +0.19 | 1.30 | 16 |
| G1>=3 | +68% | -28% | 0.61 | 924 | 35% | +0.18 | 1.27 | 17 |
| G1>=0 | +5% | -45% | 0.14 | 927 | 32% | +0.07 | 1.10 | 19 |
| SPY | +83% | | | | | | | |

## 2021-2024

| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | +65% | -24% | 0.84 | 578 | 36% | +0.20 | 1.32 | 16 |
| G1>=3 | +30% | -28% | 0.49 | 555 | 34% | +0.18 | 1.27 | 18 |
| G1>=0 | +5% | -45% | 0.16 | 559 | 32% | +0.08 | 1.12 | 19 |
| SPY | +40% | | | | | | | |

## 2025-2026

| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | +15% | -20% | 0.50 | 358 | 35% | +0.17 | 1.27 | 13 |
| G1>=3 | +28% | -16% | 0.82 | 369 | 36% | +0.17 | 1.27 | 16 |
| G1>=0 | -0% | -20% | 0.08 | 368 | 33% | +0.04 | 1.06 | 19 |
| SPY | +31% | | | | | | | |

## 2022

| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | -14% | -15% | -1.19 | 85 | 22% | -0.25 | 0.68 | 17 |
| G1>=3 | -15% | -16% | -1.27 | 78 | 23% | -0.20 | 0.73 | 18 |
| G1>=0 | -22% | -23% | -1.84 | 58 | 19% | -0.34 | 0.58 | 11 |
| SPY | -20% | | | | | | | |
