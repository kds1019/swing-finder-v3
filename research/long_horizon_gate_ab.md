# Long-horizon gate (252-session EMA200 slope) — isolated portfolio A/B (RESEARCH)

Layered ON TOP of the already-tested slope20>=-2% gate (research/current_trend_gate_ab.py), not instead of it. Motivated by ENPH (2026-09-11): 126-day slope +7.5% ("uptrend"), current 20-day slope -1.99% (just inside the -2% floor, still passes), but 252-day slope only +2.4% — a V-shaped recovery stalling at its own recent high, which neither existing check catches.
- base population: 27005 signals that already clear slope20>=-2%.
- engine: same as current_trend_gate_ab.py / g1_ab.py; pool ordered knife-tier then depth

- signal counts per variant: slope20>=-2% only (leading candidate)=27005, + slope252>=0%=25090, + slope252>=1%=24861, + slope252>=2%=24579, + slope252>=3%=24166

## full

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| slope20>=-2% only (leading candidate) | +146% | -19% | 1.10 | 868 | 38% | +0.23 | 1.38 | 16 |
| + slope252>=0% | +46% | -23% | 0.53 | 879 | 34% | +0.12 | 1.18 | 16 |
| + slope252>=1% | +50% | -22% | 0.56 | 888 | 35% | +0.14 | 1.22 | 14 |
| + slope252>=2% | +65% | -23% | 0.66 | 876 | 34% | +0.13 | 1.20 | 16 |
| + slope252>=3% | +84% | -20% | 0.79 | 859 | 36% | +0.18 | 1.28 | 16 |
| SPY | +83% | | | | | | | |

## 2021-2024

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| slope20>=-2% only (leading candidate) | +103% | -19% | 1.26 | 517 | 38% | +0.24 | 1.39 | 14 |
| + slope252>=0% | +39% | -22% | 0.65 | 528 | 33% | +0.11 | 1.16 | 14 |
| + slope252>=1% | +49% | -19% | 0.77 | 544 | 35% | +0.15 | 1.23 | 14 |
| + slope252>=2% | +60% | -19% | 0.88 | 537 | 34% | +0.14 | 1.21 | 14 |
| + slope252>=3% | +79% | -18% | 1.07 | 538 | 36% | +0.17 | 1.27 | 16 |
| SPY | +40% | | | | | | | |

## 2025-2026

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| slope20>=-2% only (leading candidate) | +21% | -15% | 0.77 | 351 | 38% | +0.22 | 1.36 | 14 |
| + slope252>=0% | +6% | -19% | 0.28 | 351 | 35% | +0.13 | 1.20 | 14 |
| + slope252>=1% | +1% | -18% | 0.13 | 344 | 35% | +0.13 | 1.20 | 14 |
| + slope252>=2% | +3% | -18% | 0.18 | 339 | 35% | +0.13 | 1.20 | 14 |
| + slope252>=3% | +3% | -15% | 0.20 | 321 | 37% | +0.19 | 1.31 | 13 |
| SPY | +31% | | | | | | | |

## 2022

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| slope20>=-2% only (leading candidate) | -9% | -14% | -0.82 | 59 | 27% | -0.18 | 0.76 | 16 |
| + slope252>=0% | -9% | -14% | -0.82 | 59 | 27% | -0.18 | 0.76 | 16 |
| + slope252>=1% | -9% | -14% | -0.82 | 59 | 27% | -0.18 | 0.76 | 16 |
| + slope252>=2% | -9% | -14% | -0.82 | 59 | 27% | -0.18 | 0.76 | 16 |
| + slope252>=3% | -9% | -14% | -0.82 | 59 | 27% | -0.18 | 0.76 | 16 |
| SPY | -20% | | | | | | | |
