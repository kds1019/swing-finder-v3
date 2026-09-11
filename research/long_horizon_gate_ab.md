# Long-horizon gate (252-session EMA200 slope) — isolated portfolio A/B (RESEARCH)

Layered ON TOP of the already-tested slope20>=-2% gate (research/current_trend_gate_ab.py), not instead of it. Motivated by ENPH (2026-09-11): 126-day slope +7.5% ("uptrend"), current 20-day slope -1.99% (just inside the -2% floor, still passes), but 252-day slope only +2.4% — a V-shaped recovery stalling at its own recent high, which neither existing check catches.
- base population: 27005 signals that already clear slope20>=-2%.
- engine: same as current_trend_gate_ab.py / g1_ab.py; pool ordered knife-tier then depth

- signal counts per variant: slope20>=-2% only (leading candidate)=27005, + slope252>=0%=25090, + slope252>=1%=24861, + slope252>=2%=24579, + slope252>=3%=24166

## full

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| slope20>=-2% only (leading candidate) | +122% | -40% | 0.57 | 858 | 38% | +0.23 | 1.37 | 14 |
| + slope252>=0% | +45% | -40% | 0.36 | 883 | 35% | +0.13 | 1.20 | 14 |
| + slope252>=1% | +50% | -40% | 0.38 | 889 | 34% | +0.11 | 1.17 | 14 |
| + slope252>=2% | +65% | -40% | 0.42 | 880 | 35% | +0.15 | 1.23 | 14 |
| + slope252>=3% | +71% | -40% | 0.44 | 866 | 36% | +0.17 | 1.26 | 14 |
| SPY | +83% | | | | | | | |

## 2021-2024

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| slope20>=-2% only (leading candidate) | +83% | -40% | 0.58 | 512 | 37% | +0.22 | 1.35 | 12 |
| + slope252>=0% | +40% | -40% | 0.41 | 533 | 34% | +0.12 | 1.19 | 14 |
| + slope252>=1% | +58% | -40% | 0.49 | 539 | 34% | +0.13 | 1.20 | 14 |
| + slope252>=2% | +72% | -40% | 0.54 | 539 | 36% | +0.17 | 1.26 | 14 |
| + slope252>=3% | +76% | -40% | 0.55 | 529 | 35% | +0.16 | 1.25 | 14 |
| SPY | +40% | | | | | | | |

## 2025-2026

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| slope20>=-2% only (leading candidate) | +21% | -14% | 0.78 | 346 | 38% | +0.24 | 1.40 | 14 |
| + slope252>=0% | +4% | -19% | 0.21 | 350 | 35% | +0.14 | 1.22 | 13 |
| + slope252>=1% | -5% | -18% | -0.10 | 350 | 33% | +0.09 | 1.13 | 14 |
| + slope252>=2% | -4% | -18% | -0.07 | 341 | 34% | +0.12 | 1.19 | 14 |
| + slope252>=3% | -3% | -18% | -0.04 | 337 | 36% | +0.18 | 1.29 | 14 |
| SPY | +31% | | | | | | | |

## 2022

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| slope20>=-2% only (leading candidate) | -9% | -13% | -0.79 | 59 | 31% | -0.04 | 0.95 | 10 |
| + slope252>=0% | -9% | -13% | -0.79 | 59 | 31% | -0.04 | 0.95 | 10 |
| + slope252>=1% | -9% | -13% | -0.79 | 59 | 31% | -0.04 | 0.95 | 10 |
| + slope252>=2% | -9% | -13% | -0.79 | 59 | 31% | -0.04 | 0.95 | 10 |
| + slope252>=3% | -9% | -13% | -0.79 | 59 | 31% | -0.04 | 0.95 | 10 |
| SPY | -20% | | | | | | | |
