# G1 (EMA200 slope) — isolated portfolio A/B (RESEARCH)

- signal set: G2[-20,3] + G3<=20 + G4<=-4 held, weak-RR dropped, G1 relaxed. 78623 signals total; G1>=5 covers 46437, >=3 54337, >=0 66270.
- engine: portfolio_backtest.py conventions; pool ordered knife-tier then depth

## Noise-aware comparison

_Noise-aware view (research/robust.py): every variant run on the full frozen set plus 10 paired random 90% ticker subsets. 'beats' = subsets where the variant beat the baseline on that SAME subset. Decide on avgR/PF and the beats counts; a single total return is mostly noise._

### full

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | +96% | +82% [+53..+106] | -26% | +0.22 / +0.21 | 1.34 / 1.34 | 931 | baseline | baseline |
| G1>=3 | +55% | +51% [+27..+112] | -27% | +0.14 / +0.16 | 1.22 / 1.25 | 900 | 2/10 | 1/10 |
| G1>=0 | -21% | -16% [-30..+3] | -44% | +0.07 / +0.08 | 1.10 / 1.12 | 924 | 0/10 | 0/10 |
| SPY | +83% | | | | | | | |

### 2021-2024

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | +61% | +64% [+47..+88] | -26% | +0.22 / +0.23 | 1.34 / 1.36 | 570 | baseline | baseline |
| G1>=3 | +56% | +46% [+26..+70] | -26% | +0.18 / +0.18 | 1.28 / 1.27 | 554 | 2/10 | 1/10 |
| G1>=0 | -14% | -5% [-17..+8] | -44% | +0.08 / +0.07 | 1.12 / 1.10 | 578 | 0/10 | 0/10 |
| SPY | +40% | | | | | | | |

### 2025-2026

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | +22% | +3% [+1..+20] | -19% | +0.22 / +0.17 | 1.35 / 1.27 | 361 | baseline | baseline |
| G1>=3 | -1% | +6% [-15..+31] | -23% | +0.08 / +0.12 | 1.12 / 1.19 | 346 | 4/10 | 1/10 |
| G1>=0 | -9% | -13% [-22..+6] | -23% | +0.05 / +0.07 | 1.07 / 1.11 | 346 | 1/10 | 1/10 |
| SPY | +31% | | | | | | | |

### 2022

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | -18% | -17% [-22..-14] | -18% | -0.34 / -0.27 | 0.58 / 0.65 | 80 | baseline | baseline |
| G1>=3 | -20% | -19% [-21..-15] | -19% | -0.30 / -0.31 | 0.63 / 0.62 | 83 | 5/10 | 4/10 |
| G1>=0 | -25% | -24% [-27..-22] | -24% | -0.43 / -0.49 | 0.49 / 0.42 | 64 | 0/10 | 0/10 |
| SPY | -20% | | | | | | | |

## full

| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | +96% | -27% | 0.77 | 931 | 36% | +0.22 | 1.34 | 15 |
| G1>=3 | +55% | -27% | 0.55 | 900 | 35% | +0.14 | 1.22 | 16 |
| G1>=0 | -21% | -51% | -0.14 | 924 | 33% | +0.07 | 1.10 | 19 |
| SPY | +83% | | | | | | | |

## 2021-2024

| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | +61% | -27% | 0.80 | 570 | 36% | +0.22 | 1.34 | 15 |
| G1>=3 | +56% | -27% | 0.77 | 554 | 35% | +0.18 | 1.28 | 15 |
| G1>=0 | -14% | -51% | -0.13 | 578 | 33% | +0.08 | 1.12 | 20 |
| SPY | +40% | | | | | | | |

## 2025-2026

| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | +22% | -14% | 0.70 | 361 | 37% | +0.22 | 1.35 | 12 |
| G1>=3 | -1% | -26% | 0.07 | 346 | 34% | +0.08 | 1.12 | 12 |
| G1>=0 | -9% | -26% | -0.18 | 346 | 34% | +0.05 | 1.07 | 15 |
| SPY | +31% | | | | | | | |

## 2022

| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G1>=5 (current) | -18% | -18% | -1.53 | 80 | 20% | -0.34 | 0.58 | 15 |
| G1>=3 | -20% | -20% | -1.69 | 83 | 19% | -0.30 | 0.63 | 15 |
| G1>=0 | -25% | -25% | -2.08 | 64 | 16% | -0.43 | 0.49 | 14 |
| SPY | -20% | | | | | | | |
