# G3 (15-day consolidation range) — isolated portfolio A/B (RESEARCH)

- signal set: G1>=5 + G2[-20,3] + G4<=-4 held, weak-RR dropped, G3 removed. 55490 signals total; G3<=20 covers 46419, <=30 52532, <=45 54763, off 55490.
- engine: portfolio_backtest.py conventions; pool ordered knife-tier then depth

## Noise-aware comparison

_Noise-aware view (research/robust.py): every variant run on the full frozen set plus 10 paired random 90% ticker subsets. 'beats' = subsets where the variant beat the baseline on that SAME subset. Decide on avgR/PF and the beats counts; a single total return is mostly noise._

### full

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| G3<=20 (current) | +96% | +81% [+59..+102] | -28% | +0.22 / +0.22 | 1.34 / 1.35 | 931 | baseline | baseline |
| G3<=30 | +75% | +76% [+40..+111] | -30% | +0.22 / +0.19 | 1.36 / 1.30 | 937 | 4/10 | 3/10 |
| G3<=45 | +100% | +89% [+43..+128] | -28% | +0.18 / +0.18 | 1.29 / 1.29 | 936 | 6/10 | 2/10 |
| G3 off | +67% | +94% [+43..+123] | -26% | +0.19 / +0.20 | 1.30 / 1.31 | 934 | 5/10 | 2/10 |
| SPY | +83% | | | | | | | |

### 2021-2024

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| G3<=20 (current) | +61% | +60% [+34..+74] | -28% | +0.22 / +0.23 | 1.34 / 1.36 | 570 | baseline | baseline |
| G3<=30 | +46% | +42% [+14..+73] | -30% | +0.23 / +0.21 | 1.36 / 1.33 | 577 | 4/10 | 5/10 |
| G3<=45 | +52% | +40% [+16..+74] | -28% | +0.18 / +0.18 | 1.29 / 1.29 | 577 | 4/10 | 4/10 |
| G3 off | +42% | +52% [+16..+72] | -26% | +0.18 / +0.21 | 1.28 / 1.33 | 575 | 3/10 | 3/10 |
| SPY | +40% | | | | | | | |

### 2025-2026

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| G3<=20 (current) | +22% | +16% [-2..+19] | -19% | +0.22 / +0.21 | 1.35 / 1.33 | 361 | baseline | baseline |
| G3<=30 | +19% | +17% [+3..+43] | -18% | +0.22 / +0.16 | 1.35 / 1.24 | 360 | 6/10 | 1/10 |
| G3<=45 | +31% | +28% [+17..+44] | -16% | +0.18 / +0.17 | 1.29 / 1.27 | 359 | 10/10 | 2/10 |
| G3 off | +17% | +27% [+11..+45] | -17% | +0.20 / +0.17 | 1.32 / 1.27 | 359 | 9/10 | 2/10 |
| SPY | +31% | | | | | | | |

### 2022

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| G3<=20 (current) | -18% | -19% [-22..-14] | -19% | -0.34 / -0.33 | 0.58 / 0.59 | 80 | baseline | baseline |
| G3<=30 | -23% | -20% [-26..-17] | -21% | -0.36 / -0.23 | 0.56 / 0.70 | 78 | 2/10 | 9/10 |
| G3<=45 | -20% | -21% [-23..-15] | -21% | -0.27 / -0.18 | 0.66 / 0.77 | 78 | 2/10 | 9/10 |
| G3 off | -19% | -19% [-23..-15] | -19% | -0.25 / -0.19 | 0.68 / 0.76 | 79 | 3/10 | 9/10 |
| SPY | -20% | | | | | | | |

## full

| G3 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G3<=20 (current) | +96% | -27% | 0.77 | 931 | 36% | +0.22 | 1.34 | 15 |
| G3<=30 | +75% | -27% | 0.63 | 937 | 36% | +0.22 | 1.36 | 14 |
| G3<=45 | +100% | -26% | 0.76 | 936 | 36% | +0.18 | 1.29 | 14 |
| G3 off | +67% | -26% | 0.59 | 934 | 36% | +0.19 | 1.30 | 14 |
| SPY | +83% | | | | | | | |

## 2021-2024

| G3 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G3<=20 (current) | +61% | -27% | 0.80 | 570 | 36% | +0.22 | 1.34 | 15 |
| G3<=30 | +46% | -27% | 0.64 | 577 | 36% | +0.23 | 1.36 | 14 |
| G3<=45 | +52% | -26% | 0.68 | 577 | 36% | +0.18 | 1.29 | 14 |
| G3 off | +42% | -26% | 0.59 | 575 | 35% | +0.18 | 1.28 | 14 |
| SPY | +40% | | | | | | | |

## 2025-2026

| G3 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G3<=20 (current) | +22% | -14% | 0.70 | 361 | 37% | +0.22 | 1.35 | 12 |
| G3<=30 | +19% | -22% | 0.61 | 360 | 36% | +0.22 | 1.35 | 11 |
| G3<=45 | +31% | -15% | 0.92 | 359 | 36% | +0.18 | 1.29 | 11 |
| G3 off | +17% | -18% | 0.57 | 359 | 36% | +0.20 | 1.32 | 11 |
| SPY | +31% | | | | | | | |

## 2022

| G3 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| G3<=20 (current) | -18% | -18% | -1.53 | 80 | 20% | -0.34 | 0.58 | 15 |
| G3<=30 | -23% | -23% | -1.89 | 78 | 18% | -0.36 | 0.56 | 14 |
| G3<=45 | -20% | -20% | -1.60 | 78 | 21% | -0.27 | 0.66 | 13 |
| G3 off | -19% | -20% | -1.59 | 79 | 22% | -0.25 | 0.68 | 13 |
| SPY | -20% | | | | | | | |
