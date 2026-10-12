# Weak-RR handling — isolated A/B (RESEARCH)

- signal set: current merged screener (G1-G4), 66146 detected signals, weak-RR share 34%
- engine: portfolio_backtest.py conventions; pool ordered knife-tier then depth
- keep = all signals | drop = exclude weak_rr | refloor = tighten weak_rr stop to exactly min_rr:1

## Noise-aware comparison

_Noise-aware view (research/robust.py): every variant run on the full frozen set plus 10 paired random 90% ticker subsets. 'beats' = subsets where the variant beat the baseline on that SAME subset. Decide on avgR/PF and the beats counts; a single total return is mostly noise._

### full

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| keep | +101% | +71% [+23..+139] | -21% | +0.17 / +0.15 | 1.28 / 1.25 | 748 | 0/10 | 1/10 |
| drop | +134% | +94% [+52..+140] | -28% | +0.21 / +0.19 | 1.33 / 1.30 | 855 | baseline | baseline |
| refloor | +106% | +83% [+67..+223] | -25% | +0.21 / +0.19 | 1.33 / 1.30 | 854 | 6/10 | 5/10 |
| SPY | +83% | | | | | | | |

### 2021-2024

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| keep | +59% | +40% [+25..+86] | -20% | +0.21 / +0.15 | 1.34 / 1.24 | 461 | 1/10 | 2/10 |
| drop | +84% | +68% [+42..+91] | -27% | +0.26 / +0.23 | 1.41 / 1.37 | 514 | baseline | baseline |
| refloor | +87% | +52% [+33..+115] | -25% | +0.22 / +0.19 | 1.34 / 1.29 | 512 | 5/10 | 2/10 |
| SPY | +40% | | | | | | | |

### 2025-2026

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| keep | +26% | +13% [-7..+48] | -16% | +0.12 / +0.13 | 1.20 / 1.21 | 287 | 3/10 | 4/10 |
| drop | +28% | +17% [+6..+36] | -17% | +0.12 / +0.14 | 1.19 / 1.22 | 341 | baseline | baseline |
| refloor | +10% | +20% [+5..+49] | -16% | +0.20 / +0.18 | 1.32 / 1.29 | 342 | 6/10 | 8/10 |
| SPY | +31% | | | | | | | |

### 2022

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| keep | -3% | -6% [-15..+3] | -13% | -0.14 / -0.21 | 0.80 / 0.71 | 63 | 8/10 | 7/10 |
| drop | -14% | -15% [-21..-2] | -18% | -0.20 / -0.26 | 0.74 / 0.67 | 61 | baseline | baseline |
| refloor | -12% | -9% [-19..-3] | -16% | -0.20 / -0.16 | 0.74 / 0.78 | 67 | 8/10 | 8/10 |
| SPY | -20% | | | | | | | |

## full

| mode | ret | maxDD | Sharpe | trades | (weak) | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|---|
| keep | +101% | -20% | 0.91 | 748 | 179 | 38% | +0.17 | 1.28 | 14 |
| drop | +134% | -21% | 1.06 | 855 | 0 | 36% | +0.21 | 1.33 | 15 |
| refloor | +106% | -24% | 0.89 | 854 | 0 | 36% | +0.21 | 1.33 | 18 |
| SPY | +83% | | | | | | | | |

## 2021-2024

| mode | ret | maxDD | Sharpe | trades | (weak) | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|---|
| keep | +59% | -17% | 0.89 | 461 | 115 | 39% | +0.21 | 1.34 | 14 |
| drop | +84% | -21% | 1.13 | 514 | 0 | 37% | +0.26 | 1.41 | 16 |
| refloor | +87% | -24% | 1.13 | 512 | 0 | 36% | +0.22 | 1.34 | 18 |
| SPY | +40% | | | | | | | | |

## 2025-2026

| mode | ret | maxDD | Sharpe | trades | (weak) | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|---|
| keep | +26% | -18% | 0.92 | 287 | 64 | 36% | +0.12 | 1.20 | 13 |
| drop | +28% | -16% | 0.91 | 341 | 0 | 35% | +0.12 | 1.19 | 15 |
| refloor | +10% | -20% | 0.40 | 342 | 0 | 37% | +0.20 | 1.32 | 14 |
| SPY | +31% | | | | | | | | |

## 2022

| mode | ret | maxDD | Sharpe | trades | (weak) | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|---|
| keep | -3% | -12% | -0.21 | 63 | 12 | 29% | -0.14 | 0.80 | 9 |
| drop | -14% | -16% | -1.27 | 61 | 0 | 23% | -0.20 | 0.74 | 16 |
| refloor | -12% | -16% | -1.08 | 67 | 0 | 25% | -0.20 | 0.74 | 18 |
| SPY | -20% | | | | | | | | |

## Trade-level: weak-RR vs non-weak (sim 'keep' closed trades)

_2021-2024_
  weak_rr                n=   115 win=  44% avgR=-0.02 PF=0.96
  non-weak               n=   346 win=  37% avgR=+0.28 PF=1.46

_2025-2026_
  weak_rr                n=    64 win=  45% avgR=+0.09 PF=1.16
  non-weak               n=   223 win=  34% avgR=+0.13 PF=1.20

_2022_
  weak_rr                n=    12 win=  42% avgR=-0.06 PF=0.90
  non-weak               n=    51 win=  25% avgR=-0.16 PF=0.79
