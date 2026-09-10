# Weak-RR handling — isolated A/B (RESEARCH)

- signal set: current merged screener (G1-G4), 43848 detected signals, weak-RR share 32%
- engine: portfolio_backtest.py conventions; pool ordered knife-tier then depth
- keep = all signals | drop = exclude weak_rr | refloor = tighten weak_rr stop to exactly min_rr:1

## full

| mode | ret | maxDD | Sharpe | trades | (weak) | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|---|
| keep | +44% | -58% | 0.36 | 833 | 165 | 37% | +0.13 | 1.21 | 14 |
| drop | +89% | -40% | 0.48 | 935 | 0 | 36% | +0.19 | 1.30 | 16 |
| refloor | +93% | -58% | 0.44 | 955 | 0 | 35% | +0.16 | 1.25 | 14 |
| SPY | +83% | | | | | | | | |

## 2021-2024

| mode | ret | maxDD | Sharpe | trades | (weak) | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|---|
| keep | +35% | -58% | 0.41 | 537 | 117 | 37% | +0.13 | 1.21 | 13 |
| drop | +65% | -40% | 0.51 | 578 | 0 | 36% | +0.20 | 1.32 | 16 |
| refloor | +73% | -58% | 0.49 | 580 | 0 | 37% | +0.21 | 1.34 | 14 |
| SPY | +40% | | | | | | | | |

## 2025-2026

| mode | ret | maxDD | Sharpe | trades | (weak) | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|---|
| keep | +4% | -18% | 0.22 | 296 | 48 | 38% | +0.13 | 1.22 | 14 |
| drop | +14% | -20% | 0.47 | 357 | 0 | 35% | +0.17 | 1.27 | 13 |
| refloor | +9% | -18% | 0.37 | 375 | 0 | 33% | +0.08 | 1.13 | 15 |
| SPY | +31% | | | | | | | | |

## 2022

| mode | ret | maxDD | Sharpe | trades | (weak) | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|---|
| keep | -19% | -19% | -1.53 | 70 | 12 | 21% | -0.31 | 0.60 | 13 |
| drop | -14% | -15% | -1.19 | 85 | 0 | 22% | -0.25 | 0.68 | 17 |
| refloor | -18% | -18% | -1.60 | 76 | 0 | 22% | -0.22 | 0.71 | 12 |
| SPY | -20% | | | | | | | | |

## Trade-level: weak-RR vs non-weak (sim 'keep' closed trades)

_2021-2024_
  weak_rr                n=   117 win=  42% avgR=-0.06 PF=0.89
  non-weak               n=   420 win=  35% avgR=+0.19 PF=1.29

_2025-2026_
  weak_rr                n=    48 win=  42% avgR=-0.09 PF=0.84
  non-weak               n=   248 win=  37% avgR=+0.18 PF=1.29

_2022_
  weak_rr                n=    12 win=  33% avgR=-0.14 PF=0.79
  non-weak               n=    58 win=  19% avgR=-0.35 PF=0.57
