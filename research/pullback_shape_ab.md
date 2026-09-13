# Pullback shape — depth regime + width-in-bars (RESEARCH)

Tests whether depth (price_vs_ema200_pct) has a nonlinear/sweet-spot relationship with outcome rather than the 'deeper is better' read implied by using it as a plain sort key, and whether width-in-bars (bars since the pullback's actual peak, via the same swing-high detection used for the Fibonacci-zone read — NOT days_since_pullback_low, which only counts forward from the low) predicts anything on its own. Neither is currently used this way in the live gate.
- signal set: today's live gate held exactly as-is, 29945 signals with a computable width_bars (0 dropped for insufficient history)
- depth terciles: [Interval(-19.998, -4.543, closed='right'), Interval(-4.543, -0.281, closed='right'), Interval(-0.281, 3.005, closed='right')]
- width terciles (bars since peak): [Interval(4.999, 10.0, closed='right'), Interval(10.0, 18.0, closed='right'), Interval(18.0, 59.0, closed='right')]

## View 1 — realized trade outcomes (one portfolio run, today's actual gate, no additional filter)

### by depth tercile

| depth_bucket | trades | win% | avgR | medianR | PF |
|---|---|---|---|---|---|
| deep | 627 | 36% | +0.21 | -1.00 | 1.33 |
| moderate | 219 | 36% | +0.26 | -1.00 | 1.42 |
| shallow | 92 | 41% | +0.16 | -1.00 | 1.27 |

### by width tercile

| width_bucket | trades | win% | avgR | medianR | PF |
|---|---|---|---|---|---|
| long | 285 | 34% | +0.19 | -1.00 | 1.30 |
| medium | 342 | 38% | +0.19 | -1.00 | 1.31 |
| short | 311 | 37% | +0.26 | -1.00 | 1.42 |

### joint depth x width (trades / win% / avgR)

| depth_bucket | width_bucket | trades | win% | avgR |
|---|---|---|---|---|
| deep | long | 192 | 34% | +0.18 |
| deep | medium | 211 | 37% | +0.15 |
| deep | short | 224 | 37% | +0.29 |
| moderate | long | 72 | 33% | +0.28 |
| moderate | medium | 88 | 38% | +0.23 |
| moderate | short | 59 | 36% | +0.28 |
| shallow | long | 21 | 38% | -0.01 |
| shallow | medium | 43 | 44% | +0.32 |
| shallow | short | 28 | 39% | +0.04 |

- signal counts per variant: no gate (current)=29945, depth=deep only=9982, depth=moderate only=9981, depth=shallow only=9982, width=short only=10524, width=medium only=10423, width=long only=8998

## View 2 — portfolio-level restriction variants (does keeping only one tercile change the blended portfolio result?)

### full

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +91% | -24% | 0.73 | 938 | 36% | +0.22 | 1.34 | 16 |
| depth=deep only | +54% | -32% | 0.53 | 1104 | 35% | +0.19 | 1.30 | 19 |
| depth=moderate only | +143% | -34% | 1.08 | 890 | 36% | +0.23 | 1.36 | 19 |
| depth=shallow only | +104% | -29% | 0.93 | 838 | 37% | +0.23 | 1.38 | 20 |
| width=short only | +100% | -20% | 0.86 | 992 | 35% | +0.19 | 1.30 | 17 |
| width=medium only | +64% | -23% | 0.64 | 845 | 37% | +0.20 | 1.32 | 19 |
| width=long only | +6% | -33% | 0.14 | 838 | 33% | +0.14 | 1.21 | 20 |
| SPY | +83% | | | | | | | |

### 2021-2024

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +66% | -24% | 0.85 | 574 | 36% | +0.21 | 1.33 | 17 |
| depth=deep only | +23% | -32% | 0.42 | 673 | 35% | +0.17 | 1.27 | 19 |
| depth=moderate only | +77% | -34% | 1.02 | 544 | 36% | +0.22 | 1.34 | 19 |
| depth=shallow only | +30% | -29% | 0.56 | 540 | 36% | +0.19 | 1.30 | 20 |
| width=short only | +51% | -20% | 0.80 | 624 | 35% | +0.22 | 1.33 | 19 |
| width=medium only | +68% | -23% | 0.94 | 537 | 38% | +0.26 | 1.41 | 18 |
| width=long only | +35% | -33% | 0.60 | 499 | 35% | +0.20 | 1.31 | 14 |
| SPY | +40% | | | | | | | |

### 2025-2026

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +14% | -18% | 0.49 | 364 | 37% | +0.22 | 1.36 | 10 |
| depth=deep only | +24% | -24% | 0.70 | 431 | 35% | +0.23 | 1.36 | 18 |
| depth=moderate only | +38% | -13% | 1.21 | 346 | 37% | +0.24 | 1.39 | 11 |
| depth=shallow only | +56% | -14% | 1.59 | 298 | 40% | +0.31 | 1.53 | 11 |
| width=short only | +33% | -16% | 0.99 | 368 | 35% | +0.15 | 1.24 | 15 |
| width=medium only | -3% | -20% | -0.01 | 308 | 36% | +0.10 | 1.16 | 19 |
| width=long only | -22% | -31% | -0.80 | 339 | 30% | +0.05 | 1.07 | 19 |
| SPY | +31% | | | | | | | |

### 2022

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | -14% | -15% | -1.19 | 85 | 22% | -0.25 | 0.68 | 17 |
| depth=deep only | -18% | -21% | -1.49 | 79 | 22% | -0.32 | 0.60 | 20 |
| depth=moderate only | -23% | -23% | -2.48 | 58 | 10% | -0.68 | 0.23 | 19 |
| depth=shallow only | -6% | -10% | -0.76 | 55 | 31% | -0.03 | 0.96 | 9 |
| width=short only | -14% | -18% | -1.58 | 66 | 26% | -0.19 | 0.74 | 17 |
| width=medium only | -6% | -12% | -0.57 | 63 | 30% | +0.02 | 1.03 | 7 |
| width=long only | -15% | -17% | -1.47 | 54 | 22% | -0.23 | 0.70 | 14 |
| SPY | -20% | | | | | | | |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected. Terciles are of THIS signal set's own realized distribution, not fixed a-priori thresholds — see the edges reported above._