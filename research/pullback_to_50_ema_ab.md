# Pullback-to-50-EMA (holding the 200-EMA) — standalone isolated backtest (RESEARCH)

A DIFFERENT entry pattern from the live screener's own -20%/+3%-vs-EMA200 deep pullback — tested standalone, not a threshold variant, per direct request without touching anything live. Criteria: EMA200 genuinely uptrending (126-day slope >=5%, the live G1 threshold) AND not stale (20-day slope >=-2%, the live current-trend gate) AND price below EMA50 AND price above EMA200 AND the same stabilization signal used everywhere else in this codebase (KnifeRiskTier == "stabilising"). Same trade management as the live system (swing-low/EMA-anchored stop, Fibonacci-extension target, +2R/give-1R trailing exit, weak-RR dropped) — holding that constant isolates the entry pattern as what's actually being tested.
- 16576 signals matched, 434 tickers, 2021-06-01 .. present
- engine: research/current_trend_gate_ab.py's run()/stat() — same costs/sizing/sector cap as every other isolated test in this repo

## full

| | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| Pullback-to-50 | +63% | -28% | 0.67 | 831 | 36% | +0.22 | 1.35 | 16 |
| SPY | +83% | | | | | | | |

## 2021-2024

| | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| Pullback-to-50 | +26% | -28% | 0.51 | 549 | 35% | +0.19 | 1.29 | 15 |
| SPY | +40% | | | | | | | |

## 2025-2026

| | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| Pullback-to-50 | +28% | -12% | 0.93 | 282 | 39% | +0.28 | 1.48 | 16 |
| SPY | +31% | | | | | | | |

## 2022

| | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| Pullback-to-50 | -13% | -15% | -1.76 | 72 | 22% | -0.24 | 0.69 | 12 |
| SPY | -20% | | | | | | | |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected. This is a NEW, untested-until-now pattern — treat these numbers with the same skepticism as any first backtest, not as validated the way the live screener's own thresholds are (those went through calibration + train/test + this same isolated-portfolio check before being trusted)._