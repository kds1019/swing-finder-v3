# Super Trend (ATR-based, period=10/mult=3) as a current-trend confirmation (RESEARCH)

Motivated by reviewing an external paper (IJSAT, 2026) that used Super Trend as a trend-confirmation input to a neural net — the paper itself is low-rigor (single stock, undisclosed backtest dates, no reported transaction costs, templated self-citations), so it's tested here on the indicator's own merits, not on the paper's authority. Super Trend computed the standard way (ATR(10) via Wilder's smoothing, bands at (H+L)/2 +- 3*ATR, direction flips when price closes through the current band) per ticker, tagged onto every signal in today's live gate (research/data/current_trend_signals.pkl, unmodified) by its own (ticker, date).
- signal set: today's live gate held exactly as-is, 29945 signals, 0 unmatched

## View 1 — realized trade outcomes, bucketed by the stock's OWN Super Trend direction at entry (one portfolio run, today's actual gate, no additional filter)

| supertrend direction at entry | trades | win% | avgR | medianR | PF |
|---|---|---|---|---|---|
| down | 778 | 36% | +0.23 | -1.00 | 1.36 |
| up | 160 | 38% | +0.15 | -1.00 | 1.25 |

- signal counts per variant: no gate (current)=29945, require supertrend up=4762

## View 2 — portfolio-level gate variant (does requiring Super Trend already flag 'up' help the blended portfolio?)

### full

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +91% | -24% | 0.73 | 938 | 36% | +0.22 | 1.34 | 16 |
| require supertrend up | +45% | -23% | 0.52 | 668 | 36% | +0.20 | 1.32 | 15 |
| SPY | +83% | | | | | | | |

### 2021-2024

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +66% | -24% | 0.85 | 574 | 36% | +0.21 | 1.33 | 17 |
| require supertrend up | +30% | -19% | 0.55 | 426 | 35% | +0.21 | 1.33 | 15 |
| SPY | +40% | | | | | | | |

### 2025-2026

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +14% | -18% | 0.49 | 364 | 37% | +0.22 | 1.36 | 10 |
| require supertrend up | +11% | -23% | 0.45 | 242 | 37% | +0.19 | 1.31 | 10 |
| SPY | +31% | | | | | | | |

### 2022

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | -14% | -15% | -1.19 | 85 | 22% | -0.25 | 0.68 | 17 |
| require supertrend up | -9% | -11% | -0.94 | 41 | 29% | +0.06 | 1.09 | 11 |
| SPY | -20% | | | | | | | |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected._