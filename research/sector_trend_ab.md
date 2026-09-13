# Sector-trend read — does the STOCK's own SECTOR trending down predict a weaker return? (RESEARCH)

Hypothesis (raised directly 2026-09-13): a pick that clears every individual-stock gate while its own SECTOR is rolling over is less likely to produce a strong return — trading against the group, not with it. Sector trend read the same way core.trend_context.compute_trend_state() reads an individual stock (EMA50/EMA200, 20-session EMA200 slope, +-0.5% flat band), applied to the SPDR sector ETF matching each signal's FMP/GICS sector label instead of the stock itself.
- signal set: today's live gate (research/data/current_trend_signals.pkl) held exactly as-is, 29945 signals, 0 unmatched (dropped from the bucket view below)
- engine: research/current_trend_gate_ab.py's run()/stat() conventions

## View 1 — realized trade outcomes, bucketed by sector_trend_state at entry (one portfolio run, today's actual gate, no additional filter)

| sector_trend_state | trades | win% | avgR | medianR | PF |
|---|---|---|---|---|---|
| downtrend | 78 | 42% | +0.45 | -1.00 | 1.81 |
| transitional | 305 | 33% | +0.13 | -1.00 | 1.19 |
| uptrend | 555 | 37% | +0.23 | -1.00 | 1.37 |

- signal counts per variant: no gate (current)=29945, exclude sector_downtrend=25372, sector_uptrend_only=14674

## View 2 — portfolio-level gate variants (does FILTERING on sector trend help the blended portfolio?)

### full

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +91% | -24% | 0.73 | 938 | 36% | +0.22 | 1.34 | 16 |
| exclude sector_downtrend | +79% | -25% | 0.68 | 954 | 36% | +0.20 | 1.31 | 16 |
| sector_uptrend_only | +58% | -27% | 0.58 | 897 | 36% | +0.20 | 1.31 | 24 |
| SPY | +83% | | | | | | | |

### 2021-2024

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +66% | -24% | 0.85 | 574 | 36% | +0.21 | 1.33 | 17 |
| exclude sector_downtrend | +65% | -25% | 0.83 | 590 | 37% | +0.22 | 1.35 | 16 |
| sector_uptrend_only | +24% | -27% | 0.45 | 549 | 34% | +0.18 | 1.29 | 24 |
| SPY | +40% | | | | | | | |

### 2025-2026

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | +14% | -18% | 0.49 | 364 | 37% | +0.22 | 1.36 | 10 |
| exclude sector_downtrend | +8% | -18% | 0.32 | 364 | 35% | +0.16 | 1.25 | 14 |
| sector_uptrend_only | +27% | -16% | 0.79 | 348 | 37% | +0.22 | 1.35 | 10 |
| SPY | +31% | | | | | | | |

### 2022

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| no gate (current) | -14% | -15% | -1.19 | 85 | 22% | -0.25 | 0.68 | 17 |
| exclude sector_downtrend | -12% | -15% | -1.00 | 83 | 23% | -0.23 | 0.71 | 16 |
| sector_uptrend_only | -20% | -20% | -2.03 | 65 | 20% | -0.33 | 0.58 | 24 |
| SPY | -20% | | | | | | | |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected. Sector ETF history starts 2019-06-01 so the 220-bar trend-state minimum is satisfied well before this backtest's 2021-06-01 signal start._