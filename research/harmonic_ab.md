# Bullish harmonic patterns (Gartley, Bat) — two-part test (RESEARCH)

- 467 tickers (research/frozen_universe.csv), signals from 2021-06-01
- detector: research/harmonic.py — ZigZag 2.5x ATR14, pattern bands fixed in advance (see its docstring), active window 20 bars after the first touch, causal (no repainting)
- live signal set (screener pass + weak-RR drop, SPY>200SMA days): 35355 signals, 818 (2.3%) with an active bullish pattern
- harmonic touch events in the universe (SPY>200SMA days): 291 (199 Gartley, 92 Bat)
- per-signal R uses the live exit rules on each signal independently; consecutive daily signals on one stock overlap, so the 'episode-first' rows (first signal per setup, >5 bars apart) are the fairer count

## Part 1 — harmonic tag on the screened signals

### 1a. Per-signal outcomes by tag

**All live signals**

| bucket | n | win% | avgR | medianR | PF | fwd20 mean | fwd20 >0 | avgR 2021-24 (n) | avgR 2025-26 (n) |
|---|---|---|---|---|---|---|---|---|---|
| no pattern | 34457 | 35% | +0.14 | -1.01 | 1.22 | +0.6% | 51% | +0.18 (19162) | +0.09 (15295) |
| pattern active (any) | 815 | 37% | +0.29 | -1.01 | 1.46 | +2.4% | 55% | +0.36 (515) | +0.17 (300) |
|   Gartley | 543 | 34% | +0.20 | -1.01 | 1.31 | +2.7% | 53% | +0.22 (352) | +0.18 (191) |
|   Bat | 272 | 45% | +0.45 | -1.01 | 1.84 | +2.0% | 57% | +0.66 (163) | +0.14 (109) |
|   in zone today | 109 | 39% | +0.25 | -1.01 | 1.41 | +3.1% | 61% | +0.27 (71) | +0.22 (38) |
|   active, bouncing off zone | 706 | 37% | +0.29 | -1.01 | 1.47 | +2.3% | 54% | +0.37 (444) | +0.16 (262) |

**Episode-first signals**

| bucket | n | win% | avgR | medianR | PF | fwd20 mean | fwd20 >0 | avgR 2021-24 (n) | avgR 2025-26 (n) |
|---|---|---|---|---|---|---|---|---|---|
| no pattern | 4749 | 34% | +0.13 | -1.01 | 1.19 | +0.8% | 50% | +0.18 (2720) | +0.05 (2029) |
| pattern active (any) | 108 | 33% | +0.09 | -1.01 | 1.14 | +2.4% | 54% | +0.11 (64) | +0.07 (44) |
|   Gartley | 80 | 31% | +0.06 | -1.01 | 1.08 | +2.6% | 54% | +0.14 (48) | -0.07 (32) |
|   Bat | 28 | 39% | +0.21 | -1.01 | 1.33 | +2.1% | 54% | +0.03 (16) | +0.44 (12) |
|   in zone today | 14 | 43% | +0.38 | -1.01 | 1.64 | +4.3% | 71% | +0.44 (9) | +0.26 (5) |
|   active, bouncing off zone | 94 | 32% | +0.05 | -1.01 | 1.07 | +2.2% | 51% | +0.06 (55) | +0.04 (39) |

**By knife tier (episode-first) — does the pattern add anything inside a tier?**

| tier | no pattern: n / avgR / PF | pattern: n / avgR / PF |
|---|---|---|
| stabilising | 1674 / +0.09 / 1.13 | 47 / -0.20 / 0.73 |
| forming | 1943 / +0.13 / 1.19 | 37 / +0.40 / 1.63 |
| still_falling | 1132 / +0.18 / 1.26 | 24 / +0.19 / 1.29 |

### 1b. Portfolio (live engine), closed trades bucketed by tag

| bucket | n | win% | avgR | medianR | PF | avgR 2021-24 (n) | avgR 2025-26 (n) |
|---|---|---|---|---|---|---|---|
| no pattern | 860 | 36% | +0.21 | -1.00 | 1.33 | +0.28 (494) | +0.12 (366) |
| pattern active | 32 | 25% | -0.20 | -1.00 | 0.73 | -0.15 (20) | -0.29 (12) |

### 1c. Portfolio variants

- signal counts: base=46050, harmonic_priority=46050, require_harmonic=1106

_Noise-aware view (research/robust.py): every variant run on the full frozen set plus 10 paired random 90% ticker subsets. 'beats' = subsets where the variant beat the baseline on that SAME subset. Decide on avgR/PF and the beats counts; a single total return is mostly noise._

### full

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| base | +121% | +71% [+45..+112] | -26% | +0.19 / +0.18 | 1.30 / 1.28 | 892 | baseline | baseline |
| harmonic_priority | +111% | +85% [+55..+158] | -26% | +0.17 / +0.17 | 1.26 / 1.26 | 892 | 8/10 | 3/10 |
| require_harmonic | +49% | +40% [+36..+48] | -10% | +0.40 / +0.39 | 1.65 / 1.62 | 281 | 0/10 | 10/10 |
| SPY | +82% | | | | | | | |

### 2021-2024

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| base | +84% | +69% [+38..+104] | -26% | +0.26 / +0.23 | 1.41 / 1.36 | 514 | baseline | baseline |
| harmonic_priority | +85% | +72% [+44..+97] | -26% | +0.21 / +0.20 | 1.32 / 1.31 | 524 | 4/10 | 2/10 |
| require_harmonic | +30% | +27% [+18..+34] | -9% | +0.48 / +0.48 | 1.81 / 1.80 | 166 | 0/10 | 10/10 |
| SPY | +40% | | | | | | | |

### 2025-2026

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| base | +21% | +4% [-10..+22] | -19% | +0.10 / +0.11 | 1.16 / 1.17 | 378 | baseline | baseline |
| harmonic_priority | +14% | +9% [+1..+33] | -16% | +0.11 / +0.10 | 1.17 / 1.16 | 368 | 9/10 | 4/10 |
| require_harmonic | +15% | +12% [+5..+18] | -10% | +0.29 / +0.27 | 1.45 / 1.40 | 115 | 9/10 | 10/10 |
| SPY | +31% | | | | | | | |

### 2022

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| base | -14% | -12% [-20..-9] | -13% | -0.20 / -0.22 | 0.74 / 0.71 | 61 | baseline | baseline |
| harmonic_priority | -11% | -12% [-20..-7] | -13% | -0.16 / -0.23 | 0.79 / 0.70 | 68 | 5/10 | 5/10 |
| require_harmonic | -0% | -1% [-2..+1] | -4% | +0.94 / +0.85 | 2.70 / 2.51 | 20 | 10/10 | 10/10 |
| SPY | -20% | | | | | | | |

#### full

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| base | +121% | -21% | 0.97 | 892 | 36% | +0.19 | 1.30 | 16 |
| harmonic_priority | +111% | -25% | 0.89 | 892 | 35% | +0.17 | 1.26 | 16 |
| require_harmonic | +49% | -10% | 0.84 | 281 | 37% | +0.40 | 1.65 | 22 |
| SPY | +82% | | | | | | | |

#### 2021-2024

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| base | +84% | -21% | 1.13 | 514 | 37% | +0.26 | 1.41 | 16 |
| harmonic_priority | +85% | -25% | 1.10 | 524 | 36% | +0.21 | 1.32 | 15 |
| require_harmonic | +30% | -9% | 0.84 | 166 | 40% | +0.48 | 1.81 | 8 |
| SPY | +40% | | | | | | | |

#### 2025-2026

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| base | +21% | -16% | 0.68 | 378 | 35% | +0.10 | 1.16 | 14 |
| harmonic_priority | +14% | -16% | 0.51 | 368 | 35% | +0.11 | 1.17 | 12 |
| require_harmonic | +15% | -10% | 0.87 | 115 | 34% | +0.29 | 1.45 | 22 |
| SPY | +31% | | | | | | | |

#### 2022

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| base | -14% | -16% | -1.27 | 61 | 23% | -0.20 | 0.74 | 16 |
| harmonic_priority | -11% | -12% | -0.98 | 68 | 25% | -0.16 | 0.79 | 9 |
| require_harmonic | -0% | -4% | -0.00 | 20 | 45% | +0.94 | 2.70 | 5 |
| SPY | -20% | | | | | | | |

### 1d. Live stop vs the pattern's invalidation level X (episode-first, pattern active)

- live stop ABOVE X (inside the pattern's normal noise): 97 (90%); stop at/below X: 11

| bucket | n | win% | avgR | medianR | PF |
|---|---|---|---|---|---|
| stop above X | 97 | 34% | +0.13 | -1.01 | 1.19 |
| stop at/below X | 11 | 27% | -0.21 | -1.01 | 0.72 |

## Part 2 — what the screener drops

Every bullish pattern, on the bar price first reaches its reversal zone.

### 2a. Funnel on the touch bar

| touch-bar verdict | events | share | caught on a later bar while active |
|---|---|---|---|
| passed (live signal) | 38 | 13% | 100% |
| passed gates, dropped weak-RR | 96 | 33% | 74% |
| rejected: extended_above_value_area | 3 | 1% | 67% |
| rejected: no_long_term_uptrend | 130 | 45% | 1% |
| rejected: not_consolidating | 21 | 7% | 43% |
| rejected: price_above_ema50 | 2 | 1% | 100% |
| rejected: price_too_far_below_ema200 | 1 | 0% | 0% |

- never a live signal at any point while the pattern was active: 168 of 291 (58%)

### 2b. Outcomes (touch-bar entry, live plan + exits) vs the live signal set

Rows with no valid live trade plan have no R (fwd20 still shown).

| bucket | n | win% | avgR | medianR | PF | fwd20 mean | fwd20 >0 | avgR 2021-24 (n) | avgR 2025-26 (n) |
|---|---|---|---|---|---|---|---|---|---|
| REFERENCE: all live signals (episode-first) | 4857 | 34% | +0.13 | -1.01 | 1.19 | +0.9% | 50% | +0.18 (2784) | +0.05 (2073) |
| all touch events | 291 | 48% | +0.04 | -1.00 | 1.08 | +2.7% | 53% | +0.16 (164) | -0.12 (127) |
| touch passed (live signal) | 38 | 32% | +0.19 | -1.01 | 1.27 | +1.6% | 53% | +0.38 (24) | -0.13 (14) |
| touch rejected, caught later | 14 | 57% | +0.18 | +0.07 | 1.40 | +2.4% | 50% | -0.38 (5) | +0.48 (9) |
| never caught (any reason) | 168 | 41% | -0.13 | -1.01 | 0.79 | +2.2% | 48% | +0.03 (92) | -0.32 (76) |
|   never caught — passed gates, dropped weak-RR | 25 | 20% | -0.61 | -1.01 | 0.26 | +1.2% | 36% | -0.75 (11) | -0.49 (14) |
|   never caught — rejected: extended_above_value_area | 1 | 0% | -1.01 | -1.01 | 0.00 | -13.3% | 0% | -1.01 (1) | – (0) |
|   never caught — rejected: no_long_term_uptrend | 129 | 47% | +0.02 | -1.01 | 1.04 | +3.1% | 53% | +0.17 (74) | -0.19 (55) |
|   never caught — rejected: not_consolidating | 12 | 25% | -0.58 | -1.01 | 0.23 | -3.3% | 33% | -0.15 (6) | -1.01 (6) |
|   never caught — rejected: price_too_far_below_ema200 | 1 | 0% | -1.00 | -1.00 | 0.00 | -13.4% | 0% | – (0) | -1.00 (1) |
| touch, tier stabilising | 3 | 33% | -0.27 | -1.02 | 0.60 | -6.9% | 33% | -1.02 (2) | +1.22 (1) |
| touch, tier forming | 86 | 47% | +0.12 | -1.01 | 1.23 | +1.6% | 50% | +0.18 (49) | +0.04 (37) |
| touch, tier still_falling | 202 | 49% | +0.01 | -1.00 | 1.02 | +3.2% | 54% | +0.17 (113) | -0.19 (89) |

### 2c. Candidate-pool squeeze

- days with more than 30 live signals: 663 of 1265 (frozen ~500-ticker universe; the live ~full universe binds far more often)
- harmonic-active live signals outside the pool: 269 of 818
- all live signals outside the pool: 14292 of 46050

| bucket | n | win% | avgR | medianR | PF |
|---|---|---|---|---|---|
| harmonic, in pool | 549 | 36% | +0.21 | -1.01 | 1.33 |
| harmonic, squeezed out | 266 | 41% | +0.44 | -1.01 | 1.75 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected (today's universe). Detector parameters fixed before the run._