# Candidate-pool order — portfolio A/B (RESEARCH)

- 466 tickers (research/frozen_universe.csv), 46050 live signals (screener pass + weak-RR drop) from 2021-06-01
- signals per SPY>200SMA day: median 32, 100% of days have more signals than the 6-slot book (only on those days, with a slot free, can the order matter)
- order variants: base, depth_only, falling_first, random (random seed 11); depth tie-break = deepest below EMA200 first

## Per-signal outcome by tier (no portfolio; every SPY>200SMA signal on its own)

| tier | all signals: n / avgR / PF | episode-first: n / avgR / PF | episode-first avgR 2021-24 / 2025-26 |
|---|---|---|---|
| stabilising | 19835 / +0.11 / 1.16 | 1721 / +0.08 / 1.12 | +0.16 / -0.03 |
| forming | 11278 / +0.19 / 1.28 | 1980 / +0.13 / 1.20 | +0.18 / +0.08 |
| unknown | 0 | 0 | +nan / +nan |
| still_falling | 4159 / +0.22 / 1.32 | 1156 / +0.18 / 1.26 | +0.21 / +0.13 |

_Episode-first = first signal of a setup (>5 bars since that ticker's previous signal)._

## Portfolio — noise-aware

_Noise-aware view (research/robust.py): every variant run on the full frozen set plus 10 paired random 90% ticker subsets. 'beats' = subsets where the variant beat the baseline on that SAME subset. Decide on avgR/PF and the beats counts; a single total return is mostly noise._

### full

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| base | +121% | +71% [+45..+112] | -26% | +0.19 / +0.18 | 1.30 / 1.28 | 892 | baseline | baseline |
| depth_only | +47% | +57% [+41..+82] | -29% | +0.18 / +0.18 | 1.28 / 1.28 | 1254 | 2/10 | 5/10 |
| falling_first | +85% | +85% [+53..+134] | -33% | +0.23 / +0.24 | 1.35 / 1.36 | 1471 | 6/10 | 9/10 |
| random | +7% | +8% [-22..+41] | -30% | +0.16 / +0.14 | 1.24 / 1.22 | 996 | 0/10 | 3/10 |
| SPY | +82% | | | | | | | |

### 2021-2024

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| base | +84% | +69% [+38..+104] | -26% | +0.26 / +0.23 | 1.41 / 1.36 | 514 | baseline | baseline |
| depth_only | +37% | +37% [+21..+46] | -29% | +0.15 / +0.14 | 1.22 / 1.21 | 678 | 0/10 | 0/10 |
| falling_first | +43% | +40% [+20..+63] | -33% | +0.22 / +0.21 | 1.33 / 1.33 | 806 | 1/10 | 3/10 |
| random | -2% | +11% [-18..+23] | -30% | +0.17 / +0.15 | 1.26 / 1.22 | 575 | 0/10 | 2/10 |
| SPY | +40% | | | | | | | |

### 2025-2026

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| base | +21% | +4% [-10..+22] | -19% | +0.10 / +0.11 | 1.16 / 1.17 | 378 | baseline | baseline |
| depth_only | +6% | +16% [+4..+31] | -16% | +0.23 / +0.21 | 1.35 / 1.32 | 576 | 8/10 | 10/10 |
| falling_first | +25% | +33% [+17..+42] | -17% | +0.25 / +0.25 | 1.37 / 1.39 | 665 | 10/10 | 10/10 |
| random | +10% | +3% [-9..+17] | -20% | +0.14 / +0.11 | 1.21 / 1.18 | 421 | 4/10 | 6/10 |
| SPY | +31% | | | | | | | |

### 2022

| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | PF full / med | trades | beats base (ret) | beats base (avgR) |
|---|---|---|---|---|---|---|---|---|
| base | -14% | -12% [-20..-9] | -13% | -0.20 / -0.22 | 0.74 / 0.71 | 61 | baseline | baseline |
| depth_only | -18% | -17% [-21..-14] | -18% | -0.34 / -0.33 | 0.58 / 0.58 | 88 | 1/10 | 2/10 |
| falling_first | -28% | -27% [-31..-25] | -27% | -0.62 / -0.57 | 0.29 / 0.34 | 89 | 0/10 | 0/10 |
| random | -18% | -17% [-21..-11] | -20% | -0.34 / -0.33 | 0.54 / 0.56 | 67 | 1/10 | 1/10 |
| SPY | -20% | | | | | | | |

## Portfolio — full

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| base | +121% | -21% | 0.97 | 892 | 36% | +0.19 | 1.30 | 16 |
| depth_only | +47% | -31% | 0.50 | 1254 | 34% | +0.18 | 1.28 | 21 |
| falling_first | +85% | -34% | 0.78 | 1471 | 34% | +0.23 | 1.35 | 29 |
| random | +7% | -34% | 0.16 | 996 | 34% | +0.16 | 1.24 | 17 |
| SPY | +82% | | | | | | | |

## Portfolio — 2021-2024

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| base | +84% | -21% | 1.13 | 514 | 37% | +0.26 | 1.41 | 16 |
| depth_only | +37% | -31% | 0.60 | 678 | 34% | +0.15 | 1.22 | 23 |
| falling_first | +43% | -34% | 0.70 | 806 | 33% | +0.22 | 1.33 | 28 |
| random | -2% | -34% | 0.04 | 575 | 34% | +0.17 | 1.26 | 17 |
| SPY | +40% | | | | | | | |

## Portfolio — 2025-2026

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| base | +21% | -16% | 0.68 | 378 | 35% | +0.10 | 1.16 | 14 |
| depth_only | +6% | -19% | 0.28 | 576 | 34% | +0.23 | 1.35 | 16 |
| falling_first | +25% | -19% | 0.84 | 665 | 35% | +0.25 | 1.37 | 17 |
| random | +10% | -18% | 0.39 | 421 | 34% | +0.14 | 1.21 | 17 |
| SPY | +31% | | | | | | | |

## Portfolio — 2022

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |
|---|---|---|---|---|---|---|---|---|
| base | -14% | -16% | -1.27 | 61 | 23% | -0.20 | 0.74 | 16 |
| depth_only | -18% | -19% | -1.70 | 88 | 19% | -0.34 | 0.58 | 23 |
| falling_first | -28% | -28% | -3.37 | 89 | 12% | -0.62 | 0.29 | 29 |
| random | -18% | -21% | -1.75 | 67 | 25% | -0.34 | 0.54 | 17 |
| SPY | -20% | | | | | | | |

## Which tiers each order actually traded (full period, closed trades)

| variant | stabilising: n / avgR | forming: n / avgR | unknown: n / avgR | still_falling: n / avgR |
|---|---|---|---|---|
| base | 885 / +0.19 | 7 / +1.00 | 0 | 0 |
| depth_only | 409 / +0.11 | 468 / +0.12 | 0 | 377 / +0.35 |
| falling_first | 47 / -0.25 | 263 / +0.11 | 0 | 1161 / +0.28 |
| random | 514 / +0.11 | 332 / +0.15 | 0 | 150 / +0.33 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected (today's universe). The live Decision Agent's own ranking judgment is not modelled — this tests the mechanical order only._