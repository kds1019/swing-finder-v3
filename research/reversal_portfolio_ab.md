# Reversal portfolio A/B (RESEARCH)

Same engine/costs as research/portfolio_backtest.py. Columns: total return, max DD, Sharpe, #trades, win%, avgR, PF, max consecutive losers, largest single-sector share of trades.

## full

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss | sectorMax |
|---|---|---|---|---|---|---|---|---|---|
| A | +104% | -61% | 0.44 | 1195 | 34% | +0.18 | 1.27 | 22 | 21% |
| B | +22% | -42% | 0.28 | 1354 | 33% | +0.18 | 1.26 | 21 | 23% |
| C | +2% | -60% | 0.27 | 916 | 34% | +0.12 | 1.18 | 18 | 20% |
| D | +8% | -60% | 0.29 | 952 | 33% | +0.08 | 1.12 | 19 | 20% |
| E | -0% | -59% | 0.25 | 781 | 36% | +0.13 | 1.20 | 22 | 18% |
| SPY | +83% | -25% | | | | | | | |

- A exit reasons (full): {'stop_hit': 775, 'trail_stop': 271, 'target_hit': 115, 'expired': 30, 'open_end': 4}
- B exit reasons (full): {'stop_hit': 902, 'trail_stop': 286, 'target_hit': 140, 'expired': 22, 'open_end': 4}
- C exit reasons (full): {'stop_hit': 591, 'trail_stop': 219, 'target_hit': 58, 'expired': 42, 'open_end': 6}
- D exit reasons (full): {'stop_hit': 621, 'trail_stop': 224, 'target_hit': 57, 'expired': 44, 'open_end': 6}
- E exit reasons (full): {'stop_hit': 489, 'trail_stop': 194, 'expired': 52, 'target_hit': 43, 'open_end': 3}

## 2021-2024

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss | sectorMax |
|---|---|---|---|---|---|---|---|---|---|
| A | +57% | -61% | 0.45 | 701 | 34% | +0.16 | 1.24 | 21 | 24% |
| B | +11% | -42% | 0.27 | 856 | 32% | +0.15 | 1.22 | 19 | 26% |
| C | -10% | -60% | 0.29 | 549 | 34% | +0.16 | 1.24 | 18 | 22% |
| D | -7% | -60% | 0.30 | 581 | 33% | +0.12 | 1.18 | 19 | 21% |
| E | -6% | -59% | 0.27 | 472 | 34% | +0.11 | 1.17 | 21 | 17% |
| SPY | +40% | -25% | | | | | | | |

- A exit reasons (2021-2024): {'stop_hit': 456, 'trail_stop': 158, 'target_hit': 69, 'expired': 18}
- B exit reasons (2021-2024): {'stop_hit': 579, 'trail_stop': 175, 'target_hit': 90, 'expired': 12}
- C exit reasons (2021-2024): {'stop_hit': 355, 'trail_stop': 131, 'target_hit': 38, 'expired': 25}
- D exit reasons (2021-2024): {'stop_hit': 380, 'trail_stop': 136, 'target_hit': 39, 'expired': 26}
- E exit reasons (2021-2024): {'stop_hit': 306, 'trail_stop': 106, 'expired': 32, 'target_hit': 28}

## 2025-2026

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss | sectorMax |
|---|---|---|---|---|---|---|---|---|---|
| A | +28% | -19% | 0.81 | 494 | 34% | +0.20 | 1.31 | 12 | 17% |
| B | +8% | -29% | 0.32 | 498 | 34% | +0.22 | 1.33 | 20 | 22% |
| C | +12% | -22% | 0.43 | 367 | 34% | +0.06 | 1.09 | 16 | 18% |
| D | +15% | -20% | 0.52 | 371 | 33% | +0.02 | 1.03 | 16 | 19% |
| E | +5% | -19% | 0.26 | 309 | 39% | +0.15 | 1.25 | 17 | 19% |
| SPY | +31% | -19% | | | | | | | |

- A exit reasons (2025-2026): {'stop_hit': 319, 'trail_stop': 113, 'target_hit': 46, 'expired': 12, 'open_end': 4}
- B exit reasons (2025-2026): {'stop_hit': 323, 'trail_stop': 111, 'target_hit': 50, 'expired': 10, 'open_end': 4}
- C exit reasons (2025-2026): {'stop_hit': 236, 'trail_stop': 88, 'target_hit': 20, 'expired': 17, 'open_end': 6}
- D exit reasons (2025-2026): {'stop_hit': 241, 'trail_stop': 88, 'target_hit': 18, 'expired': 18, 'open_end': 6}
- E exit reasons (2025-2026): {'stop_hit': 183, 'trail_stop': 88, 'expired': 20, 'target_hit': 15, 'open_end': 3}

## 2022 only

| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss | sectorMax |
|---|---|---|---|---|---|---|---|---|---|
| A | -24% | -24% | -2.27 | 70 | 14% | -0.53 | 0.38 | 15 | 29% |
| B | -30% | -31% | -2.35 | 86 | 14% | -0.47 | 0.45 | 18 | 21% |
| C | -29% | -30% | -2.46 | 63 | 19% | -0.39 | 0.52 | 17 | 19% |
| D | -29% | -29% | -2.44 | 64 | 16% | -0.52 | 0.38 | 18 | 19% |
| E | -11% | -13% | -1.32 | 59 | 34% | +0.03 | 1.04 | 9 | 20% |
| SPY | -20% | -25% | | | | | | | |

- A exit reasons (2022 only): {'stop_hit': 59, 'trail_stop': 7, 'expired': 2, 'target_hit': 2}
- B exit reasons (2022 only): {'stop_hit': 74, 'trail_stop': 8, 'target_hit': 3, 'expired': 1}
- C exit reasons (2022 only): {'stop_hit': 51, 'trail_stop': 6, 'expired': 4, 'target_hit': 2}
- D exit reasons (2022 only): {'stop_hit': 54, 'trail_stop': 5, 'expired': 3, 'target_hit': 2}
- E exit reasons (2022 only): {'stop_hit': 39, 'trail_stop': 14, 'expired': 4, 'target_hit': 2}

## Signal counts (eligible, full period)

- A: 29752 eligible signals over the whole window
- B: 57652 eligible signals over the whole window
- C: 33973 eligible signals over the whole window
- D: 57652 eligible signals over the whole window
- E: 33973 eligible signals over the whole window