# Portfolio backtest — recalibrated screener + trailing exit

- universe: 482 cached tickers (survivorship-biased), 2021-06-01 .. 2026-09-11
- 100,000 start, 4% risk/trade, max 6 positions, 20% position cap, 3/sector, 5bps slip
- exit: trail +2R activate / give 1R, 30-bar max hold
- regime filter (SPY > 200-SMA to open): ON

## Result

| metric | strategy | SPY (same window) |
|---|---|---|
| total return | +45.2% | +82.7% |
| CAGR | +7.3% | +12.1% |
| max drawdown | -37.9% | -25.4% |
| Sharpe (daily, ann.) | 0.47 | — |

## Trades

- closed trades: 1052
- win rate (R>0): 43.6%
- avg R: +0.158   median R: -1.00   profit factor: 1.28
- avg hold: 5.6 bars
- exit reasons: {'stop_hit': 580, 'target_hit': 304, 'trail_stop': 126, 'expired': 39, 'open_at_end': 3}
- weak-RR share of trades taken: 43%

## By year

| year | trades | win% | avg R | end equity |
|---|---|---|---|---|
| 2021 | 56 | 48% | +0.20 | 107,853 |
| 2022 | 69 | 28% | -0.17 | 92,079 |
| 2023 | 228 | 43% | +0.09 | 85,340 |
| 2024 | 282 | 46% | +0.21 | 129,671 |
| 2025 | 217 | 43% | +0.10 | 118,138 |
| 2026 | 200 | 46% | +0.33 | 145,166 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected._