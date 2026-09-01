# Portfolio backtest — recalibrated screener + trailing exit

- universe: 502 cached tickers (survivorship-biased), 2021-06-01 .. 2026-08-31
- 100,000 start, 4% risk/trade, max 6 positions, 20% position cap, 3/sector, 5bps slip
- exit: trail +2R activate / give 1R, 30-bar max hold
- regime filter (SPY > 200-SMA to open): ON

## Result

| metric | strategy | SPY (same window) |
|---|---|---|
| total return | +12.2% | +82.7% |
| CAGR | +2.2% | +12.2% |
| max drawdown | -52.0% | -25.4% |
| Sharpe (daily, ann.) | 0.24 | — |

## Trades

- closed trades: 1096
- win rate (R>0): 40.1%
- avg R: +0.087   median R: -1.00   profit factor: 1.15
- avg hold: 5.3 bars
- exit reasons: {'stop_hit': 645, 'target_hit': 258, 'trail_stop': 152, 'expired': 36, 'open_at_end': 5}
- weak-RR share of trades taken: 37%

## By year

| year | trades | win% | avg R | end equity |
|---|---|---|---|---|
| 2021 | 72 | 44% | +0.12 | 110,905 |
| 2022 | 66 | 15% | -0.56 | 75,875 |
| 2023 | 231 | 39% | -0.05 | 72,560 |
| 2024 | 297 | 43% | +0.19 | 91,981 |
| 2025 | 220 | 43% | +0.18 | 97,076 |
| 2026 | 210 | 41% | +0.19 | 112,213 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected._