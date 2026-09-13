# Portfolio backtest — recalibrated screener + trailing exit

- universe: 482 cached tickers (survivorship-biased), 2021-06-01 .. 2026-09-11
- 100,000 start, 4% risk/trade, max 6 positions, 20% position cap, 3/sector, 5bps slip
- exit: trail +2R activate / give 1R, 30-bar max hold
- regime filter (SPY > 200-SMA to open): ON

## Result

| metric | strategy | SPY (same window) |
|---|---|---|
| total return | +106.1% | +82.7% |
| CAGR | +14.7% | +12.1% |
| max drawdown | -28.5% | -25.4% |
| Sharpe (daily, ann.) | 0.88 | — |

## Trades

- closed trades: 1187
- win rate (R>0): 35.0%
- avg R: +0.223   median R: -1.00   profit factor: 1.35
- avg hold: 4.9 bars
- exit reasons: {'stop_hit': 763, 'trail_stop': 276, 'target_hit': 122, 'expired': 22, 'open_at_end': 4}
- weak-RR share of trades taken: 0%

## By year

| year | trades | win% | avg R | end equity |
|---|---|---|---|---|
| 2021 | 83 | 41% | +0.48 | 113,878 |
| 2022 | 76 | 18% | -0.33 | 95,105 |
| 2023 | 234 | 32% | +0.06 | 108,658 |
| 2024 | 272 | 40% | +0.37 | 168,127 |
| 2025 | 297 | 33% | +0.18 | 169,151 |
| 2026 | 225 | 37% | +0.37 | 206,125 |

_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not corrected._