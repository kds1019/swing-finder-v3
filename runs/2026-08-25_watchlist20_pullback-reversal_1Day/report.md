# Backtest report — core.pullback_reversal + core.trade_plan

## Performance vs Benchmarks

| Symbol | Total Return | Strategy Sharpe | Strategy Max DD | Benchmark Return | Benchmark Sharpe | Benchmark Max DD | Round Trips |
|---|---:|---:|---:|---:|---:|---:|---:|
| CTVA | 45.30% | 0.419 | -24.67% | 113.00% | 0.636 | -35.31% | 26 |
| HAL | -4.43% | 0.027 | -22.74% | 93.32% | 0.51 | -55.47% | 18 |
| SUPN | -30.52% | -0.815 | -35.82% | 47.48% | 0.372 | -46.01% | 20 |
| DOW | -24.05% | -0.7 | -31.18% | -43.62% | -0.135 | -70.87% | 10 |
| RIOT | 11.04% | 0.19 | -15.13% | 17.47% | 0.523 | -95.76% | 7 |
| BG | -1.69% | 0.022 | -16.94% | 67.08% | 0.474 | -45.52% | 11 |
| MUR | -24.76% | -0.621 | -38.22% | 184.72% | 0.588 | -61.31% | 15 |
| VNOM | -12.40% | -0.017 | -42.23% | 218.75% | 0.773 | -36.69% | 14 |
| MIAX | 19.49% | 1.119 | -7.98% | 4.24% | 0.525 | -36.99% | 5 |
| AKR | -8.62% | -0.079 | -25.34% | 38.47% | 0.347 | -46.08% | 17 |
| HASI | -36.88% | -1.095 | -36.88% | -37.97% | 0.074 | -77.58% | 18 |
| HLF | 61.40% | 0.421 | -4.04% | -74.54% | -0.122 | -91.21% | 4 |
| MO | 50.28% | 0.389 | -18.95% | 61.92% | 0.506 | -30.65% | 12 |
| CELC | 5.43% | 0.139 | -8.28% | 256.57% | 0.602 | -82.68% | 8 |
| DAN | -34.25% | -1.361 | -34.25% | 52.12% | 0.406 | -72.61% | 10 |
| COLM | -18.67% | -0.621 | -20.93% | -31.96% | -0.06 | -56.86% | 9 |
| CWEN | -4.56% | 0.007 | -20.36% | 1.76% | 0.186 | -54.93% | 22 |
| POR | 11.69% | 0.286 | -11.81% | 19.14% | 0.261 | -31.73% | 10 |
| LXU | -4.16% | -0.177 | -8.48% | 251.79% | 0.674 | -81.32% | 5 |

## Aggregate pattern metrics (all symbols combined)

- Round trips: 241
- Hit rate: 0.2739
- Profit factor: 0.9994654040085672
- Average R-multiple: 0.022

## Strategy configuration

- Symbols: CTVA, HAL, SUPN, DOW, RIOT, BG, MUR, VNOM, MIAX, AKR, HASI, HLF, MO, CELC, DAN, COLM, CWEN, POR, LXU
- Period: 2016-01-01 to 2026-08-25
- Timeframe: 1Day
- Fill model: next_open, 5 bps slippage
- Initial cash per symbol: $100,000

## First and last trade

- First: CTVA entered 2021-06-25 04:00:00 @ 44.6723, exited 2021-06-28 04:00:00 @ 43.8481 (stop_hit)
- Last: LXU entered 2026-07-08 04:00:00 @ 11.2706, exited 2026-08-04 04:00:00 @ 9.6652 (stop_hit)

## Assumptions

- next_open fill model (signal on bar T close, fill at bar T+1 open) -- no look-ahead bias
- 5 bps slippage on entry and exit fills; $0 commissions
- 4% of current equity risked per trade, independent $100,000 account per symbol
- one open position per symbol at a time
- symbols are today's live scan universe replayed over history (survivorship bias, disclosed in notes.md)

## Data fingerprint

- CTVA: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=86320.29
- HAL: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=44867.95
- SUPN: 1528 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=51433.76
- DOW: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=73942.02
- RIOT: 1528 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=23411.07
- BG: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=140644.85
- MUR: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=48493.02
- VNOM: 1528 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=48341.77
- MIAX: 259 bars, 2025-08-14 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=11030.63
- AKR: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=28009.14
- HASI: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=55869.48
- HLF: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=34496.83
- MO: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=77087.26
- CELC: 1427 bars, 2020-07-29 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=40322.75
- DAN: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=28027.10
- COLM: 1528 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=121656.01
- CWEN: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=46478.88
- POR: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=70775.94
- LXU: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=14234.21

## Caveats (see notes.md for full detail)

- Universe/survivorship bias: symbols are today's live scan output replayed over history, not a point-in-time historical universe reconstruction.
- Equity curve steps at trade open/close, not full daily mark-to-market of open positions — Sharpe/max-drawdown are approximate. Hit rate/profit-factor/avg-R above are not affected.
- Trading-activity fees (SEC/FINRA/etc.) excluded — only slippage modeled, $0 commissions.

> **Important disclosure**
> This backtest is a hypothetical historical simulation and does not represent actual
> trading performance. Backtested results do not guarantee future results. Results
> depend on market-data quality, data feed selection, corporate-action handling, fees,
> slippage, liquidity, taxes, execution assumptions, and implementation details. This
> material is for research and educational purposes only and is not investment
> advice, a recommendation, an offer, or a solicitation to buy or sell securities,
> options, cryptocurrencies, or any other financial product. All investments involve
> risk and may lose value. Review Alpaca's disclosures and agreements at
> [alpaca.markets/disclosures](https://alpaca.markets/disclosures).
