# Backtest report — core.pullback_reversal + core.trade_plan

## Performance vs Benchmarks

| Symbol | Total Return | Strategy Sharpe | Strategy Max DD | Benchmark Return | Benchmark Sharpe | Benchmark Max DD | Round Trips |
|---|---:|---:|---:|---:|---:|---:|---:|
| HAL | -4.43% | 0.027 | -22.74% | 92.37% | 0.507 | -55.47% | 18 |
| MO | 50.28% | 0.389 | -18.95% | 61.08% | 0.502 | -30.65% | 12 |
| EBAY | 3.72% | 0.123 | -22.17% | 80.79% | 0.493 | -54.33% | 10 |

## Aggregate pattern metrics (all symbols combined)

- Round trips: 40
- Hit rate: 0.35
- Profit factor: 1.4733018207617163
- Average R-multiple: 0.384

## Strategy configuration

- Symbols: HAL, MO, EBAY
- Period: 2016-01-01 to 2026-08-25
- Timeframe: 1Day
- Fill model: next_open, 5 bps slippage
- Initial cash per symbol: $100,000

## First and last trade

- First: HAL entered 2021-07-27 04:00:00 @ 20.4102, exited 2021-08-03 04:00:00 @ 19.93 (stop_hit)
- Last: EBAY entered 2026-02-26 05:00:00 @ 85.9129, exited 2026-04-10 04:00:00 @ 95.3723 (expired_unresolved)

## Assumptions

- next_open fill model (signal on bar T close, fill at bar T+1 open) -- no look-ahead bias
- 5 bps slippage on entry and exit fills; $0 commissions
- 4% of current equity risked per trade, independent $100,000 account per symbol
- one open position per symbol at a time
- symbols are today's live scan universe replayed over history (survivorship bias, disclosed in notes.md)

## Data fingerprint

- HAL: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=44867.78
- MO: 1525 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=77086.91
- EBAY: 1528 bars, 2020-07-27 04:00:00 to 2026-08-25 04:00:00, feed=iex, adjustment=split, close_sum=95364.45

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
