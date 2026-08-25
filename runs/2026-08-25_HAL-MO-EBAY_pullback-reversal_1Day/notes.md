# Backtest notes — core.pullback_reversal + core.trade_plan

## Original request

"Research whether Alpaca offers a backtesting setup I could connect my screener to" ->
confirmed yes (Alpaca's `alpacahq/alpaca-skills` Backtesting Skill) -> "how do we do
that based off of my scanner filters" -> this run.

The goal is to answer a question `agents/decision_agent.py`'s own system prompt already
flags: the pullback/EMA200/bounce screener "has NOT been statistically validated the way
the system it replaced was found to have no edge" — nothing in this codebase has ever
actually backtested `core/pullback_reversal.py`. This is that first test.

## Confirmed strategy interpretation

- **Symbols**: HAL, MO, EBAY — chosen from today's actual live scan output
  (`results/2026-08-24_2323Z.json`, the 2026-08-24 23:23Z run), all currently in the
  live universe (price 10-150, volume >=500k), long trading histories (decades),
  three different sectors. This is a single-symbol-style first pass (per explicit
  user choice), not the full multi-symbol portfolio simulation.
- **Timeframe**: 1Day bars, feed=iex, adjustment=split. `agents/market_data_agent.py`
  already documents why: "This account only has IEX (free-tier) market data access —
  SIP (all-exchange, paid) returns 'subscription does not permit querying recent SIP
  data.'" (confirmed again live here: the first real run of this backtest 403'd on
  `feed=sip` before this was corrected to match production's actual feed).
- **Date range**: 2016-01-01 through 2026-08-25. The first ~427 bars
  (300-bar warmup convention from `config/settings.py::bars_lookback_days`, plus the
  screener's own 126-bar EMA200-trend lookback) are warmup only — not eligible for
  signal evaluation, matching `core/pullback_reversal.py::MIN_BARS_FOR_SCREENER`.
- **Indicators**: uses the project's actual `core/indicators.py` and
  `core/volume_profile.py` functions directly (imported, not re-derived) —
  - EMA20/EMA200: `close.ewm(span=N, adjust=False).mean()`.
  - ATR14: True Range then `tr.ewm(alpha=1/14, adjust=False).mean()`.
  - Volume profile: 20 bins over the trailing 60 bars, POC = max-volume bin, value
    area = bins holding the top 70% of volume.

  **Disclosed deviation from this skill's canonical indicator formulas
  (reference.md#indicator-formulas)**: the skill's canonical EMA is SMA-seeded and
  its canonical ATR is Wilder-seeded (simple average of the first N values as the
  seed). This project's actual EMA/ATR (`pandas .ewm(adjust=False)`) seeds at the
  first data point instead. The two converge within a few multiples of the
  span/period and are effectively identical by the time 300+ warmup bars have
  passed, but they are not bit-identical near the start of a series. This
  deviation is intentional: the entire point of this backtest is testing this
  project's actual, live scanner behavior, not the skill's generic default.

- **Entry signal** (`core.pullback_reversal.detect_pullback_reversal`, evaluated at
  bar T's close using only data up to and including T):
  1. EMA200 uptrend: `(EMA200[T] - EMA200[T-126]) / EMA200[T-126] >= 5.0%`
  2. Price vs EMA200: `-12.0% <= (Close[T]-EMA200[T])/EMA200[T] <= 8.0%`
  3. Consolidation: over trailing 15 closes, `(max-min)/Close[T] <= 15.0%`
  4. Bounce off low: `(Close[T]-window_low)/window_low >= 3.0%`
  5. Not extended above value area: `Close[T] <= value_area_high` (60-bar volume profile)
  6. Requires >=127 bars of history to evaluate at all

- **Stop/target** (`core.trade_plan.compute_trade_plan`, computed at the same bar T):
  - Stop: `min(10-day swing low, EMA20[T] - 1.3*ATR14[T])`, refined to nearest support
    cluster (pivot-based, window=10, `core.trade_plan.find_support_resistance`) if
    within 3xATR14 of price.
  - Target: Fibonacci 1.618 extension of the last 20 bars, floored at 3:1 R:R
    (`config/settings.py::min_risk_reward`), refined to nearest resistance cluster if
    tighter and still clears 3:1.

- **Fill model: `next_open`** (user-confirmed) — signal on bar T's close, fill at bar
  T+1's open. The live pipeline itself computes entry/stop/target from bar T's own
  close and effectively treats that as the entry price — which is not something you
  could actually have executed at (the close is only known once the session ends).
  `next_open` removes that look-ahead bias; results here will NOT exactly match the
  `Entry` price the live pipeline prints for the same setup, by design.
  Friction: `slippage_bps=5` applied to both entry and exit fills (bar-based fill,
  no quotes fetched for this daily-bar strategy). Not modeled in production at all —
  this is a disclosed backtest-only assumption, chosen as a conservative default
  rather than assuming frictionless fills.

- **Exit** (`core.trade_plan.resolve_trade_plan_outcome` — reused verbatim, this is
  the same function the live pipeline uses to score its own past picks): walk forward
  up to `MAX_HOLD_DAYS=30` trading days (`core/pick_tracking.py::MAX_HOLD_DAYS`);
  stop checked before target on same-bar conflicts (conservative); unresolved after 30
  bars -> exit at that bar's close (`expired_unresolved`). A signal whose 30-bar
  resolution window runs past the end of the fetched data range is excluded from
  round-trip stats entirely (not scored either way) — per this skill's own guidance.

- **Position sizing**: `equity_fraction`-style, matching production's real
  `config/settings.py::risk_per_trade_pct = 4.0` exactly — `shares =
  floor(equity * 0.04 / abs(entry_fill - stop))`, computed at signal time (bar T
  close), each symbol as its own fully independent $100,000 account (no shared/
  competing capital across symbols in this first pass). Only one open position per
  symbol at a time; a new signal is not evaluated again for a symbol until its prior
  trade has fully resolved.

- **Benchmark**: buy-and-hold of the same symbol (per this skill's mandatory-benchmark
  rule for a single-symbol strategy), entering at the first bar the strategy could
  have possibly traded (same warmup boundary), same friction/feed/adjustment
  assumptions.

## Known limitations (disclosed, not hidden)

- **Universe/survivorship bias** (user-confirmed to accept for this first pass):
  these three tickers are today's live universe/scan output, replayed over history.
  This is not a point-in-time historical reconstruction of "which ~900 stocks passed
  the live FMP screener's price/volume filters on each past date" — no such
  historical snapshot exists. A stock that would have failed today's filters in the
  past (or a delisted name that would have passed) isn't represented. Treat this as
  "does the pattern show edge on stocks that look like today's candidates," not a
  claim about the full historical universe.
- **Equity curve granularity**: `equity.csv` steps at trade open/close, not full
  daily mark-to-market of open positions. Sharpe/max-drawdown computed from it are
  therefore approximate. The primary metrics that actually answer "does this pattern
  have edge" — hit rate, profit factor, average R-multiple — are computed directly
  from `round_trips.csv` and are NOT affected by this simplification.
- **Fees**: trading-activity fees (SEC/FINRA/etc., see `fee_source.json`) are
  excluded from this first pass — only slippage is modeled. Commissions are $0
  (matches Webull's real commission-free equity trading).
- This is a **hypothetical historical simulation**, not a promise of future results —
  see the mandatory disclosure block in `report.md`.

## How to actually run this

Not yet executed — needs real Alpaca credentials this sandbox doesn't have.

```bash
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
export ALPACA_QUIET=1
alpaca doctor   # confirm auth before fetching

for SYMBOL in HAL MO EBAY; do
  alpaca data bars \
    --symbol "$SYMBOL" \
    --start 2016-01-01 \
    --end 2026-08-25 \
    --timeframe 1Day \
    --feed iex \
    --adjustment split \
    --quiet > "raw/bars_${SYMBOL}.json"
done

python run.py
```

`run.py` reads `raw/bars_<SYMBOL>.json`, computes signals/trades using the real
`core.indicators` / `core.pullback_reversal` / `core.trade_plan` modules (imported
directly from the repo, not re-implemented), and writes `summary.json`, `report.md`,
`trades.csv`, `round_trips.csv`, `equity_<SYMBOL>.csv`,
`benchmark_equity_<SYMBOL>.csv`, and `warnings.json`.

## Disclosures

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
