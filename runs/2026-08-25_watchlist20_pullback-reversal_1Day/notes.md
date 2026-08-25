# Backtest notes — core.pullback_reversal + core.trade_plan (20-symbol variant)

## Run lineage

Variant of `runs/2026-08-25_HAL-MO-EBAY_pullback-reversal_1Day/`. What changed:
**universe expanded from 3 hand-picked long-history tickers (HAL, MO, EBAY) to the 20
tickers from the live pipeline's own 2026-08-24 23:23Z ranked-picks output**
(`results/2026-08-24_2323Z.json`) — same tickers already added to the user's "Claude"
Webull watchlist. Everything else (entry/exit rules, fill model, sizing, fees) is
identical to the 3-symbol run. `run.py` is copied unchanged; only `config.json`'s
`symbols` list differs.

Full symbol list: CTVA, HAL, SUPN, DOW, RIOT, EBAY, BG, MUR, VNOM, MIAX, AKR, HASI,
HLF, MO, CELC, DAN, COLM, CWEN, POR, LXU.

## Why this set, and what it actually tests

The user's Webull "Claude" watchlist has 25 tickers total; only these 20 came from
yesterday's scan (the other 5 — XPEV, BILI, IREN, AFRM, PDD — predate that scan and
were never evaluated by `core.pullback_reversal`, so they're excluded here).

**Important scope caveat**: these 20 are not "every ticker that matched the raw
technical screener." The live pipeline's 2026-08-24 run reviewed 26 candidates that
passed the screener + sector cap + earnings buffer, and the Decision Agent (Claude,
using fundamentals/news/catalyst research) selected 20 of those 26 as its final
ranked picks. So this backtest tests **the technical pattern plus that downstream
curation** — closer to "would trading what the live system actually recommends have
worked" than a pure test of the technical pattern alone. That's arguably the more
useful question to answer first (it's what you'd actually trade), but it's a
different question from the 3-symbol run's more isolated pattern test, and results
from the two should not be blended together as if they're the same test.

## Confirmed strategy interpretation

Identical to the 3-symbol run — see
`runs/2026-08-25_HAL-MO-EBAY_pullback-reversal_1Day/notes.md` for the full entry/exit
rule derivation, indicator formulas, and disclosed deviations from this skill's
canonical EMA/ATR seeding. Summary:

- **Timeframe**: 1Day bars, feed=sip, adjustment=split
- **Date range**: 2016-01-01 through 2026-08-25 (warmup: first 127 bars per symbol,
  per `core/pullback_reversal.py::MIN_BARS_FOR_SCREENER`)
- **Entry**: `core.pullback_reversal.detect_pullback_reversal` — EMA200 uptrend >=5%
  over 126 bars, price within -12%/+8% of EMA200, <=15% consolidation range over 15
  bars, >=3% bounce off that window's low, not extended above the 60-bar volume
  profile's value area
- **Stop/target**: `core.trade_plan.compute_trade_plan` — swing-low/EMA-anchored
  stop refined to nearest support, Fibonacci 1.618 extension target floored at 3:1 R:R
  and refined to nearest resistance
- **Fill model**: `next_open` (signal on bar T close, fill at bar T+1 open), 5 bps
  slippage, $0 commissions
- **Exit**: `core.trade_plan.resolve_trade_plan_outcome` — up to 30 trading days,
  stop checked before target on same-bar conflicts
- **Position sizing**: 4% of current equity risked per trade (matches
  `config/settings.py::risk_per_trade_pct`), independent $100,000 account per symbol
- **Benchmark**: buy-and-hold of the same symbol, same assumptions

## Additional known limitations for this 20-symbol set

- **Short-history tickers**: MIAX IPO'd August 2025 (~1 year of data); CTVA and DOW
  are 2019 DowDuPont spinoffs (~7 years). The CLI fetch returns whatever history
  actually exists per symbol — these tickers will simply contribute fewer or zero
  signal occurrences, which is expected, not a bug. Don't read "zero trades" on a
  short-history ticker as the pattern failing; it may just never have had 127+ bars
  in an eligible configuration.
- **Comparing against the 3-symbol run**: don't average results across both runs as
  if they're one larger sample — they answer different questions (raw pattern vs.
  pattern+curation) and should be reported and read separately, then compared.
- All other limitations (survivorship bias, equity-curve granularity, excluded
  trading-activity fees) are identical to the 3-symbol run — see its notes.md.

## How to actually run this

Not yet executed — needs real Alpaca credentials this sandbox doesn't have.

```bash
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
export ALPACA_QUIET=1
alpaca doctor   # confirm auth before fetching

for SYMBOL in CTVA HAL SUPN DOW RIOT EBAY BG MUR VNOM MIAX AKR HASI HLF MO CELC DAN COLM CWEN POR LXU; do
  alpaca data bars \
    --symbol "$SYMBOL" \
    --start 2016-01-01 \
    --end 2026-08-25 \
    --timeframe 1Day \
    --feed sip \
    --adjustment split \
    --quiet > "raw/bars_${SYMBOL}.json"
done

python run.py
```

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
