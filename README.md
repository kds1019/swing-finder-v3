# SwingFinder Agents

A standalone 4-agent pipeline that replaces `swing-finder-v2`'s Tiingo data layer,
while porting its proven indicator/scoring logic so results stay comparable.
`swing-finder-v2` (the reference app, at `../swingfinder`) is untouched — read-only
reference for what was ported.

```
Market Data Agent (Alpaca)   --+
Research Agent (FMP)          -+--> Decision Agent (Claude) --> Ranked trade plans
Portfolio Agent (Webull)      -+
```

- **Market Data Agent** — Alpaca bars for the whole universe scan + SmartScore.
- **Research Agent** — FMP fundamentals/earnings/news/analyst ratings/VIX, called
  only on the post-SmartScore/post-sector-cap shortlist, never the full universe.
- **Portfolio Agent** — Webull SDK positions/balance/orders. `place_order()`
  defaults to `dry_run=True`; flipping to live execution is a deliberate,
  separate decision.
- **Decision Agent** — Anthropic API, pure synthesis on top of the deterministic
  numbers above. Never recomputes SmartScore or any filter.

## Setup

1. `python -m venv .venv && .venv\Scripts\activate` (or your preferred venv tool)
2. `pip install -r requirements.txt`
3. `copy .env.example .env` and fill in your keys:
   - `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` — this account is IEX (free-tier) feed
     only, not SIP; `MarketDataAgent` already requests `feed=DataFeed.IEX` explicitly.
   - `FMP_API_KEY` (financialmodelingprep.com). **Note:** `ResearchAgent` uses FMP's
     newer `/stable/` API, not the old `/api/v3/` — v3 is fully deprecated for keys
     created after 2025-08-31 (every v3 endpoint 403s with "Legacy Endpoint"). If
     your key predates that cutoff, v3 may still work for you, but `/stable/` works
     either way, so no config changes needed.
   - `WEBULL_APP_KEY` / `WEBULL_APP_SECRET` / `WEBULL_TOKEN_DIR` — reuse the same
     values already configured for `webull-openapi-mcp` so this project shares
     the existing authenticated token instead of re-triggering 2FA.
   - `ANTHROPIC_API_KEY` (console.anthropic.com)
4. Export the "SwingFinder Master Universe" Google Sheet to CSV and save it as
   `data/universe.csv` (columns: Ticker, Company Name, Exchange, Sector, Industry,
   Price, Market Cap ($M), Volume).

## Running

```
python pipeline.py --limit 20 --skip-decision   # fast smoke test — Alpaca only, no FMP/Anthropic
python pipeline.py --skip-decision               # full universe, still no FMP/Anthropic
python pipeline.py --skip-ml                     # skip ML forecast/MTF/RS enrichment (faster)
python pipeline.py                               # full pipeline, all 4 agents
```

## Freshly-designed logic — validate before trusting at scale

Three pieces of screening logic referenced by the project's planning doc do not
exist as code anywhere in `swing-finder-v2` — they were only described in a
separate Google Sheet ("SwingFinder Screening Parameters (v2 - Tuned)"). They
were built fresh here as reasonable interpretations of that sheet, **not
verified ports** — check them against the actual sheet before relying on them
at full scale:

- `core/sector_cap.py` — max N per sector, post-ranking.
- `core/deep_discount_filter.py` — stabilization checks gating the Fibonacci
  "Deep Discount" SmartScore bonus.
- The always-on Market-Bias Buffer in `core/smartscore.py` (in the reference
  app this is gated behind an opt-in "Smart Mode" toggle, default off).

## Known gaps vs. the reference app

- Stop/target/R:R calculation (`utils/target_calculator.py` in the reference)
  was not part of this initial port — `min_risk_reward`/`atr_stop_multiple` are
  configured in `config/settings.py` but not yet enforced anywhere.
- Pattern detection (bull flag, cup and handle, head & shoulders, etc.) and
  support/resistance clustering from `utils/indicators.py` were not ported —
  out of scope for this build.

## Verification status

All four agents have been run live end-to-end (real Alpaca bars, real FMP
`/stable/` responses, real Webull account, real Anthropic synthesis) against a
small 5-ticker test universe — not just imported/compiled. Before a full
945-ticker run:

1. Sanity-check SmartScore against known tickers (e.g. EMBJ, AR) once the real
   universe CSV is in place.
2. Watch API usage/cost on a full run — `enrich_with_technical_analysis` trains
   a fresh RF+GB model per shortlist ticker (`DEEP_HISTORY_LOOKBACK_DAYS=750`
   re-fetch), and the Decision Agent makes one Anthropic call per run (not per
   ticker), but FMP calls scale with shortlist size (several calls/ticker).
