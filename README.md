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

## Running via GitHub Actions

`.github/workflows/scan.yml` runs the full pipeline on GitHub's cloud runners —
no local machine needed. Trigger manually from the repo's **Actions** tab
("SwingFinder Scan" → **Run workflow**, optionally setting `limit`/`skip_ml`/
`skip_decision`), or via `gh workflow run scan.yml`. Results are written to
`results/latest.json` (plus a timestamped copy in `results/`) and committed
back to the repo automatically.

Requires these **Actions secrets** (Settings → Secrets and variables →
Actions → New repository secret) — same values as your local `.env`:

- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- `FMP_API_KEY`
- `WEBULL_APP_KEY`, `WEBULL_APP_SECRET`
- `WEBULL_TOKEN_CONTENT` — **base64-encoded** contents of your local
  `token.txt` (e.g. `C:\Users\ksher\.webull-mcp\conf\token.txt`). Encode with:
  `[Convert]::ToBase64String([IO.File]::ReadAllBytes("<path to token.txt>")) | Set-Clipboard`
  in PowerShell, then paste the clipboard as the secret value (base64 avoids
  any multi-line paste corruption in GitHub's secret form). The workflow
  decodes it back to a token file on the runner before each run. This token
  auto-refreshes on use, but if it ever lapses into `PENDING` (only happens
  after ~15+ days of total inactivity across every project that shares it),
  you'll need to re-run the local 2FA auth flow and update this secret with
  the newly re-encoded token contents.
- `ANTHROPIC_API_KEY`

`WEBULL_REGION_ID`/`WEBULL_ENVIRONMENT` aren't secret and are hardcoded in
the workflow (`us`/`prod`).

## Freshly-designed logic — still being validated

Two pieces of screening logic have no prior implementation to verify against
(no confirmed source doc, unlike most of this port) — reviewed 2026-07-09,
still worth treating as experimental:

- `core/sector_cap.py` — confirmed correct: runs after full-universe
  SmartScore scoring, on the already-ranked list, purely for sector
  diversification in the final shortlist.
- `core/deep_discount_filter.py` — stabilization checks gating the Fibonacci
  "Deep Discount" SmartScore bonus. Genuinely new/experimental (no prior
  version anywhere), intent confirmed reasonable (don't reward a "discount"
  reading unless there's evidence it's actually stabilizing, not still
  falling) but not yet validated at scale.

A third piece, a Market-Bias Buffer that shifted SmartScore's setup-
classification thresholds based on SPY's own trend, was removed 2026-07-09
after review — see core/smartscore.py's module comment for why (it was a
blanket, universe-wide classification gate that could exclude a ticker
before any of the more precise per-ticker signals — ML edge, patterns,
relative strength — ever got to evaluate it).

## ML edge confidence research

`docs/ml-edge-confidence-research.md` evaluates whether the ML ensemble's
`confidence` output (`core/ml_forecast.py`) is trustworthy and how to improve
it (meta-labeling, IC/rank-IC, calibration). `research/walk_forward_backtest.py`
generates historical (prediction, confidence, actual outcome) data offline —
much faster than waiting on the live `ml_predictions.csv` log to accumulate
one row per ticker per manual run — and `research/analyze_confidence.py`
computes IC/rank-IC and confidence-bucket calibration over it. Both require
`ALPACA_API_KEY`/`ALPACA_SECRET_KEY` like the live pipeline:

```
python -m research.walk_forward_backtest      # ~60 tickers, 2 years, writes research/walk_forward_results.csv
python -m research.analyze_confidence          # reads that CSV, prints IC/rank-IC + bucket report
```

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
