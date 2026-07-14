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

- **Market Data Agent** — Alpaca bars for the whole universe scan, screened by
  `core/pullback_reversal.py` (a pullback into a rising 200-day EMA that has
  stabilized and shown an early bounce, not extended above its own volume
  profile's value area), calibrated directly against a real trade.
- **Research Agent** — FMP fundamentals/earnings-beat-miss-history/quarterly
  growth/analyst ratings + 6-12 months of news, called only on the
  post-screener/post-sector-cap candidate pool, never the full universe.
- **Portfolio Agent** — Webull SDK positions/balance/orders. `place_order()`
  defaults to `dry_run=True`; flipping to live execution is a deliberate,
  separate decision.
- **Decision Agent** — Anthropic API. Reads the Research Agent's fundamentals/
  news context and IS the ranking/selection step — it picks the final watchlist
  from the candidate pool, it doesn't just polish an already-decided score
  (there is no score anymore). Never recomputes the technical screener's
  numbers, sector cap, or trade-plan stop/target.

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
python pipeline.py                               # full pipeline, all 4 agents
```

## Running via GitHub Actions

`.github/workflows/scan.yml` runs the full pipeline on GitHub's cloud runners —
no local machine needed. Trigger manually from the repo's **Actions** tab
("SwingFinder Scan" → **Run workflow**, optionally setting `limit`/
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

## Known gaps

- The technical screener (`core/pullback_reversal.py`) hasn't been
  walk-forward validated — treat its output as a candidate filter, not a
  proven signal.
- `core/sector_cap.py`'s stabilization logic was reviewed for correctness
  but never validated against a confirmed source spec (no prior
  implementation existed to check against).

## Verification status

All four agents have been run live end-to-end (real Alpaca bars, real FMP
`/stable/` responses, real Webull account, real Anthropic synthesis) against a
small 5-ticker test universe — not just imported/compiled. Before a full
945-ticker run, watch API usage/cost: FMP calls scale with candidate-pool size
(`pipeline.py::CANDIDATE_POOL_SIZE`, several calls/ticker), and the Decision
Agent makes one Anthropic call per run (not per ticker).
