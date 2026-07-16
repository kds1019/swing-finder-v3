# SwingFinder Agents — standing instructions

This is a single-user personal trading tool. The pipeline (`pipeline.py` → `agents/*.py`) is
the only thing that matters here — there is no separate "ml edge" research project anymore.
An earlier phase of this project prototyped an ML-forecast/SmartScore system and ran extensive
walk-forward research on it; that entire system was found to have no demonstrated edge and was
deleted from the repo. If you find yourself reasoning about SmartScore, ML forecasts, or
walk-forward backtests, stop — none of that exists in this codebase anymore. The live system is:

```
Market Data Agent (Alpaca)   --+
Research Agent (FMP)          -+--> Decision Agent (Claude) --> Ranked trade plans
Portfolio Agent (Webull)      -+
```

## Running the pipeline

- Local: `python pipeline.py` (full universe) or `python pipeline.py --limit N --skip-decision` (fast smoke test).
- Production runs happen via the `.github/workflows/scan.yml` GitHub Actions workflow (`workflow_dispatch`).
  Passing an empty string for `limit` does NOT get you the full universe — the workflow's declared
  default (`"20"`) silently wins instead. To force a full-universe run, pass a `limit` value larger
  than the universe size (e.g. `"2000"`) so `.head(N)` just returns every row.
- The final output is `agents/decision_agent.py::FINAL_WATCHLIST_SIZE` ranked picks (currently 20).

## How to present pipeline results — ALWAYS

When reporting pipeline results (from a live run, from `results/latest.json`, or from a GitHub
Actions run), present **every** ranked pick returned (up to `FINAL_WATCHLIST_SIZE`), never just a
top-N subset or a condensed table. For each pick, show the full detail:

```
**N. TICKER** — Entry $X / Stop $X / Target $X / R:R X.XX | N sh, risk $X, value $X | Sentiment: X
Highlight: <research_highlight>
Rationale: <rationale>
Flags: <flags, semicolon-separated>
```

Also surface, before the per-ticker list: market bias, VIX/gate status, and — if present in the
output — `pick_track_record` (the system's own historical win rate) and any account-balance /
buying-power caveat the Decision Agent's `overall_recommendation` raises about position sizing not
being executable at current cash levels. These are not optional footnotes — the user has been
burned before by picks that look clean technically but come with a weak track record or unusable
sizing, and wants that surfaced prominently, not buried.

Do not default to a short "top 3/5" summary or a compressed markdown table — that is not what this
user wants, regardless of how a fresh session might otherwise choose to summarize a large result set.

## Account / risk configuration (do not change without explicit request)

- `config/settings.py::risk_per_trade_pct = 4.0` — user's chosen risk-per-trade percentage.
- `config/settings.py::excluded_tickers = ("HELP", "CYBN")` — the user's existing long-term
  holds (same company, renamed ticker); never swing candidates, always excluded pre-screener.
- `agents/decision_agent.py::MODEL = "claude-sonnet-5"` — **rejects any non-default sampling
  parameter** (`temperature`/`top_p`/`top_k` set to anything other than the API default returns a
  400). Do not add `temperature=0` or similar to try to stabilize output — it will break every
  Decision Agent call. There is currently no sampling-parameter lever for run-to-run determinism
  on this model.

## Verification discipline

Diagnose from real evidence (live run output, actual error messages), verify fixes against real
runs, and report full/complete results — never assume success or present partial results as done.
