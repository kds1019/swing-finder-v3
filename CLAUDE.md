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
**N. TICKER** — Entry $X / Stop $X / Target $X / R:R X.XX | N sh, risk $X, value $X | Sentiment: X | Catalyst: X | Support: X | Setup: X | Exit: X | Short: X
Highlight: <research_highlight>
Rationale: <rationale>
Bear case: <bear_case>
Flags: <flags, semicolon-separated>
```

`Catalyst:` is the Decision Agent's `catalyst_status` field (`recent` / `upcoming` / `none`) — always
show it, don't drop it for "recent"-only picks. A `none` catalyst on an otherwise-clean technical
setup is exactly the kind of thing the user wants visible, not smoothed over.

`Support:` is the `support_status` field (`confirmed` / `forming` / `still_falling`) — the Decision
Agent's read of whether the pullback has actually stopped falling. Always show it; a `still_falling`
that made it into the list at all is worth the user's scrutiny.

`Setup:` is the `setup_type` field (`trend_continuation` / `reversion_bounce` / `null`) — a
DETERMINISTIC, Python-computed read (core/trend_context.py), separate from `Support:`/
`support_status`, which only says whether the drop has stopped, not whether that's happening
inside an uptrend or a downtrend. `trend_continuation` = uptrend pullback in the classic
38.2-61.8% Fib retracement zone; `reversion_bounce` = the same short-term stabilization signal,
but in a downtrend or transitional trend (see docs/strategy.md's CRUS/RDW cases for why this
matters — a stock can clear the screener's slower 126-session EMA200 gate while its faster,
20-session EMA200 slope has already rolled over into a real downtrend). Backtested separately
per-bucket (docs/strategy.md's Phase 2 results):
`trend_continuation` showed a real, repeatable edge over `reversion_bounce` (win rate ~38-39% vs
~25-28%, profit factor ~1.26-1.32 vs ~1.07-1.12) across independent backtest runs. `null` means
neither bucket applied (e.g. the stabilization signal itself never fired, or it's an uptrend
pullback outside the Fib zone) — treat it like any other technically-clean-but-uncategorized pick.

`Exit:` is the `exit_mode` field (`trailing` / `fixed_target`), set from `setup_type`:
`reversion_bounce` picks get `fixed_target` (trailing disabled — treat the quoted `Stop`/`Target`
as real, fixed levels for a quick in-and-out; `position_shares`/`risk_amount`/`position_value`
for these are already sized at `reversion_bounce_size_mult` — normally half — of a normal pick,
not the full `risk_per_trade_pct`). Everything else (`trend_continuation`, `null`) gets
`trailing`: `Target` is a ceiling only, the live exit is the +2R-activated trailing stop, so
realised R:R normally lands below the quoted `R:R` — this is the unchanged pre-existing behavior.

`Short:` summarizes `days_to_cover` / `short_percent_of_float` / `short_interest_change_pct`
(e.g. "5.2d cover, 7.7% float, +2.8% 2wk" or "n/a" if the lookup failed) — real, bi-weekly
FINRA-reported short interest via Nasdaq's own public data (see agents/research_agent.py's
`get_short_interest`), inherently up to ~2 weeks stale, always show it when available. It cuts
both ways and needs the Decision Agent's own `rationale`/`bear_case`/`flags` for which reading
applies to THIS pick, not a fixed rule: elevated short interest fighting a `trend_continuation`
is a real headwind (smart money betting against the exact continuation the setup implies);
elevated short interest on a `reversion_bounce` can mean either fragile short-covering (stalls
once covering ends) or genuine squeeze fuel (accelerates the bounce) — read the `flags`
(`HeavilyShorted` / `ShortsAdding`) and `bear_case` for which one the Decision Agent judged for
that specific pick, don't infer a reading from the raw number alone.

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
