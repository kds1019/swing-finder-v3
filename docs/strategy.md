# SwingFinder — screening strategy

This is the deliberate design intent behind the screener. It exists because the only
"why" previously recorded in the code was the calibration story in
`core/pullback_reversal.py`'s docstring (thresholds fit to one real trade, EMBJ), and
that made the actual strategy easy to lose track of.

## The trade

A **deep pullback inside an intact long-term uptrend that has stabilized and started to
turn back up.**

- **Long-term uptrend still intact** — the 200-day EMA is rising. The move up is real
  and, at the 200-day timescale, undamaged.
- **A real pullback, not a shallow dip** — price has corrected down to (or through) the
  rising 200-day EMA. This is *not* the Qullamaggie / momentum "buy the 3–5% dip in a
  leader near new highs" trade. The depth is the point: more distance between the entry
  and the prior highs is more room for the move back up, and a better reward-to-risk on
  the return trip.
- **Stabilized** — the pullback is no longer in free-fall (a loose recent-range bound).
  We are *not* waiting for a confirmed bounce — the calibration showed the best entries
  are with price still at the low.
- **Not chasing** — price sits a real margin *below* its own recent volume-profile value
  area, and no more than a few % above the 200-day EMA. If it has already run back up,
  the trade is gone.

The thesis is mean-reversion *within* an uptrend: a strong stock went on sale, the
selling has exhausted, and it is likely to resume the prior trend.

## What each filter is for

Thresholds below are the **calibrated** values (2026-08-31 — see "Calibration status").

| Filter (`core/pullback_reversal.py`) | Value | Purpose |
|---|---|---|
| `EMA200_MIN_UPTREND_PCT` | ≥ 5% over ~126 days | the long-term uptrend is real, not a dead-cat bounce |
| `PRICE_VS_EMA200_MIN/MAX_PCT` | −20% … +3% | lower bound = "a real, deep pullback"; upper bound = "not chasing". Edge is monotonic in depth. |
| `CONSOLIDATION_MAX_RANGE_PCT` / `..._LOOKBACK_DAYS` | ≤ 20% over 15 d | loose sanity bound — a wider range was, if anything, slightly better |
| `MIN_BOUNCE_OFF_LOW_PCT` | 0% (gate off) | calibration: best entries are price still at the low; kept as a constant for future re-tuning |
| value-profile gate (`price_vs_value_area_high ≤ −4%`) | ≤ −4% | a real margin below the value-area high, not merely "not above it" |

## Deliberately excluded

- **No volume-surge / "highest volume" gate.** A volume-surge filter catches stocks
  *after* the up-day has already happened — for a pullback entry that means you are late.
  Volume is verified by hand at entry instead. (A *contracting* volume pattern during the
  pullback is a useful quality tell and may be added as an informational field, but not
  as a filter.)
- **No "near 52-week high" / RS-line-at-highs filter.** Buying the deep pullback is the
  whole strategy; a near-highs filter would exclude exactly the setups we want — and the
  calibration confirmed it (within 4% of the 52-week high *loses money* for this pattern).
- **No relative-strength floor, no `50-EMA > 200-EMA` gate.** Both were expected to help
  (falling-knife guard / trend-structure), both were contradicted by the calibration — a
  deep pullback *is* relative weakness and *is* usually a 50/200 cross. See "Calibration
  status".

## Trade management (`core/trade_plan.py`)

Calibrated the same way as the screener — against the labelled dataset, via
`research/exit_analysis.py`. The finding was decisive and the exit change matters
**more than the entry recalibration** (avg R +0.20 → +0.36 vs +0.13 → +0.20):

- **Entry**: the close of the signal bar.
- **Initial stop**: `min(10-day swing low, EMA20 − 1.3·ATR)`, refined to the nearest
  support cluster within 3·ATR. Position size is `risk_per_trade_pct` of equity ÷
  `(entry − stop)`.
- **Exit — trailing stop.** Hold the initial stop until price reaches **entry + 2R**,
  then trail the stop at **(running peak high − 1R)**, never loosening. No fixed
  profit target — the Fibonacci "target" is a hard ceiling only.
- **Max hold**: 30 trading days, then mark to close.

Why not a fixed target: every fixed target tested (1R–3R) was *strictly worse* than the
trail (PF 1.13–1.21 vs 1.71), every year. This setup is fat-tailed — winning trades
average ~12R of favourable excursion — so capping them destroys the edge. A +2R/give-1R
trail also turned 2022 (the one losing year under a fixed target) positive.

Known limitation: this is a daily-bar simulation. Real intraday whipsaw and fill
slippage will shave some off the +0.36 R; there is margin over the fixed-target +0.20.

## Universe floors (`config/settings.py` → `core/universe.py`)

- `price_min` / `price_max` — avoid sub-$10 market-structure noise; price ceiling is a
  legacy crude-liquidity proxy, up for review now that the two floors below exist.
- `min_volume` (shares) + `min_dollar_volume` (Price × Volume) — liquidity. Dollar volume
  is the meaningful unit; the share count is kept as a secondary floor.
- `market_cap_min_musd` — cut micro/small caps whose "pullback" is disproportionately the
  start of a dilution spiral or a news collapse that gaps through the stop. A
  deep-pullback entry is already buying weakness, so the company needs enough size that
  the weakness reads as a correction.
- `sector_cap` — concentration control on the output (`core/sector_cap.py`).

## Intended architecture: screen → setup → decision

Three layers, currently partly fused:

1. **Screen** (`core/universe.py` + `core/pullback_reversal.detect_pullback_reversal`) —
   necessary conditions, calibrated from data: liquidity / price / market cap floors,
   200-EMA rising, price in the −20%…+3% band, not extended above the value area. A wide
   net by design. The candidate pool is then ordered **knife-risk tier first**
   (`core.pullback_reversal.classify_knife_risk` → stabilising / forming / still_falling),
   deepest-pullback within a tier, so that when more names match than fit the pool /
   pre-research sector cap, the still-falling ones are what gets squeezed out — not the
   stabilised-but-shallower setups (a pure deepest-first sort did the opposite). Depth
   stays the in-tier tie-breaker (the calibration's one real technical gradient).
2. **Decision** (`agents/decision_agent.py`) — the soft layer. Two independent axes:
   (a) **has the pullback found support** — `support_status` confirmed / forming /
   still_falling. Anchored to the pre-computed `KnifeRiskTier`
   (`core.pullback_reversal.classify_knife_risk`, from `DaysSincePullbackLow`, `HigherLowPct`,
   `CloseVsEMA20Pct`, `Last5dReturnPct`) — the Decision Agent starts from that tier and
   overrides it only with a stated reason, so the "has it stopped falling?" read uses one
   fixed definition every run instead of being re-derived ad hoc. Cross-checked against
   `RangeContractionRatio` / `DownUpVolumeRatio` / recent returns. A `still_falling` ticker
   is excluded/bottom-ranked regardless of fundamentals. The calibration showed this must
   NOT be a hard screener gate (a bounce requirement hurt expectancy; the −1R stop caps a
   failed entry) — and that the tier is a *weak* predictor individually (a fitted model
   scores AUC ~0.59); its value is the rate of bad entries for a discretionary trader
   taking a handful of positions, and consistency of definition, not precision.
   (b) fundamentals / earnings / catalyst / portfolio fit — the ranking. The Decision Agent
   **ranks every candidate it receives**; `pipeline.py` then applies the 3-per-sector
   diversification cap to that ranking (keeping each sector's 3 highest-ranked names, not
   the 3 the screener happened to surface first) and the top `FINAL_WATCHLIST_SIZE` survive.
   Includes an **analyst price-target revision** signal (`AnalystRating.targetRevisionRecentPct`
   = last-month vs last-quarter avg target, from FMP `price-target-summary`): analysts actively
   cutting targets is a headwind regardless of the catalyst story, and price already at/above
   the latest average target means little upside left. Added 2026-09-01 after a manual TipRanks/
   Zacks-style vetting pass caught it flagging names the fundamentals-only read rated positively
   (ON: targets −29% in a month, +8% upside left).
3. **Exit** (`core/trade_plan.py`) — the +2R trailing stop (see "Trade management").

Why "find support" is soft not hard: the screener optimises average expectancy over
thousands of instances, where the trailing stop already handles falling knives. A
discretionary trader taking ~6 positions cares about the *rate* of bad entries, which is a
judgement call with a chart in front of you — exactly what an LLM Decision Agent, given the
recent-price-action fields, can do and a fixed threshold can't.

## Calibration status

**Done (2026-08-31).** The setup-layer thresholds in `core/pullback_reversal.py` were
re-derived from data instead of from the single EMBJ reference trade.

Pipeline:

1. **Feature logging** (`core/pick_tracking.py`) — joins each screener measurement onto
   every logged pick, so `pick_outcomes.csv` is a standalone `(features → outcome)` table.
2. **Historical labelled dataset** (`research/build_calibration_dataset.py`) — replays
   bars for 500 sampled universe tickers, records the raw measurements + candidate
   features (ATR%, RS vs SPY, 52-week-high distance, 50-vs-200) at every bar in a wide
   net, labels each with the triple-barrier outcome. Output:
   `research/data/calibration_dataset.csv` (~191k rows, 461 tickers, 2021–2026).
3. **Analysis** (`research/analyze_calibration.py` → `research/calibration_findings.md`)
   — bins every feature against realised R-multiple / profit factor, walk-forward by year.

### What the data said

- The EMBJ-fit gates barely beat the wide net (PF **1.19 vs 1.18**). Re-derived gates
  reach PF **≈1.27** (train 1.24 / test 1.33 on a 2021–24 vs 2025–26 split).
- **Pullback depth is the signal** and it's monotonic — the −25%…−12%-below-EMA200 bins
  were the *best*; above +3% is dead. → band widened down (−20%) and tightened up (+3%).
- **Value-area-high** needs a real margin (−4%), not just "not above it".
- **The early-bounce requirement was backwards** — best outcomes were price still at the
  low. → gate removed.
- **Consolidation tightness didn't help** → loosened to 20%.

### What the data said NOT to do (against prior expectation)

- **No relative-strength floor.** RS laggards did *mildly better* for this setup — a deep
  pullback *is* relative weakness. A positive-RS gate would remove the best setups.
- **No `50-EMA > 200-EMA` gate.** `50 < 200` bins were better — that cross is the
  pullback's signature.
- **No near-52-week-high filter.** Within 4% of the high *loses money* here (PF 0.92).
- **No SPY-trend regime gate.** No measured benefit; 2022's damage is already covered by
  the VIX ≤ 20 gate in `pipeline.py`.
- **ATR% floor, POC gate:** measurable but marginal — they cut candidate volume without
  raising PF. Left out for now; revisit if the trade-plan / stop logic changes.

### Caveats / next

- ~1 market cycle of IEX history; survivorship-biased to today's universe; weak-RR trade
  plans excluded from the PF numbers.
- Thresholds were read off the bin tables and rounded, then train/test checked — not
  formally optimised. Re-run the calibration as more history and resolved live picks
  accumulate (`pick_outcomes.csv` now carries the features for exactly this).
- The trailing exit (above) came out of the same calibration and is now the live exit.
- **Weak-RR candidates are now dropped at the screener** (`config.settings.drop_weak_rr_candidates`,
  default on). ~32% of matches get a plan whose stop/target geometry fell below the R:R
  floor after the support/resistance refinement. An isolated portfolio backtest
  (`research/weak_rr_ab.py`) found these carry *negative* expectancy (PF < 1 in every year,
  in and out of sample — the higher hit rate doesn't cover the oversized losses relative to
  the compressed target) and that dropping them raises return and cuts max drawdown
  (−58% → −40%). Tightening the stop to the floor instead ("refloor") tested worse
  out-of-sample. `weak_rr` is still computed on every plan; the toggle just gates on it.
- **New gate added 2026-09-11: `EMA200_CURRENT_SLOPE_MIN_PCT` (-2%, 20-session EMA200
  slope).** Motivated by a live finding: on a real full-universe scan, 24 of 26 candidates
  had `EMA200_TREND_LOOKBACK_DAYS`'s 126-day check reading "uptrend" while their *current*
  20-day EMA200 slope had already turned negative — the 126-day window is deliberately slow
  (so a temporary flattening mid-pullback doesn't reject a genuine uptrend), but that same
  slowness let names that had actually rolled into a decline through too (PLAB was the
  concrete case: 126-day slope +16.9%, but 60-day -6.2%, 20-day -2.2% — clearly rolling
  over). SPY itself was positive across every window that same day, so this wasn't a broad
  market rollover, just the screener's own blind spot. An isolated portfolio A/B
  (`research/current_trend_gate_ab.py`) tested several floors and found -2% wins in *every*
  window — full period +122% vs +89% baseline, 2021-2024 train +83% vs +66%, 2025-2026 test
  +21% vs +13%, and even reduces damage in the 2022 bear year (-9% vs -14%, PF 0.95 vs
  0.68). Stricter floors (0%, requiring flat-or-better) tested WORSE despite sounding more
  intuitive — a mild negative slope is the normal signature of being mid-dip; only an
  outright breakdown should reject. `EMA200CurrentSlopePct` is now a real gate reason
  (`"current_trend_rolled_over"`) in `detect_pullback_reversal()`, not just informational.
- **A second, longer (252-day/1-year) version of the same idea was tested and REJECTED —
  do not re-add without new data.** Motivated by ENPH the same day: 126-day slope +7.5%
  ("uptrend"), current 20-day slope -1.99% (just inside the -2% floor above, still passes),
  but the 252-day slope only +2.4% — a V-shaped recovery off a low ~5 months back, stalling
  at its own recent high, not a genuinely sustained trend. Tempting to also gate on this,
  but an isolated portfolio A/B (`research/long_horizon_gate_ab.py`, layered on top of the
  already-adopted -2% gate) made EVERY window worse at every threshold tested (0/1/2/3%) —
  full period +122%→as low as +45%, and the 2025-2026 test window flips from +21% to
  *negative* (-3% to -5%). It can't distinguish a recovery that's stalling from one that's
  genuinely continuing, so blocking both loses more than it saves, and it caught zero 2022
  bear-year signals at any threshold (no downside protection either). Instead, the 252-day
  slope is exposed as an INFORMATIONAL field to the Decision Agent
  (`core/trend_context.py`'s `ema200_long_slope_pct` → `TrendEMA200LongSlopePct`), with
  explicit prompt guidance (`agents/decision_agent.py`) to weigh a large gap between it and
  the 126-day `EMA200UptrendPct` against the actual research (News/EarningsHistory/
  IncomeGrowth) rather than a mechanical rule — this is a case where the isolated backtest
  said a hard gate genuinely doesn't work, so judgment is the right layer for it, not a
  fallback for a gate we didn't get around to building.

## Trend context & setup_type (`core/trend_context.py`) — Phase 1 + backtest

Motivated by a live case (2026-09-11, CRUS): the scanner flagged the short-term
higher-low/contracting-range/falling-down-up-volume signal (`KnifeRiskTier` /
`support_status`) while the ticker was still below both its 50-day and 200-day EMA, only
~30% retraced off its down-leg — a reversion bounce, not a confirmed pullback in an
uptrend, but the old single "confirmed support" read didn't distinguish the two. That's
possible even though `core.pullback_reversal`'s own gate requires a "rising" EMA200,
because that gate's uptrend read (EMA200 risen ≥5% over the last *126* sessions) is much
slower than this module's 20-session EMA200 slope check.

`core/trend_context.py` adds that faster-reacting read as a separate, informational axis:
- `compute_trend_state()`: EMA50/EMA200 (reused from `core.indicators.compute_indicators()`
  — same requirement as `core.pullback_reversal`'s own functions), price above/below each,
  EMA200 slope over the last 20 sessions (deliberately much shorter than the screener's
  126-day EMA200 check — the point is to catch a more current rollover/reclaim). Classifies
  `uptrend` / `downtrend` / `transitional`.
  **Switched from SMA to EMA on 2026-09-11** after a live case (RDW): SMA200's slope still
  read a mild uptrend while price had already fallen back below both EMA50/EMA200 and
  EMA200's own 20-session slope had gone flat — EMA reacts faster to a recent stall because
  it weights recent bars more heavily, exactly the property this read needs at the
  inflection points where it matters most. Checked live afterward across the 26 candidates
  from that day's scan: only 2 of 26 were within 1.5% of either EMA (a "could flip on a
  small move" zone) — RDW's case was a genuine outlier, not evidence of a systemic problem,
  but the fix (reacting faster) is still the right one for exactly that kind of case.
- `measure_swing_fib_retracement()`: the most recent major swing high/low over a 60-session
  window (vs. the existing 20-bar short-term Fib helper in `core/indicators.py`), % retraced
  from that swing high, and whether it's in the classic 38.2–61.8% zone.
- `classify_setup_type()`: `trend_continuation` (uptrend + in the Fib zone + the *same*
  stabilization signal used for `KnifeRiskTier == "stabilising"`) vs. `reversion_bounce`
  (same stabilization signal, but downtrend or transitional). One shared definition of
  "has it stabilised" for both `setup_type` and `support_status`, not two that could drift
  apart.

**Two-phase rollout, deliberately not done in one step:**
- **Phase 1 (shipped):** compute `TrendState`/`SetupType`/swing-Fib fields for every screener
  match (`agents/market_data_agent.py::scan_universe`), log them to `pick_outcomes.csv`
  (`core/pick_tracking.py::SCREENER_FEATURE_COLUMNS`), and attach them onto each final pick in
  `results/*.json` (`pipeline.py::attach_trend_context`, post-hoc in Python — NOT part of
  `agents/decision_agent.py`'s JSON contract, since `setup_type` is deterministic, unlike the
  LLM-judged `support_status`). Purely additive; doesn't affect the live screener gate,
  Decision Agent ranking, or position sizing.
- **Phase 2 (shipped, backtest-only):** `research/trend_context_backtest.py`, forked from
  `research/portfolio_backtest.py` so that script's numbers stay an untouched baseline. Tags
  every historical signal with `setup_type`; `trend_continuation` keeps the existing
  chandelier-style trail unchanged, `reversion_bounce` gets trailing disabled (pure fixed
  stop/target — the pre-2026-08-31 exit behavior, scoped to just this bucket) and
  `REVERSION_BOUNCE_SIZE_MULT` (0.5, an uncalibrated placeholder) applied to position size.
  Supports three candidate-selection-under-capacity modes (see `PRIORITY_MODE` in that file):
  `full` (default — hard priority trend_continuation > reversion_bounce > unclassified),
  `--depth-only-priority` (ablation — baseline's own tie-break, no priority), and
  `--reserved-slots[=N]` (up to N of `MAX_POSITIONS` reserved for trend_continuation only,
  rest filled depth-only). Reports win-rate/PF/avg-R **per bucket**, not just blended.
- **Phase 3 (not done):** wiring `setup_type` into the live Decision Agent's ranking or
  actual position sizing — deferred until Phase 2's numbers show the split is real, given the
  system's live track record (13.3% win rate / -2.63% avg return over 60 picks as of
  2026-09-11) means nothing new should get real size on a hypothesis alone.

### Phase 2 results (2026-09-11, cached universe, 2021-06-01 .. 2026-08-31)

Ran all three selection modes to separate "is the setup_type split real" from "is this
particular portfolio-construction rule for trading both buckets at once any good" — they
turned out to be two different questions with two different answers.

| | baseline (unbucketed) | `full` priority | `depth_only` ablation | `reserved-slots=2` |
|---|---|---|---|---|
| total return | +12.2% | +43.0% | +9.6% | +9.0% |
| max drawdown | -52.0% | **-78.5%** | -47.0% | -60.2% |
| trades | 1096 | 700 | 948 | 879 |
| trend_continuation: n / win% / PF | — | 272 / 39.3% / 1.32 | 2 (no signal) | 301 / 37.9% / 1.26 |
| reversion_bounce: n / win% / PF | — | 398 / 28.1% / 1.12 | 269 / 26.0% / 1.08 | 173 / 25.4% / 1.07 |

**The per-bucket split is real and robust.** `trend_continuation` beats `reversion_bounce` on
every metric in *both* runs that give it an actual sample (`full`: 39.3% win / PF 1.32;
`reserved-slots=2`: 37.9% win / PF 1.26) — consistent across two different selection rules,
not an artifact of one ordering. `reversion_bounce`'s PF sits at 1.07–1.12 across *all three*
variants regardless of how candidates are picked — also robust. This is the trustworthy
result: `setup_type` is measuring something real, and `reversion_bounce` is meaningfully
weaker than `trend_continuation`, matching the motivating CRUS case.

**The blended portfolio numbers are NOT reliable, and are a separate problem.**
`trend_continuation` is only ~3% of raw signals (1,311 of 44,033) vs. `reversion_bounce`'s 38%
— rare enough that pure depth-based selection (`depth_only`) almost never gives it a slot
(n=2), so getting *any* sample at all requires deliberately favoring it. But hard-favoring it
(`full`) let it claim disproportionate concurrent slots — likely correlated ones, given it's
a rare, narrow category — and blew up max drawdown to -78.5%, worse than doing nothing.
Reserving a fixed 2-of-6 slots (`reserved-slots=2`) tamed that (-60.2%) but didn't fully fix
it, and `full`'s standout +43% return did not reappear (+9.0%) — that number looks like a
concentration/variance artifact of unrestricted prioritization, not a repeatable effect.
**Conclusion: none of these three variants' blended return/DD should be read as a forecast of
live behavior** — a 6-position/3-sector-cap daily sim is too coarse an instrument to settle
how a live 3-5-name watchlist should split capital across the two buckets. That's a real,
separate, harder question than "is the split real," and is unresolved — treat it as a Phase 3
prerequisite, not something these numbers already answer.

**Caveats:** the swing-Fib lookback (60 sessions), EMA200 slope window (20 sessions) and
flat-band (0.5%), and `REVERSION_BOUNCE_SIZE_MULT` are first-cut defaults, not independently
calibrated the way the pullback-reversal thresholds above were — re-tune from
`trend_context_backtest.md`'s per-bucket numbers before trusting either bucket with real size.
The three backtest runs above are also survivorship-biased to today's ~480-ticker cache and
don't model intraday whipsaw or real fills, same as `portfolio_backtest.py`.

### Phase 3 (shipped 2026-09-11) — scoped to sidestep the unresolved prerequisite above

The portfolio-construction question above (how many *concurrent* slots to give
trend_continuation vs. reversion_bounce) is about a multi-position SIMULATION and is still
unresolved. Phase 3 didn't wait on it, because it doesn't need it: `pipeline.py` never calls
`agents/portfolio_agent.py::place_order` (confirmed by reading pipeline.py — `dry_run` is
accepted as a CLI arg but never threaded into an actual order call) — this system produces a
ranked, sized RECOMMENDATION list, not automatic concurrent execution, so "how many of each
bucket to hold at once" is the user's own manual call when they act on the list, not something
the code needs to arbitrate. Given that, Phase 3 was scoped to per-pick trade management,
which the robust per-bucket backtest result (above) already justifies on its own:
- `agents/decision_agent.py`: SYSTEM_PROMPT now describes TrendState/RetracementPct/
  InFibZone/SetupType and instructs ranking a comparable trend_continuation above a
  reversion_bounce-only candidate — the same LLM-judgment mechanism already used for
  support_status's confirmed/forming/still_falling tie-break.
- `pipeline.py::apply_trend_context_trade_management`: reversion_bounce picks get
  position_shares/risk_amount/position_value recomputed at `reversion_bounce_size_mult`
  (config/settings.py, default 0.5) of normal size — deterministic Python, overriding the
  Decision Agent's own numbers for just those three fields, not asked of the LLM (setup_type
  itself isn't part of its JSON contract either, for the same reason).
- `core/pick_tracking.py::score_due_picks`: now setup_type-aware — a reversion_bounce pick
  resolves against a pure fixed stop/target (trailing disabled), matching
  `research/trend_context_backtest.py`'s bucketed exit; this is the one that actually changes
  the system's own live track record going forward, since it changes how a real pick's
  outcome gets scored.
- `CLAUDE.md`'s pick-format convention gained `Setup:`/`Exit:` lines.

The screener gate itself (`core/pullback_reversal.py::detect_pullback_reversal`) is still
untouched — a ticker that doesn't qualify for either bucket (setup_type=null) still passes
through exactly as before, with default (trailing, normal-size) trade management.

## Short interest (`agents/research_agent.py::get_short_interest`) — added 2026-09-11

Motivated by wanting to know whether a pick's setup is being fought by short sellers — e.g. is
a reversion_bounce's "bounce" fragile short-covering, or is a trend_continuation's pullback
being pressed by shorts adding into it. Neither FMP nor Webull's OpenAPI expose this (checked
live: several plausible FMP endpoint names all 404; Webull's `financial_alert`/
`financial_indicators` don't carry it either). Data comes from Nasdaq's own public,
undocumented, no-key-required short-interest API (the standard bi-weekly FINRA-reported
settlement data) plus FMP's `shares-float` for percent-of-float.

**Known coverage gap, confirmed live (2026-09-11): Nasdaq's endpoint only covers Nasdaq-listed
tickers.** An NYSE-listed ticker (e.g. KEY, NEE, KMI — roughly half this project's universe,
which spans NYSE/NASDAQ/AMEX) returns `{"data": null, "message": "Short interest is only
supported for Nasdaq Listed stocks"}`, which `get_short_interest` correctly degrades to `{}`
for — but that means roughly HALF of all candidates will show no short-interest data, not just
illiquid/obscure ones. This wasn't caught by initial testing because both test tickers (AAPL,
CRUS) happen to be Nasdaq-listed. Options considered for full NYSE+NASDAQ coverage: NYSE has
no equivalent free public endpoint found so far; `financialdata.net`'s short-interest API
requires a paid key (401 without one); a from-scratch FINRA bulk short-interest file parser
(the bi-weekly files FINRA itself publishes, covering all exchanges) would be free but is real
new engineering, not a quick add. Also checked live (2026-09-11) and ruled out: Zacks'
`compare_stocks` tool CLAIMS "insider and institutional ownership with short interest" in its
own description, but the real response has no short-interest field for either AAPL or KEY
tested — the claim doesn't match live behavior; TipRanks has no short-interest-named tool
among ~65 checked; Webull app-side subscriptions don't carry over to the OpenAPI anyway (per
Webull's own docs), so upgrading the app wouldn't help even if the app UI shows it.
`decision_agent.py`'s prompt already instructs treating an empty ShortInterest as
"unavailable," never as "no shorts" — important given how large this gap actually is.
**Decision (user, 2026-09-11): leave the Nasdaq-only gap as-is for now** — user is
independently researching further options. Revisit if that turns up something, or if the gap
proves costly enough in practice to justify the FINRA-parser build.

Wired in: `enrich_shortlist()` fetches it per shortlist ticker; `decision_agent.py`'s prompt
weighs DaysToCover/ShortPercentOfFloat/ShortInterestChangePct against TrendState/SetupType
(the same number means different things in different setups — see the prompt for the exact
reasoning it's asked to apply) and sets `HeavilyShorted`/`ShortsAdding` flags with a required
bear-case callout; `pipeline.py::attach_short_interest` guarantees the raw numbers land on
every final pick in `results/*.json` regardless of whether the LLM's prose mentions them.

## Insider activity (`agents/research_agent.py::summarize_insider_activity`) — added 2026-09-11

The natural complement to short interest: does insider buying/selling agree or disagree with
where the shorts are positioned. The raw fetch (`get_insider_trades`, FMP's
`insider-trading/search`) already existed in this file — it was built to feed the deleted
`core.ml_forecast` system (see CLAUDE.md) and sat unused since. `summarize_insider_activity`
picks it back up: filters to genuine open-market transactions only (`P-Purchase`/`S-Sale`)
over a trailing 90-day window, excluding the transaction types that are routine compensation
mechanics, not a voluntary market decision (`A-Award` stock grants, `M-Exempt` option
exercises, `F-InKind` tax-withholding surrenders, `G-Gift`, `J-Other`, `D-Return`) — confirmed
live (2026-09-11) that these can be 80%+ of a ticker's raw Form 4 activity, which would drown
out the real signal if counted the same way.

Confirmed live on real 2026-09-11 candidates: WULF (heavily shorted, 31.9% of float) had 2
genuine open-market purchases on 2026-08-17, but ALSO 4 sales over the same 90-day window
worth far more (~$9.0M vs ~$0.1M) — net insider activity is actually strongly negative
despite those 2 purchases, a more complete and different picture than eyeballing just the
purchase rows would suggest. This is exactly why the summary does real date-filtered
aggregation rather than surfacing raw transaction lists — the full picture (`net_value`) can
disagree with what a partial read implies.

Wired in the same pattern as ShortInterest: `enrich_shortlist()` adds an `InsiderActivity`
column; `decision_agent.py`'s prompt treats it as complementary to ShortInterest (insider
buying into a heavily-shorted name argues the short thesis may be wrong; insider selling
alongside heavy/rising short interest reinforces rather than offsets the bear case) and sets
`InsiderBuying`/`InsiderSelling` flags; `pipeline.py::attach_insider_activity` guarantees the
raw counts/net_value land on every final pick in `results/*.json` regardless of the LLM's
prose. Caveat: filing-date lag (Form 4s can be filed up to a few days after the actual trade)
and the 90-day window are both first-cut choices, not independently tuned.
