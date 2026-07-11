# ML edge confidence: research notes

Evaluates four suggested tools/techniques (Qlib, Lopez de Prado meta-labeling,
ml4t, vectorbt) against the actual confidence implementation in this repo, not
the technique in the abstract. Findings below are grounded in
`core/ml_forecast.py`, `core/ml_tracking.py`, `pipeline.py`, and
`agents/decision_agent.py` as they exist today.

## Current implementation, as it actually behaves

`ensemble_ml_forecast()` (`core/ml_forecast.py:301`) trains a fresh RF +
GBM per ticker per run and produces a `confidence` field:

```python
dir_conf = max(pct_correct_dir - 0.5, 0.0) * 50.0
r2_adj = max(1.0 + rf_raw_r2 * 5.0, 0.3)
confidence = round(dir_conf * r2_adj, 1)
```

Observed values in `ml_predictions.csv` are `0.1`–`1.7` — this checks out
against the formula: daily-bar RF/GBM `R²` on 5-day-forward returns is
routinely at or below zero out-of-sample, so `r2_adj` sits near its `0.3`
floor, and test-set directional accuracy rarely clears 55%, capping
`dir_conf` around 2–3. The formula's own ceiling (100% direction accuracy,
strongly positive `R²`) is in the 40s, but nothing in the observed data gets
close. **This "confidence" is a rarely-populated, unbounded-in-theory,
tiny-in-practice number** — not the 0/1-10/10-20/20% bucket scheme implied in
the framing; no such bucketing exists in this codebase.

More importantly: **the confidence value is computed and logged, but nothing
downstream reads it.**

- `evaluate_ml_edge_score()` (`core/ml_forecast.py:349`) — the function that
  actually adjusts SmartScore — only looks at the *sign* of
  `ensemble_price - current_price`. It applies a flat `+10` /
  `-15` (`ML_EDGE_POSITIVE_BONUS` / `ML_EDGE_NEGATIVE_PENALTY`) regardless of
  whether confidence was `0.1` or `25`. Call site: `pipeline.py:102`.
- Position sizing in `DecisionAgent` (`agents/decision_agent.py:69-74`) is
  `risk_amount / abs(entry - stop)` — confidence never enters the formula.
- The only place a confidence-*adjacent* signal reaches the LLM is
  `ml_track_record` (`core/ml_tracking.py:141`, `compute_accuracy_summary`),
  which is a rolling directional-accuracy % over the last 60 *scored*
  predictions, aggregated across all tickers — not per-prediction, and gated
  behind `MIN_SAMPLE_SIZE = 15`. As of this run, the log
  (`ml_predictions.csv`) has ~10 unique dated prediction rows (duplicates
  from repeated pipeline runs same-day), all still unscored (`scored=False`,
  5-trading-day window hasn't elapsed) — so `ml_track_record` is currently
  producing `sufficient_data: False` on every run.

So: before reaching for a heavier technique, there's an existing gap — the
confidence number this pipeline already computes has no effect on scoring or
sizing. Any of the ideas below should close that loop, not add a second
disconnected number.

## Per-repo assessment

**Qlib** — the concept (IC / rank-IC, decay analysis) is the right lens:
directional accuracy alone can't tell you whether the *magnitude* of a
forecast tracks the *magnitude* of realized return, which is exactly what's
missing from `evaluate_ml_edge_score`'s sign-only check. But Qlib itself is a
poor infra fit here: it wants its own data provider (`qlib.init()`), Alpha158
handlers, a workflow YAML, MLflow experiment tracking — built for training
persistent models over a curated point-in-time data warehouse. This pipeline
trains a throwaway RF/GBM per ticker inside a GitHub Actions runner
(`.github/workflows/scan.yml`, manual `workflow_dispatch`) with no persisted
model state between runs. Importing Qlib to get `calc_ic()` would be a large
dependency for a computation that's `scipy.stats.spearmanr` /
`pearsonr(predicted_return, actual_return)` over the existing
`ml_predictions.csv` columns (`predicted_return_pct`, `actual_return_pct`)
once rows are scored — a ~10-line addition to `ml_tracking.py`, no new
dependency. **Borrow the metric, skip the framework.**

**Meta-labeling (López de Prado)** — the best-fit idea, and it maps almost
exactly onto infrastructure that already exists: `ml_predictions.csv` already
has `direction_correct` as a label once scored, and the confidence-adjacent
inputs (`rf_confidence`, `gb_confidence`, ensemble `agreement`, `r2_score`)
are already computed in `random_forest_forecast`/`gradient_boosting_forecast`
(`core/ml_forecast.py:173-298`) but discarded — `ensemble_ml_forecast`
doesn't even pass `agreement` or the individual `r2_score`s through to the
log today. A secondary classifier trained on
`{rf_r2, gb_r2, agreement, vix_regime, ...} → P(primary call correct)` would
replace the ad hoc `dir_conf * r2_adj` formula with something calibrated, and
its output probability is a principled position-sizing input in a way the
current flat `+10`/`-15` isn't.

**The blocker is data volume, not design.** A meta-model needs on the order
of hundreds of scored predictions to avoid overfitting; the log currently has
single digits of unique scored rows. This needs weeks-to-months of runs
accumulating before a meta-model is trainable — the right sequencing is:
(1) fix the log to capture the features a meta-model would need (it's
discarding `agreement`/individual `r2_score` right now), (2) let it accumulate,
(3) train the meta-model once `MIN_SAMPLE_SIZE`-scale data exists for it too.
Doing it now would mean fitting a classifier on ~10 rows.

**ml4t (Jansen)** — best used as a reference to sanity-check the existing
hyperparameters, not as a library. Two concrete things worth checking against
it: (1) `max_depth=4, min_samples_leaf=20` on ~1500-bar lookback windows
(`prepare_features`, `lookback=1500`) is a very shallow, heavily-regularized
tree for that much data — plausibly leaving accuracy on the table, plausibly
correctly conservative given how noisy 5-day-forward stock returns are; worth
a controlled sweep rather than assuming either direction. (2) the
book's calibration-curve approach (reliability diagrams via
`sklearn.calibration`) is the more rigorous version of the ml4t-style sanity
check for "does a 0.7-confidence bucket actually hit 70% of the time" — same
conclusion as the meta-labeling point: this pipeline currently has no
calibration step at all, it goes straight from raw R² to an ad hoc formula.

**vectorbt** — same verdict as Qlib: the specific operation wanted (group
predictions by confidence bucket, compare mean forward return / win rate per
bucket) is a `groupby` over `ml_predictions.csv` once it has volume, not a
vectorized backtesting engine. Pulling in vectorbt for that is disproportionate
dependency weight for a project whose entire ML stack today is
`scikit-learn` (see `requirements.txt`). Worth doing as a plain pandas
analysis, but it hits the same wall as meta-labeling: not enough scored rows
yet to say anything statistically meaningful about bucket-vs-return.

**FinRL / TradingAgents / FinRobot** — agreed to skip. This pipeline's
four-agent structure (`README.md`) already *is* the decision/execution
orchestration layer these frameworks provide; none of them address signal
calibration, which is the actual gap.

## Recommendation, ranked by leverage vs. cost

1. **Wire the existing confidence into `evaluate_ml_edge_score` and position
   sizing** before adding anything new — right now the number is computed and
   ignored. Even a simple confidence-scaled bonus (e.g. scale the ±10/-15 by
   `confidence`, clipped) closes the most obvious gap for near-zero cost.
   *(Not applied here — this changes live scoring/trading behavior on a
   production pipeline and shouldn't land without an explicit go-ahead.)*
2. **Stop discarding `agreement` and individual `r2_score` at the log layer**
   (`ml_tracking.py` / `record_predictions`) so the features a future
   meta-model needs are being banked starting now, not from whenever the
   meta-model work begins.
3. **Add IC / rank-IC to `compute_accuracy_summary`** — cheap, no new
   dependency, and a strictly more informative signal than directional
   accuracy alone for the same `ml_track_record` payload already fed to the
   Decision Agent's prompt.
4. **Defer meta-labeling and confidence-bucket backtesting** until the
   scored-prediction log has enough volume (hundreds of rows) to train/
   validate on without overfitting. Design now, execute later — happy to
   sketch the meta-model's feature/label schema in detail when that's the
   next step.
5. **Use ml4t as a reference**, not a dependency, for a hyperparameter sweep
   and a calibration-curve check on the existing RF/GBM.

## What I did not do

No production code (`core/ml_forecast.py`, `pipeline.py`,
`agents/decision_agent.py`) was changed. Item 1 above changes what SmartScore
does for real trade candidates and item 2 changes the schema of a log this
pipeline treats as durable history — both are one-line-ish changes but with
real consequences, so they're flagged for a decision rather than applied
silently.

## Update 2026-07-10: walk-forward backtest results

Items 1-3 above were implemented (`ML_EDGE_CONFIDENCE_SATURATION`-scaled
SmartScore adjustment; `rf_r2`/`gb_r2` now exposed on `ensemble_ml_forecast`'s
return; `research/walk_forward_backtest.py` + `research/analyze_confidence.py`
built and run via `.github/workflows/ml_confidence_backtest.yml`). The
backtest covered 60 sector-balanced, price/volume-filtered tickers over 2
years (2,426 walk-forward predictions, 10 excluded as stock-split data
artifacts — see below). Real results, not the synthetic-data smoke test this
doc originally shipped with:

| metric | value |
|---|---|
| IC (Pearson, predicted vs. actual return) | -0.0067, p=0.74 |
| Rank-IC (Spearman) | 0.0257, p=0.21 |
| Overall directional accuracy | 51.2% (n=2,416) |
| Confidence bucket win rate, low→high | 51.5% → 51.0% → 50.4% |
| Confidence bucket mean return, low→high | 0.62% → 0.56% → 0.40% |

**Conclusion: this ensemble, on this feature set, has no statistically
significant edge, and confidence does not separate good calls from bad ones
— if anything the trend is flat-to-inverted.** Neither p-value clears 0.05;
51.2% direction accuracy on n=2,416 is not distinguishable from a coin flip.

**This changes the recommendation from the original draft above: do not
build the Phase 3 meta-model.** Meta-labeling amplifies a real primary
signal — training a secondary classifier on top of a primary model that's
statistically indistinguishable from guessing would fit noise and produce a
false sense of calibrated confidence, which is worse than the current
crude-but-honest formula. The Phase 0 confidence-scaled SmartScore adjustment
is still reasonable to keep (proportional beats flat-regardless-of-confidence
as a matter of principle), but shouldn't be described as "confidence-weighted"
in any predictive sense until the underlying signal actually has edge.

### Bug found along the way: unadjusted price data

10 of 2,426 rows showed implausible returns (e.g. ARQQ: +2471% over 5 days).
Root cause: `agents/market_data_agent.py`'s `StockBarsRequest` never set
`adjustment`, so Alpaca returned raw (split-unadjusted) bars by default. ARQQ
did a 1:25 reverse split around Nov 2024; the pre/post-split prices in the
same fetched window created a fake discontinuity. This isn't backtest-only —
it corrupts the **live pipeline** too, silently, whenever a shortlisted
ticker splits during its 1500-bar training lookback or during the 5-day
scoring window in `ml_tracking.py`. Fixed by adding
`adjustment=Adjustment.SPLIT` to the request (dividend adjustment
deliberately left out — swing entries/stops/targets need actual tradeable
prices, and dividend-adjusting would shift historical prices off of that).

### What would actually move the edge needle (separate from confidence calibration)

The confidence question and the "does this model have real edge" question
turned out to be the same question, and the answer to the second is
currently no. Improving *that* is a bigger lift than anything above — it's
model/feature/architecture work, not a calibration fix. In rough order of
expected leverage:

1. **Pool training data cross-sectionally instead of one bespoke model per
   ticker.** `prepare_features`/`random_forest_forecast` currently train a
   fresh model per ticker on ~300-1500 rows of that ticker's own history —
   daily-bar single-stock return series are extremely low signal-to-noise
   and heavily autocorrelated, so the *effective* independent sample size is
   much smaller than the row count suggests. Training one model on pooled
   (ticker, date) samples across the universe (rank/z-score-normalized
   features so they're comparable across price levels and tickers) turns
   2,400 samples from 60 tickers into tens of thousands, and lets the model
   learn genuinely cross-sectional patterns (relative momentum, relative
   mean-reversion) instead of trying to fit a wiggly per-stock time series to
   a few hundred rows. This is the actual substance behind Qlib's
   Alpha158/360 handlers (not their IC utility, which is trivial to
   hand-roll) — a shared panel model, not a per-instrument one. It's also a
   real architecture change: the live pipeline currently trains from
   scratch every run with no persisted model; a pooled model would need to
   be trained periodically (e.g. via a scheduled job reusing the
   walk-forward harness's data collection) and loaded, not retrained, on
   each pipeline run.
2. **Feed in signals the pipeline already computes but doesn't hand the
   model**: `core/relative_strength.py` (RS rank vs. SPY),
   `core/multi_timeframe.py` (weekly/daily alignment), and
   `core/volume_profile.py` (price vs. point-of-control) are all already
   computed for SmartScore/pattern purposes elsewhere in the pipeline but
   never reach `prepare_features`. Cheapest possible test before the bigger
   architecture change above — wire 2-3 of these in as additional columns
   and re-run the walk-forward harness to see if IC moves at all.
3. **Reconsider the prediction target.** Raw 5-day forward return is mostly
   market-wide beta noise that a technical-only feature set has no way to
   predict anyway; predicting *excess* return over SPY or sector may isolate
   the idiosyncratic component the feature set could plausibly explain.
4. **Set a realistic bar.** Published equity factor research treats IC in
   the 0.02-0.05 range as meaningful for daily/short-horizon signals — not
   the 0.3+ that "confidence" implicitly promised. Any of the above should
   be judged against that bar, and re-validated through
   `research/walk_forward_backtest.py` before being trusted, given how easily
   in-sample R² alone (the original approach) produces a misleadingly clean
   picture.

None of this is committed to the pipeline as of this update — it's a
prioritized punch list, not yet executed. Item 2 is the cheapest to try
first and would tell us quickly whether it's worth doing item 1's bigger
rework at all.

## Update 2026-07-10 (later same day): item 2 result — real but modest

Item 2 was implemented: `rs_20`/`rs_60` (relative strength vs. SPY),
`weekly_uptrend` (weekly EMA20-vs-EMA50, vectorized), and `vwap_position_60`
(a rolling-VWAP proxy for the point-of-control histogram) were added to
`prepare_features`, and the walk-forward backtest was re-run on the same 60
tickers / 2-year window / 2,426-prediction sample as the first run.

Raw result: IC (Pearson) jumped to 0.0715 (p=0.0004) and rank-IC to 0.0453
(p=0.0255) — both now statistically significant, vs. -0.0067/p=0.74 and
0.0257/p=0.21 before. But the split-adjustment fix (separate PR) didn't
eliminate every large real move in the sample — ARQQ, RCAT, CXW, REAL, and
VRDN each had a genuine >50%-in-5-days swing in this window (verified: no
more split-discontinuity artifacts, just real small-cap volatility). Removing
those 8 rows for an apples-to-apples comparison with the first run's cleaned
number:

| metric | run 1 (baseline) | run 2 (+ RS/weekly/VWAP features) |
|---|---|---|
| IC (Pearson), cleaned | -0.0067, p=0.74 | 0.0332, p=0.10 |
| Rank-IC (Spearman), cleaned | 0.0257, p=0.21 | 0.0436, p=0.032 |
| Directional accuracy, cleaned | 51.2% | 51.7% |
| Confidence bucket win rate, low→high | 51.5%→51.0%→50.4% | 52.9%→50.6%→49.5% |

**Honest read: the new features produced a small, real improvement in
rank-IC (the more outlier-robust of the two metrics) that crosses the
significance threshold — but the Pearson IC improvement is partly
outlier-driven and doesn't clear it on its own, directional accuracy barely
moved, and the confidence buckets are still flat-to-inverted.** This isn't
"no edge" anymore, but it isn't a strong edge either — rank-IC ~0.04 is at
the low end of the "meaningful" range cited above, not comfortably inside
it.

This changes the item-1 recommendation from "not worth it, nothing to
amplify" to **worth pursuing now that there's something real, if weak, to
amplify** — pooled cross-sectional training exists specifically to make a
small effect like this detectable and usable with far more samples than any
single ticker's history can provide. Expectations should stay modest (aiming
for rank-IC in the 0.03-0.06 range, not a jump to strong directional
accuracy), and confidence-bucket calibration remains a separate, still-open
problem that this result doesn't resolve on its own — worth re-checking
after item 1, not assumed fixed by it.

## Update 2026-07-10 (still later same day): item 1 result — pooling made it worse. Stopping here.

`research/pooled_model_experiment.py` trained one shared RF + GBM on all 60
tickers' pooled (ticker, date) samples (7,422 rows in the held-out test
period alone — far more than any single ticker's history), with a strict
calendar-date train/test split (no ticker's later data could leak into
another's earlier test window). Result, vs. the per-ticker walk-forward's
best (and three-times-reproduced) numbers:

| metric | per-ticker walk-forward | pooled ensemble |
|---|---|---|
| IC (Pearson) | 0.0715 (raw) / 0.0332 (cleaned) | 0.0145, p=0.21 — not significant |
| Rank-IC (Spearman) | 0.0453, p=0.026 — significant | -0.0216, p=0.063 — wrong sign |
| Directional accuracy | 51.8% | 51.2% |

The pooled Gradient Boosting model alone was worse still: rank-IC -0.0247,
p=0.033 — *statistically significant in the wrong direction*, i.e. its
predictions were inversely related to what actually happened, not just
noise.

**Pooling did not amplify the per-ticker approach's weak-but-real signal —
it looks like it made things worse, or at best didn't help.** One caveat:
the evaluation isn't perfectly apples-to-apples. The per-ticker walk-forward
tests across ~45 different rolling time windows spread over 2 years; the
pooled model was evaluated on a single held-out block (the most recent ~20%
of the date range) — a noisier estimate that could reflect a specific hard
regime rather than a flaw in pooling itself. But that caveat is a reason for
humility about the *pooled* number, not a reason to trust it over the
per-ticker result, which has now reproduced identically across three
separate runs.

**Recommendation: stop here.** Don't persist a pooled model or change
`pipeline.py` to load-and-score instead of train-per-run — that's real
architectural cost for a technique that underperformed the simpler
per-ticker approach in this test. A properly rigorous pooled evaluation
(walk-forward the pooled model too, not just one holdout block) is more
engineering effort for a technique that already came back negative once;
not worth it given what's already been validated and shipped.

**What's actually been established and kept, from this whole line of
work:** the confidence-scaled SmartScore adjustment (the "Update
2026-07-10" section above, live in `core/ml_forecast.py`), and the
relative-strength/weekly-trend/volume-profile features feeding the
per-ticker ensemble (live in `pipeline.py` via `enrich_with_technical_
analysis`, giving a reproducible, statistically significant rank-IC of
~0.045). Confidence-bucket calibration is still unresolved — buckets remain
flat-to-inverted regardless of which of these approaches was tested — and
would need its own investigation if picked up again later.

## Update 2026-07-11: further techniques and data-source research

Two questions: are there other techniques/repos worth knowing about, and do
Alpaca/Webull/FMP already have data that could help beyond what's used
today? Findings grounded in what's actually wired into this pipeline right
now (`agents/research_agent.py`, `agents/market_data_agent.py`) rather than
the platforms' full capabilities in the abstract.

### Data this pipeline has access to but doesn't use as an ML feature

`ResearchAgent` (`agents/research_agent.py`) already calls FMP for VIX,
earnings calendar, fundamentals, analyst ratings, and news — but only as
qualitative context handed to the Decision Agent's LLM prompt, never as a
`prepare_features()` column. Same gap pattern as the RS/weekly-trend/
volume-profile features before item #2 fixed it — data already reachable,
never fed to the model. Concretely available via FMP but currently unused
anywhere:

- **Insider transactions** (`insiderTrades`) and **institutional
  ownership/13F changes** (`form13F`) — both have real, widely-replicated
  academic literature behind them as return predictors, and matter because
  they're a genuinely *different* information category from anything in the
  current feature set (which is 100% derived from price/volume). This is a
  stronger candidate than another technical-indicator variant, precisely
  because it isn't just another transformation of the same OHLCV signal
  everything else already extracts from.
- **Analyst estimate revisions** (`analyst`) — FMP exposes more than the
  single current-rating snapshot `get_analyst_ratings()` already pulls;
  revision *direction/momentum* over time is the more commonly cited
  predictive signal in the literature, not the point-in-time rating.
- **Congressional trading** (`senate`) and **futures positioning**
  (`commitmentOfTraders`) — lower-confidence, noisier alt-data signals, but
  free within the existing FMP subscription and worth a cheap test via the
  walk-forward harness before dismissing.

**Alpaca** (`agents/market_data_agent.py` only fetches bars today) has a
separate **News API** — 6+ years of historical news (Benzinga-sourced),
explicitly documented as usable for sentiment-model training — a real,
currently-untapped data category, not just more price history. Its
**Corporate Actions** endpoint is now largely redundant with the
`adjustment=Adjustment.SPLIT` fix already applied, but could still serve as
an explicit "avoid trading around this" flag similar to the existing
earnings buffer.

**Webull** (`agents/portfolio_agent.py`, brokerage/execution only today) has
no meaningfully new data for this purpose — its research endpoints
(analyst rating/target price, company profile) duplicate what FMP already
provides. It does expose tick-level order-flow/"footprint" data
(bid/ask imbalance, tape), which is a genuinely different data type, but a
poor fit for a 5-day-ahead swing signal — that granularity is built for
intraday/scalping horizons, not multi-day holds.

### Techniques

- **Swap `GradientBoostingRegressor` → `HistGradientBoostingRegressor`**
  (still scikit-learn, zero new dependency) — sklearn's own
  histogram-based implementation, directly inspired by LightGBM, generally
  faster and more accurate at this row-count scale. Free to try against the
  existing walk-forward harness before touching anything else.
- **LightGBM's `LGBMRanker` (`lambdarank` objective)** — trains directly on
  ranking rather than regression MSE, which matches what's actually being
  evaluated (rank-IC) better than a regression loss does. Mature, widely
  used, small new dependency. The more targeted, currently-adoptable version
  of a very recent idea: [LambdaRankIC](https://arxiv.org/abs/2605.00501)
  (May 2026) proposes directly optimizing rank-IC itself as a custom
  XGBoost objective and reports it beating both regression and NDCG-style
  ranking losses on real market data — no public implementation found, so
  it's a "watch this space" reference, not adoptable today, but `LGBMRanker`
  is the practical middle ground available right now.
- **Isotonic calibration** (`sklearn.calibration`, already in
  `requirements.txt` via scikit-learn) — the direct, low-effort answer to
  the still-open "confidence buckets are flat" problem from earlier updates.
  Isotonic calibration can overfit on small data, so this should wait for
  more scored predictions to accumulate (same data-volume caveat as the
  original meta-labeling discussion) rather than being tried on the current
  thin sample.
- **[When Alpha Breaks](https://arxiv.org/abs/2603.13252)** (Feb 2026) — a
  more sophisticated, DEUP-based uncertainty-for-ranking approach,
  conceptually the advanced version of the same confidence-calibration
  problem. Appropriate to revisit *after* isotonic calibration has been
  tried and shown to need more than a simple fix — not a first move given
  where this project actually is (rank-IC ~0.045, barely above the noise
  floor).
- Reference-only repos in the same spirit as ml4t before (study, don't
  import): [gonzalezcortes/Cross-Sectional-Equity-Returns-Prediction](https://github.com/gonzalezcortes/Cross-Sectional-Equity-Returns-Prediction),
  [tmro98/machine-learning-in-asset-pricing](https://github.com/tmro98/machine-learning-in-asset-pricing),
  [jerryxyx/AlphaTrading](https://github.com/jerryxyx/AlphaTrading).

### Recommendation

Given pooling already underperformed the simpler per-ticker approach and the
current signal is weak (rank-IC ~0.045), the highest-leverage next step is
a genuinely new information category — **insider transactions and analyst
revision momentum from FMP** — tested cheaply through the existing
`research/walk_forward_backtest.py` harness before any architecture
investment, the same discipline that caught pooling's failure and the
split-adjustment bug. The `HistGradientBoostingRegressor` swap is a
free companion test to run alongside it. `LGBMRanker` and calibration work
are reasonable next steps *if* the new data category moves rank-IC further;
not worth doing first on a signal this weak.

## Update 2026-07-11: insider/rating data — a wash; meta-labeling doesn't clear the bar

Insider-trading and daily rating features were implemented and backtested
(`agents/research_agent.py`'s `get_insider_trades`/`get_rating_history`,
threaded into `prepare_features`). The first live run looked like it
changed nothing at all — IC/rank-IC identical to four decimal places with
the pre-insider-data baseline — which turned out to be a real bug: the
insider-trading endpoint path (`search-insider-trades`, guessed from a doc
page URL and the FMP MCP tool's internal alias) 404'd on every one of the
60 tickers, silently caught and returned as an empty frame. Fixed to the
correct path (`insider-trading/search`, confirmed via a maintained
third-party FMP client's endpoint registry) and added visible error
logging so a wrong path can't silently produce a "successful"-looking
empty result again.

With the fix in place, the numbers did move (confirming real data was now
flowing) — but only within noise:

| metric | RS/weekly/VWAP baseline | + insider/rating data |
|---|---|---|
| Rank-IC (cleaned) | 0.0436, p=0.032 | 0.0407, p=0.045 |
| IC (cleaned) | 0.0332, p=0.10 | 0.0379, p=0.063 |
| Directional accuracy | 51.7% | 51.5% |

Same 8 known-real volatile-stock outliers as prior runs (ARQQ, CXW, RCAT,
REAL, VRDN) — no new data artifacts. **Insider/rating data is a statistical
wash: not measurably better, not measurably worse.** Left in the codebase
(harmless, optional, degrades gracefully) but not counted as a validated
improvement.

`research/meta_labeling_experiment.py` (new — trains a secondary classifier
on `{rf_confidence, gb_confidence, rf_r2, gb_r2, agreement_pct, confidence}`
→ `direction_correct`, reusing `walk_forward_backtest.py`'s output and
`pooled_model_experiment.py`'s time-based split) was run against this same
data: logistic regression AUC 0.52, random forest AUC 0.54 — both barely
above the 0.5 no-better-than-guessing line, and neither showed a clean
monotonic calibration curve on held-out data. **Honest read: with a primary
signal this weak (rank-IC ~0.04), there isn't enough real information for a
meta-model to meaningfully out-calibrate the current ad hoc formula.** This
is not a meta-labeling implementation failure — it's the expected result of
trying to calibrate confidence on top of a signal that barely clears
statistical significance in the first place.

**Where this leaves the project**: every technique tried after the
RS/weekly/VWAP feature set (pooled cross-sectional training, insider/rating
data, meta-labeling) has come back negative or null. That feature set —
rank-IC 0.045, reproduced identically across three separate runs — remains
the one validated, real improvement from this entire line of work. Further
gains likely require either a fundamentally different data category not yet
tried, or accepting that rank-IC ~0.04 may be close to this feature set's
and model class's practical ceiling for 5-day swing prediction. Recommend
treating this as a stopping point rather than continuing to iterate on
model/feature variations without a specific new hypothesis to test.
