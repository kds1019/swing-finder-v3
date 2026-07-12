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

## Plan (not yet built): reframe the prediction target around Entry/Stop/Target

Everything above trains the model to predict a continuous 5-day return —
an inherently hard, noisy target, and the likely reason the achievable
rank-IC has topped out around 0.04-0.045 regardless of what features get
added. The one genuinely different idea not yet tried: stop predicting a
return number, and predict something tied directly to what actually
determines a trade's outcome.

**The reframe.** `core/trade_plan.py::compute_trade_plan()` already computes
a real `entry`/`stop`/`target`/`rr_ratio` for every candidate (swing-low/
EMA-anchored stop, Fibonacci-extension target refined against real support/
resistance). Instead of "predict the return," train the model to answer:
*given this specific entry/stop/target, does price hit target before stop?*
— a binary classification label, not a regression target. This is the
"triple-barrier" labeling method (Lopez de Prado) applied as the *primary*
model's target, not just meta-labeling on top of an existing regression (a
distinction worth being precise about — this replaces what the primary
model predicts, it doesn't add a second model on top).

Why this is worth trying before assuming rank-IC ~0.04 is the ceiling:
- Return magnitude is dominated by market-wide noise a technical feature set
  has little hope of explaining. "Will price reach *this specific,
  already-computed* level before *that specific* level" is a narrower,
  more concrete question — plausibly more learnable from the same features.
- It sidesteps the outlier sensitivity that repeatedly complicated
  evaluation this session (ARQQ/RCAT/CXW-style extreme moves distort a
  continuous-return metric; hitting-a-barrier-or-not is far less sensitive
  to how far past the barrier price ran).
- It produces a probability, not a return guess — which is what the "how
  would this get used in trading" discussion below actually needs.

**Model: LightGBM classifier**, trained on `hit_target_first` (1) vs.
`hit_stop_first` (0), as a companion change alongside the reframe rather
than a separate experiment — replaces `RandomForestRegressor`/
`GradientBoostingRegressor` with `LGBMClassifier`, same per-ticker training
shape as today, cheap to swap given it's still scikit-learn-API-compatible.

**How this would change trading usage, if validated:**
1. **Expected value instead of a ranking nudge.** `rr_ratio` is already
   computed; combined with `P(hit target)` this gives
   `EV ≈ P(target) × reward − P(stop) × risk` — an actual expected-value
   number to rank and filter candidates by, not an arbitrary score bump.
2. **A concrete veto signal.** "68% technically clean setup, but the model
   gives this exact entry/stop/target only a 35% chance of hitting target
   first" is a more specific, actionable flag than the current "predicted
   return leans negative."
3. **Position sizing — only after calibration is proven.** A well-calibrated
   `P(target)` could in principle inform position sizing (bet size scales
   with edge, Kelly-criterion-style). This is explicitly *not* a green
   light to wire that up on day one — it requires the same walk-forward
   validation discipline as everything else in this doc: bucket predicted
   probability into quintiles and confirm a 70%-bucket actually hits target
   first about 70% of the time, the classification analogue of the
   confidence-bucket check that kept catching problems all session. If that
   calibration check fails the way the original confidence formula did,
   this stays a ranking/filtering input only, same as today — the reframe
   doesn't get to skip the validation step just because it sounds more
   principled.

**Validation path**: extend `research/walk_forward_backtest.py`'s harness to
also record whether each historical trade plan hit target or stop first
within the historical window, train the `LGBMClassifier` variant, and reuse
`research/analyze_confidence.py::confidence_bucket_report()` (already
generalized to take an arbitrary probability column) to check calibration
— same infrastructure, new label and model. Not started; this section is a
plan to pick up later, not a result.

## Update 2026-07-11 (next day): the validated edge was a bug artifact — it's gone

While scoping the triple-barrier work above and tracing `prepare_features()`'s
row alignment carefully (needed to reuse its feature engineering correctly),
I found a real, pre-existing bug in `core/ml_forecast.py`, present since before
this research began: `random_forest_forecast`/`gradient_boosting_forecast`'s
live prediction step used `X[-1]` as "today's features." But `X` is built by
dropping the trailing `days_ahead` (5) rows of the feature table — those rows
don't have a known 5-day-forward return yet, needed only for *training*.
`X[-1]` is therefore not today's row; it's the row from **5 trading days
ago**, confirmed empirically with a synthetic 700-bar series where `X[-1]`'s
date traced back to bar 694, not the true last bar (699).

Mechanically, this meant the model's "prediction" was really re-deriving the
return from 5 days ago to today — a quantity already directly observable
from price history — and presenting it as a forecast of the next 5 days.
Correlating that backward-facing number against genuine forward returns is,
in effect, testing whether recent short-term momentum predicts the next few
days (a real but different phenomenon, and not what the RF/GBM ensemble was
actually being credited for).

**Fixed** (PR #15): `prepare_features()` now also returns `current_features`
— the true, untruncated last row's feature vector — and both forecast
functions use that instead of `X[-1]` for the live/current prediction. The
training data (`X_train`/`y_train`) was never affected by this bug; only the
single "what does the model predict right now" step was wrong.

**Re-ran the full walk-forward validation with the fix in place** (run #7,
same 60-ticker sample, same 2-year window): rank-IC dropped from
0.041–0.045 (significant, reproduced 3 times) to **-0.0086, p=0.67** —
statistically indistinguishable from zero. IC similarly collapsed to
-0.02, not significant. Directional accuracy: 51.6%, unchanged, still a
coin flip. Same outlier-cleaning applied as every prior run; not a data
artifact.

**This supersedes every "found a real edge" conclusion earlier in this
document.** The RS/weekly-trend/volume-profile features, the insider/rating
data test, and the meta-labeling test were all run against the buggy
prediction step — their relative comparisons to each other (pooling made
things worse than per-ticker, insider data was a wash, meta-labeling had
weak AUC) likely still hold as *relative* statements, but the absolute
conclusion "this ensemble has a small real edge, rank-IC ~0.04" does not.
Once the model is made to genuinely forecast forward rather than
re-describe the recent past, this feature set and model class shows no
detectable edge at all.

**Correctness note, not a reason to revert**: the fix is correct regardless
of this result — a live trading system silently forecasting from 5-day-old
data is wrong on its own terms, independent of whether that mistake happened
to look profitable in backtesting. Leaving it in place because it produced
better-looking numbers would have been keeping a bug for its side effects.

**Where this actually leaves the project**: back to a genuine "no known
edge" state for the continuous-return regression approach — worse than the
"stopping point with one validated improvement" conclusion from the
previous update, but a more honest one. This raises the priority of the
triple-barrier reframing plan above: it was already the identified
"next genuinely different idea" before this fix, and now there's no
regression-based edge left to lose by pursuing it. Also worth flagging: the
live pipeline has been running with this bug — historical `ml_predictions.csv`
entries and any `ml_track_record` statistics computed before this fix
reflect the same stale-feature prediction step, not a genuine forecast.

## Update 2026-07-11 (same day): triple-barrier reframe built, not yet validated

Built the plan above: `research/triple_barrier_walk_forward.py` + `research/
analyze_triple_barrier.py`, wired into `.github/workflows/ml_confidence_backtest.yml`
as new steps alongside (not replacing) the regression backtest. No result yet — this
records what was built and how, not a finding.

**Refactor first.** `core/ml_forecast.py`'s feature engineering (the RS/weekly-trend/
VWAP/insider/rating/VIX block, ~150 lines) was extracted out of `prepare_features()` into
a new `build_feature_table()`, so the triple-barrier script can reuse the identical,
already-battle-tested feature set instead of duplicating it. `prepare_features()` now
just calls `build_feature_table()` and adds its own days_ahead-forward-return label on
top — verified byte-identical output on synthetic data before and after the split
(same X, same current_features, same dates).

**Label generation** (`build_ticker_dataset()` in the new script): for each historical
as-of date (sampled every `step_days=3` bars — denser than the regression backtest's
step_days=10, since this script trains one model per ticker instead of retraining at
every step, so it can afford more labeled examples per ticker), compute
`core.trade_plan.compute_trade_plan()` on the df truncated to that date (real entry/
stop/target, same function the live pipeline uses — not a synthetic label), then walk
forward through the actual (already-historical) bars that follow via the same
`resolve_trade_plan_outcome()` `core/pick_tracking.py` uses to resolve live picks.
`expired_unresolved` rows (neither stop nor target touched within `MAX_HOLD_DAYS`=30)
are dropped — ambiguous, not a clean binary label.

**Model**: `LGBMClassifier`, one per ticker, trained on `direction_correct`
(target_hit=1/stop_hit=0) with a single time-based 80/20 train/test split — matching
"same per-ticker training shape as today's regression forecasters" from the plan, not
the regression backtest's per-step retrain loop (unnecessary here since each row already
carries its own historical as-of date via the truncated `compute_trade_plan` call).

**Tested against synthetic random-walk price data first** (same discipline as every
other script this session) before this went anywhere near real data: dataset
generation, LGBM training, prediction, AUC/point-biserial correlation, and the
`p_target` quintile-bucket calibration report (reusing `confidence_bucket_report()`
from `research/analyze_confidence.py`, column="p_target" — no changes needed there,
its column-name parameter already generalized it) all run cleanly end-to-end. AUC on
synthetic data came out ~0.35–0.5, as expected for a pure random walk with no real
target-vs-stop asymmetry to learn — not a finding, just confirmation the pipeline
doesn't produce nonsense before spending real API credits on it.

## Update 2026-07-11 (same day, real-data run): triple-barrier reframe also shows no edge

Ran `ml_confidence_backtest.yml` against real data (60-ticker sample, 760-day lookback,
same seed as every prior run) with both the regression backtest and the new
triple-barrier steps. Both came back null.

**Regression ensemble (unchanged code, fresh sample window)**: rank-IC -0.009,
p=0.66 — reproduces the post-fix null result from the previous update, not a fluke of
that one run.

**Triple-barrier classifier (first real-data run)**: 1,715 held-out predictions across
tickers that had enough labeled history to train on.
- **AUC 0.4876** — indistinguishable from a coin flip (0.5), if anything a hair below.
- **Point-biserial correlation 0.0026, p=0.9151** — no statistically detectable
  relationship between `p_target` and whether target was actually hit first.
- **Calibration bucket report is flat and non-monotonic**: win rates of 8.9%, 13.0%,
  12.5%, 7.9%, 9.9% across the five `p_target` quintiles — the top bucket (`p_target`
  0.05–0.95, i.e. the model's own most-confident calls) did not win more often than the
  bottom bucket (`p_target` ≈0). A well-calibrated classifier would show these numbers
  climbing in step with the bucket's own probability range; instead the model's stated
  confidence carries no information about the real outcome.
- **Base rate context**: target was hit before stop only 10.3% of the time overall —
  `compute_trade_plan`'s Fibonacci-extension target (floored at a 3:1 R:R) is a genuinely
  hard bar to clear within `MAX_HOLD_DAYS`=30, which the classifier's own historical
  training labels reflect (heavily imbalanced toward stop_hit/expired), likely part of
  why it couldn't find a learnable signal from this feature set.

**Conclusion**: the triple-barrier reframe was a reasonable next hypothesis — a
narrower, more concrete question than "predict a return number" — but it doesn't hold
up against real data either. Combined with the regression ensemble's null result, this
feature set (technical indicators + RS/weekly-trend/VWAP + optional insider/rating/VIX)
and these model classes (RF/GBM regression, LGBM classification) show no detectable
5-day-to-30-day forward edge on this universe, under either framing. Both negative
results are now independently confirmed, not just one.

**Where this leaves the project**: neither the original regression approach nor its
proposed replacement has cleared the bar this doc has applied to every claim throughout
this session.
Further progress most likely requires a fundamentally different information source —
this doc's "Update 2026-07-11: further techniques and data-source research" section
already flagged real, still-untried candidates that are a genuinely different category
from the 100%-price/volume-derived feature set both null results above share: FMP
institutional-ownership/13F changes, analyst estimate-revision *momentum* (not the
point-in-time rating already tested), and Alpaca's historical News API for sentiment —
rather than another model or label variation on the same technical inputs. Absent that,
the practical move is to accept this as the ceiling for now and leave
`evaluate_ml_edge_score`'s live SmartScore adjustment as a soft, unvalidated nudge
rather than building position sizing or hard filtering on top of either approach.

**Now run against real data — see the update directly below**, which reports the actual
AUC, point-biserial correlation, and `p_target` bucket report from that run. Per the
plan's own caution, even a promising AUC would have only justified ranking/filtering use
until the bucket report showed real calibration (a 70%-`p_target` bucket actually
resolving to target-hit-first about 70% of the time) — moot here since the bucket report
came back flat rather than calibrated.

## Update 2026-07-11 (later): analyst revision momentum feature added, not yet validated

Repo research (`stefan-jansen/machine-learning-for-trading`, `hudson-and-thames/mlfinlab`,
`dreyhsu/Meta_Labeling`) didn't turn up a usable off-the-shelf tool, but it did sharpen
which untried data category to try next: **analyst estimate revision momentum**, not
another cut of data already tested. Two things worth recording from that research:

- Insider trading (Form 4) is now *doubly* disconfirmed — our own FMP-based test found it
  "a wash" earlier in this doc, and an independent public analysis
  (`jvgalvin/Insider-Trading`) reached the same conclusion from scratch. Deprioritized.
- What `rating_score`/`rating_score_change_20d` already test is FMP's own daily
  fundamentals-ratio composite score (`historical-ratings`'s `overallScore`) — not real
  sell-side analyst revisions. The actual academically-supported "revision momentum"
  signal is about analyst *rating changes* over time, a genuinely different FMP endpoint
  we hadn't wired up.

**Built**: `ResearchAgent.get_grade_history()` (`agents/research_agent.py`) calls FMP's
`grades` endpoint — verified live before writing any parsing code (established discipline
after the earlier wrong-endpoint-path bug) — which returns real, dated, individual
sell-side analyst rating-change events (`date`, `gradingCompany`, `previousGrade`,
`newGrade`, `action` ∈ {upgrade, downgrade, maintain, initiate}). Confirmed the `limit`
query param isn't honored server-side (1,771 rows came back for a 10-row-limit AAPL
request spanning back to 2012) — handled with a client-side `.tail(limit)`.

New feature in `core/ml_forecast.py::build_feature_table()`: `analyst_revision_net_90d` —
net count of upgrade-minus-downgrade actions in a trailing 90-day window, using FMP's own
upgrade/downgrade classification directly rather than building a hand-rolled ordinal
mapping across grading firms' incompatible letter-grade scales ("Outperform" vs "Buy" vs
"Overweight" isn't directly comparable, but FMP has already resolved each firm's own
previousGrade→newGrade pair into a direction for us). "maintain"/"initiate" actions carry
no revision direction and are excluded rather than treated as zero-signal noise.

Threaded `grades_df` through `prepare_features()`, both regression forecasters,
`ensemble_ml_forecast()`, `research/walk_forward_backtest.py`,
`research/triple_barrier_walk_forward.py`, and `pipeline.py` — same optional-parameter
pattern as `insider_df`/`rating_df` throughout, degrades to "feature skipped" if
`FMP_API_KEY` is unset, same as everything else in this family.

**Tested against synthetic data first**: the feature engineering (net-signed rolling
window, correctly excluding maintain/initiate, correctly reindexed onto the daily feature
table) and both downstream paths (`research/walk_forward_backtest.py`'s per-step
retraining, `research/triple_barrier_walk_forward.py`'s labeled-dataset construction) run
cleanly end-to-end with fabricated grade events. Also confirmed `ensemble_ml_forecast`
degrades gracefully with `grades_df=None` and with an empty DataFrame.

**Not yet run against real data.** Next step: trigger `ml_confidence_backtest.yml` (no
workflow changes needed — it already calls these scripts, which now fetch and use grades
data automatically whenever `FMP_API_KEY` is set) and check whether rank-IC / AUC move at
all, and whether `analyst_revision_net_90d` shows up in the RF feature-importance ranking
these analysis scripts already print. Given this session's track record (two independent
null results so far), the honest expectation going in is that this also comes back null —
but it's a genuinely different, previously-untested data category, unlike most of the
variations tried before the triple-barrier reframe.

## Update 2026-07-12: analyst revision momentum — also null, third result in a row

Ran `ml_confidence_backtest.yml` against real data with the new `analyst_revision_net_90d`
feature live (60-ticker sample, 760-day lookback, same seed as every prior run).

**Regression ensemble**: rank-IC -0.0065, p=0.75 (previously -0.009, p=0.66) — essentially
unchanged from the pre-feature run, within noise of each other, not a meaningful shift.

**Triple-barrier classifier**: AUC 0.4937, n=1,715 (previously 0.4876) — still
indistinguishable from a coin flip. Point-biserial correlation 0.0005, p=0.9849
(previously 0.0026, p=0.9151) — if anything weaker. The `p_target` bucket report is still
flat/non-monotonic (8.7%, 13.4%, 11.4%, 8.7%, 10.2% win rate across quintiles).

**Caveat — statistical framing.** "No detectable edge" is not the same claim as "proven no
edge." All three p-values above are large (0.66-0.98) — the tests fail to reject "no
relationship," which is a much weaker statement than a small p-value confirming the
absence of one. A true weak signal, diluted or masked by the rest of the feature set,
would produce exactly this same result. What these three runs do rule out is a signal
*strong enough for this model/feature-set combination to surface* — not the underlying
existence of any signal in analyst revisions, insider trading, or the technical feature
set more broadly.

**Caveat — feature-level attribution was unavailable, not just uninspected.**
`research/walk_forward_backtest.py::backtest_ticker()` retrains a fresh RF/GBM at every
walk-forward step and only writes `RESULT_COLUMNS` (predictions/confidence) per step to
`walk_forward_results.csv` — it never captures `rf_model.feature_importances_` anywhere,
so there was no record to go back and inspect for this run or any prior one. (Correction
to this doc: an earlier draft of this paragraph said `pooled_model_experiment.py` already
computed importances "once per ticker" — wrong, it trains one single pooled model over
all tickers combined and only ever printed that one model's top 10.) See the next update
below for what was actually built to close this gap.

**Conclusion.** Three independent framings — return regression, triple-barrier
classification, and now both of those with a genuinely new sell-side analyst-revision
data category added — all show no *detectable* edge on this universe, in the weaker
statistical sense above. Practical takeaways:

- The "just add a new data source" pattern isn't a reliable path forward on its own,
  without also revisiting whether this model class (shallow, heavily-regularized RF/GBM
  over ~30 mixed features) can even detect a single weak signal buried among that many
  others.
- Recommend treating this as this session's stopping point for reflexively adding more
  data sources to the same model shape.
- If this line of work continues, do it via either (a) the feature-importance-isolated
  single-feature test described above, run *before* folding any new signal into the full
  feature set, or (b) stepping back to the conclusion from the "algo trading and financial
  ML" discussion earlier in this doc — this pipeline's value is more likely the
  rules-based screening and risk management than a still-undiscovered ML edge.

## Update 2026-07-12 (later): feature-importance instrumentation built, not yet run

Closed the attribution gap flagged above, in both model framings rather than just the
one originally scoped:

- **`research/pooled_model_experiment.py`**: now fetches `insider_df`/`rating_df`/
  `grades_df` per ticker (previously never wired in at all — this script's pooled dataset
  didn't include `analyst_revision_net_90d`, or insider/rating features, until now) and
  writes the *full* ranked RF feature-importance list to
  `research/pooled_feature_importances.csv` (`rank`, `feature`, `importance`), not just
  the top-10 console print from before.
- **`research/triple_barrier_walk_forward.py`**: captures each per-ticker `LGBMClassifier`'s
  `feature_importances_` after every fit, averages across all trained tickers (same spirit
  as the pooled model's single ranked list, but built from N per-ticker models instead of
  one pooled one — this script deliberately doesn't pool, see its own module docstring),
  and writes `research/triple_barrier_feature_importances.csv` (`feature`,
  `mean_importance`, `n_tickers`).

Both wired into `ml_confidence_backtest.yml` — the pooled-model step returns (previously
removed once pooling-for-prediction was ruled out; it's back now for this diagnostic
purpose only, not as a reconsideration of pooling as the primary approach) and the
existing triple-barrier step now also produces its importances file as a side effect of
the run it already does.

Tested against synthetic data first: both scripts' importance-aggregation logic (groupby-
mean over per-ticker/per-model rows, correct sort/rank) verified end-to-end, and confirmed
`analyst_revision_net_90d` actually appears in the output when synthetic grade events are
present. **Not yet run against real data** — next step is triggering the workflow and
checking where `analyst_revision_net_90d` (and the older `insider_net_buy_pct_90d`/
`rating_score_change_20d`) actually rank, which will tell us whether the null result above
is "no signal" or "signal present but diluted among ~30 other features."

## Update 2026-07-12 (later): feature-importance results — analyst revision ranks last

Ran the instrumentation above against real data (60 tickers, 760-day lookback).

**`research/pooled_feature_importances.csv` (single pooled RF, 31 features, n=318 test
rows): not usable for `analyst_revision_net_90d` or rating attribution.** Only
`insider_net_buy_pct_90d` appears (rank 28/31); neither `analyst_revision_net_90d` nor
`rating_score`/`rating_score_change_20d` made it into this file at all. Root cause:
`build_pooled_dataset()`'s canonical-feature-matching logic sets the "expected" feature
set from the *first* ticker processed and skips every later ticker whose columns don't
match exactly — the first ticker in this run apparently had insider data but not enough
grades/rating coverage, so every ticker that *did* have those columns got silently
excluded from the pool rather than included. A real limitation of this script surfaced by
running it for real, not something synthetic data would have caught (the synthetic test
gave every ticker the same feature set). Not fixing this now — the triple-barrier file
below doesn't share the limitation and already answers the question.

**`research/triple_barrier_feature_importances.csv` (53 per-ticker LGBM models, 32
features): answers the question directly.** `mean_importance` is each feature's
LGBM split-usage share (normalized to sum to 1 per ticker, per the #21 fix — see that
PR), averaged across every ticker whose model actually included it; `n_tickers` is how
many of the 53 trained models had that column at all.

- **`analyst_revision_net_90d` ranks dead last — 32nd of 32 features**
  (mean_importance 0.0093, present in 29/53 tickers' feature sets). This is the strongest
  version of "no signal" this session has produced for this specific feature: not just a
  null AUC/rank-IC at the ensemble level, but the trees themselves consistently found it
  the least useful thing to split on, out of everything available.
- **`rating_score`/`rating_score_change_20d` never appear at all** — 0 of 53 tickers had
  enough historical-ratings coverage to clear `build_feature_table()`'s 50%-coverage gate
  (that function drops `rating_score`/`vix` entirely for a ticker unless real data covers
  at least half its rows — see `build_feature_table()`'s own comments in
  `core/ml_forecast.py` — rather than injecting a mostly-empty column).
  This means the "rating data is a wash" conclusion from earlier in this doc was likely
  never actually tested against real rating data in the first place — worth flagging as a
  distinct, previously-unnoticed gap, separate from today's analyst-revision result.
- **`insider_net_buy_pct_90d` ranks surprisingly high — 2nd of 32**
  (mean_importance 0.0637, present in 27/53 tickers), well above most technical features.
  **This does not contradict the earlier "insider trading has no edge" conclusion** (this
  doc's own regression test, plus the independent public analysis found during repo
  research) — feature importance measures how often a trained model *used* a feature to
  split during training, not whether doing so improved out-of-sample accuracy. A model can
  lean heavily on a feature that's mostly fitting training-set noise. Worth a dedicated,
  narrower follow-up (does the *held-out* AUC/rank-IC change if `insider_net_buy_pct_90d`
  is the only non-technical feature included, isolated from the other ~30) before reading
  anything more into this than "the model finds it interesting," which is a much weaker
  claim than "it works."

**Conclusion.** The attribution question this instrumentation was built to answer now has
a real answer for `analyst_revision_net_90d`: it's not a diluted signal, it's genuinely
the least-used feature in the entire set. Recommend closing out the analyst-revision-momentum
line of work as tested and null, not just under-explored. The insider-trading
result is more ambiguous and would need the narrower isolation test above before drawing
any conclusion from it either way. Per the plan agreed earlier, moving to FinBERT
sentiment next rather than continuing to iterate on this feature set.

**TL;DR for this update:**
- **Closed, null:** `analyst_revision_net_90d` — ranks last of 32 features, not diluted.
- **Reopened as an unknown, not settled:** `rating_score`/`rating_score_change_20d` — never
  actually had enough coverage to be tested; the earlier "wash" verdict doesn't stand on
  real evidence.
- **Open question, not a finding:** `insider_net_buy_pct_90d` — high training-time usage
  despite a null held-out result; needs the isolation test above, not a conclusion either way.
- **Next step:** FinBERT sentiment (a genuinely untested data category), not another pass
  over this feature set.

## Update 2026-07-12 (later still): FinBERT news sentiment built, not yet validated

Built the FinBERT sentiment feature per the plan above — a genuinely different data
category (text/news-derived) from every feature tested so far (all price/volume/filing-
derived).

**Data source**: `agents/market_data_agent.py::fetch_news()` — Alpaca's News API
(Benzinga-sourced, free, no separate credential beyond `ALPACA_API_KEY` already
required), verified live in this session (`NewsClient`/`NewsRequest`/`News` model fields
inspected directly against the installed `alpaca-py` package) before writing any parsing
code, per this session's established discipline. Headline+summary only
(`include_content=False`), never full article bodies.

**Model**: `ProsusAI/finbert` (`core/sentiment.py`) via HuggingFace `transformers` — the
standard, most widely-used open-source FinBERT checkpoint. `transformers`/`torch` are new
dependencies (`requirements.txt`), lazy-imported inside `core.sentiment._get_pipeline()`
so importing the module doesn't force the load for callers that never score anything.

**Feature**: `news_sentiment_net_30d` in `core/ml_forecast.py::build_feature_table()` —
rolling 30-day mean of each article's FinBERT `positive - negative` score. Shorter window
than `insider_net_buy_pct_90d`/`analyst_revision_net_90d`'s 90 days (news sentiment decays
faster than an ownership/rating change) and a rolling *mean* over unfilled (NaN, not
zero-filled) days rather than those two features' rolling *sum* over a zero-filled series
— a newsless day isn't a meaningful "neutral" observation to average in, unlike a
no-trades day being a meaningful "$0" to sum.

Threaded `sentiment_df` through `build_feature_table`/`prepare_features`/both
forecasters/`ensemble_ml_forecast`, `research/walk_forward_backtest.py`, `research/
triple_barrier_walk_forward.py`, and `pipeline.py` — same optional-parameter pattern as
`grades_df` throughout. Scoring happens once per ticker before any walk-forward looping
(FinBERT inference is comparatively expensive; every step for one ticker shares the same
underlying news history), mirroring how insider/rating/grades are fetched once up front
rather than re-fetched per step.

**Tested — with one real limitation.** The feature-engineering and aggregation logic
(rolling-mean-over-NaN-gaps, correct reindexing, `build_ticker_dataset`/`backtest_ticker`
integration) is verified end-to-end with mocked/synthetic sentiment scores, and
`fetch_news()`'s DataFrame-shaping is verified against a mocked Alpaca response matching
the real `News`/`NewsSet` model's actual field names. **The FinBERT model itself was
never actually run in this development session** — this sandbox's network policy blocks
`huggingface.co` (confirmed via the proxy's own status endpoint, same class of restriction
that also blocked `download.pytorch.org` for a CPU-only torch wheel), so model download
and real inference are both untested outside of GitHub Actions. This is a materially
bigger unknown than the "not yet run against real data" caveat on every other feature this
session — those all reused already-proven scoring code (RF/GBM/LGBM) against a new
column; this is new scoring code (FinBERT inference) that has literally never executed.
**Next step**: trigger `ml_confidence_backtest.yml` and confirm the model downloads and
produces sane-looking sentiment scores before reading anything into rank-IC/AUC — a
pipeline bug here (e.g. a silently-empty or constant sentiment feature) could look
identical to a genuine null result without careful inspection of the actual scored values,
not just the downstream backtest metrics.

## Update 2026-07-12 (later still): FinBERT ran for real — engineering validated, edge still null

Ran `ml_confidence_backtest.yml` against real data for the first time with FinBERT live
(60 tickers, 760-day lookback). **The model itself worked**: torch/transformers installed
cleanly (CPU-only wheel, ~1 minute), the model downloaded, and CPU inference completed
across all tickers with no failures — the biggest unknown flagged in the prior update is
resolved. Total runtime rose from ~35 to ~55 minutes (both the regression and
triple-barrier steps independently fetch and score news, same redundant-per-process
pattern already accepted for insider/rating/grades).

**Sanity check first, before reading anything into the metrics** (per the caution in the
prior update): `news_sentiment_net_30d` ranks **6th of 33 features** in
`research/triple_barrier_feature_importances.csv` (mean_importance 0.0436, present in
53/53 trained tickers — full coverage, unlike insider/grades' partial coverage). A
tree-based model gives a constant or silently-broken column ~zero importance, since
there's no split to make on a single value — ranking 6th with a non-trivial value this
close to several genuine technical features is itself strong evidence that the FinBERT scoring
pipeline is producing real, varied sentiment values, not a placeholder. This was
specifically checked because a broken pipeline could otherwise masquerade as a null
result.

**But the held-out predictive metrics are still null, same as everything else this
session:**
- Regression ensemble: rank-IC -0.0024, p=0.91 (unchanged in character from every prior
  run).
- Triple-barrier classifier: AUC 0.4897 (still ~coin flip), point-biserial r=0.0221,
  p=0.36 (not significant), `p_target` bucket report still flat/non-monotonic (7.3%-12.9%
  win rate across quintiles, no relationship to predicted probability).

**This is the same pattern already flagged for `insider_net_buy_pct_90d`**: high
training-time feature usage that does not translate into genuine out-of-sample signal.
Feature importance measures how much the trees *used* a column, not whether doing so
improved held-out accuracy — a model can lean on a feature that's mostly fitting
training-set noise. Sentiment joins insider trading in the "ambiguous, ranks high but
unproven" bucket, not the "closed, tested null" bucket `analyst_revision_net_90d` and
`rating_score` are in.

**Where this leaves the project.** Four data categories tested this session
(price/volume-derived technicals, insider trading, analyst revision momentum, news
sentiment) — none has produced a statistically significant held-out edge. Two of the four
(insider, sentiment) show unusually high training-time feature importance without a
matching accuracy improvement, which is a real, specific, and now twice-observed pattern
worth its own investigation rather than two isolated coincidences — both would benefit
from the same narrower single-feature-isolation test proposed earlier
(`research/pooled_model_experiment.py`-style single-shot training with just one
non-technical feature added, to see if held-out AUC/rank-IC moves at all when the
distracting ~30-feature technical noise is removed). Recommend that isolation test as the
next concrete step if this line of work continues, rather than adding a fifth data
category on top of an already-large, still-unexplained importance/accuracy mismatch.

## Update 2026-07-13: does the base rules-based system even have an edge? Built, not yet run

New direction, agreed after stepping back to research how other quant/algo traders
actually validate systems (see conversation — daily-bar technical-indicator ML prediction
is the single most competed-for slice of the market, and academic literature backs up
that it routinely performs "no better than random"; position sizing/risk management
matters far more than entry-signal quality per repeatedly-cited practitioner research,
e.g. Van Tharp's finding that position sizing explained ~90% of outcome variance across
otherwise-identical systems). Given that, and five null ML results in a row, decided to
downgrade ML's role and ask a question this entire session never actually asked:
**does the base rules-based system — `core.smartscore.compute_smartscore` +
`core.trade_plan.compute_trade_plan`, no ML anywhere — have a real edge on its own?**
Every walk-forward test so far tested whether an ML layer improves on this base system;
none tested the base system itself.

**Built**: `research/rules_based_walk_forward.py` — same walk-forward-through-history
shape as `research/triple_barrier_walk_forward.py` (reuses `resolve_trade_plan_outcome()`,
same `MAX_HOLD_DAYS`=30 vertical barrier, same historical-truncation causal-correctness
discipline), but with no ML model at all: at each historical as-of date, calls
`compute_smartscore()` and `compute_trade_plan()` on data truncated to that date, records
the resulting SmartScore/entry/stop/target/rr_ratio, and resolves the real outcome against
the actual bars that followed. Skips dates where `compute_smartscore` finds no setup or
near-miss, matching the live pipeline's own behavior (`MarketDataAgent.scan_universe` only
computes a trade plan after a ticker already has a non-None SmartScore). No FMP or Alpaca
News dependency at all — this system is 100% price/volume-derived.

**`research/analyze_rules_based.py`** asks two new questions that no prior analysis script
this session asked:
1. **Expectancy** — is the realized average R-multiple per trade (+`rr_ratio` on a win,
   -1 on a loss, exact by construction since `resolve_trade_plan_outcome` resolves to the
   literal stop/target price, never a partial fill) significantly greater than zero via a
   one-sample t-test? This is a direct expected-value check on the entry/stop/target logic
   itself, independent of any ranking.
2. **Calibration** — does SmartScore's own ranking separate good setups from bad ones?
   Reuses `research/analyze_confidence.py::confidence_bucket_report()` completely
   unmodified (`column="smartscore"`) — `rules_based_walk_forward.py`'s output CSV names
   its win/loss and return columns `direction_correct`/`actual_return_pct` specifically so
   that function's existing contract works without any changes.

Also breaks results down by setup type (Breakout / Pullback / near-miss-only) as a third,
simpler cut.

**Scope note** (same simplification already accepted for the ML walk-forward scripts):
tests `compute_smartscore`/`compute_trade_plan` per ticker in isolation, not the full live
pipeline's cross-sectional sector-cap/deep-discount adjustments, which need the whole
universe scanned simultaneously on each historical date.

Wired into `ml_confidence_backtest.yml` as two new steps (no new secrets needed — bars
only). Tested end-to-end against synthetic random-walk price data first, per this
session's established discipline: confirmed real Breakout/Pullback/near-miss setups get
classified, trade plans get computed, outcomes resolve correctly, and both new analysis
functions (`compute_expectancy`'s t-test, the reused `confidence_bucket_report`) run
without error. Synthetic-data expectancy was negative, as expected for pure random-walk
noise with no real pattern to exploit — not a finding, just confirmation the pipeline
doesn't produce nonsense before spending real API credits on it.

**Not yet run against real data.** Next step: trigger `ml_confidence_backtest.yml` and
read `research/rules_based_summary.txt` for the actual expectancy t-test and SmartScore
bucket report. This is arguably the most consequential result this whole research effort
could produce — if the base system itself has no real edge, that's a far more important
finding than anything about ML feature engineering; if it does, that's the foundation
worth building on rather than the ML layer sitting on top of it.

## Update 2026-07-12 (real-data result): the base system loses money, but the mechanism points at target-setting, not entry-timing

Ran `ml_confidence_backtest.yml` against real data (60 tickers, 760-day lookback,
`step_days=3`). 429 scored trade plans resolved.

**This is not a null result — it's a statistically significant *negative* one.**
- Win rate: **5.4%** (target hit before stop, out of 429 trade plans).
- Mean R-multiple: **-0.68** (t=-9.53, p≈0.0 — nowhere close to noise; the average trade
  loses roughly two-thirds of a risk-unit).
- Mean R:R ratio: **10.55** — the average trade's target is set over 10x further away
  than its stop, in risk-distance terms.

**The mechanism matters more than the headline number here.** A win rate this low
combined with an R:R ratio this high is the textbook signature of *targets set too far
away for the available time window*, not necessarily "the entries are bad." `core.
trade_plan.compute_trade_plan()`'s target comes from a Fibonacci 1.618 extension of the
most recent 20-bar swing (floored at `settings.min_risk_reward`=3.0 only when the raw
extension falls short) — a 1.618x extension of a real swing can be a very large price
move, and `MAX_HOLD_DAYS`=30 may simply not be enough time for price to travel that far
in most cases, independent of whether the entry timing/direction call itself was any
good. The math: at true 33% (1-in-3) breakeven odds implied by a 3:1 floor, a 10.55:1
average R:R implies a breakeven win rate of only ~8.7% (`1/(1+10.55)`) — the realized 5.4%
still falls short of even that generous bar, but the gap between "needs 8.7%" and "needs
enough time to travel a 16x-swing-range extension" is a target-distance problem, not
necessarily an entry-quality problem.

**SmartScore's own ranking shows no differentiation** — win rate is flat across
quintiles (5.4%, 5.5%, 6.1%, 5.1%, no monotonic trend at all). But this is confounded by
the near-uniform failure rate: when essentially every bucket loses money for the same
target-distance reason, there's little room for a ranking signal to show through even if
SmartScore's entry-timing logic has real merit on its own.

**By setup type**: Breakout 3.7% win rate, Pullback 6.6%, near-miss-only 7.8% — all low,
though Pullback and near-miss modestly outperform Breakout. Not enough to change the
overall conclusion, but a thread worth pulling if this gets revisited.

**What this actually tells us.** The system as currently configured — SmartScore's setup
detection paired with the Fibonacci-extension target and a 30-day hold window — has a
real, statistically robust negative expectancy. That's a genuinely important finding,
independent of every ML result in this document: **before any ML overlay is worth
reconsidering, the base trade-plan's target distance vs. hold-window mismatch needs
addressing.** The concrete next step, if this continues: rerun this exact same backtest
with a more conservative target (e.g. a flat R:R floor without the Fibonacci extension,
or a target distance scaled to the ticker's own realized volatility over 30 days) to
isolate whether SmartScore's entry-timing/ranking has real value once the target isn't
set unrealistically far for the hold window it's paired with. That test would cleanly
separate "the setup detection is bad" from "the target is bad," which this run cannot
distinguish on its own.

## Update 2026-07-12 (later): flat-target isolation test built, not yet run

Built the follow-up test proposed above, to separate "the target is set too far" from
"the entries/SmartScore are bad."

**`research/rules_based_walk_forward.py --target-mode flat`**: reuses `compute_trade_plan()`'s
entry/stop exactly as-is (its swing-low/EMA-anchored stop logic isn't implicated in the
negative-expectancy finding) but overrides the target to a flat `settings.min_risk_reward`
multiple of that same risk distance — no Fibonacci extension, no resistance-based
refinement. `apply_flat_target()` is a small experimental override living only in the
research script; `core.trade_plan.compute_trade_plan()` itself is untouched, so live
pipeline behavior is unaffected. Wired into `ml_confidence_backtest.yml` as two more
steps (`research/rules_based_results_flat_target.csv` /
`research/rules_based_summary_flat_target.txt`), reusing `research/analyze_rules_based.py`
unmodified via its existing `--input` flag.

**Tested against synthetic data first**: on the same synthetic random-walk price series,
`fibonacci` mode gave mean R:R 9.1 / win rate 9.8%, while `flat` mode gave mean R:R 3.0
(exactly, by construction) / win rate 25.8% — a large, qualitatively expected shift in the
direction the target-distance hypothesis predicts, confirming the override logic works
correctly before spending real API credits on it. (Not a finding — synthetic random-walk
data has no real edge to detect either way; this only confirms the mechanism behaves as
designed.)

**Not yet run against real data.** Next step: trigger `ml_confidence_backtest.yml` and
compare `research/rules_based_summary_flat_target.txt` against the Fibonacci-target
result documented above. If the flat-target expectancy is close to breakeven or positive
and/or SmartScore's bucket ranking shows real separation once win rates aren't uniformly
crushed by an unreachable target, that isolates the Fibonacci extension as the specific
fixable problem. If flat-target expectancy is still significantly negative, that would
point back at the entry/setup-detection logic itself, not just target distance.

## Update 2026-07-12 (real-data result): flat target helps a lot, but doesn't fix it — both mechanisms are real

Ran the isolation test for real (GH Actions run 29199835039, same `n_tickers=60,
lookback_days=760, step_days=10` sampling as every prior run this session; results
committed to `main` at `277fcfb` — `research/rules_based_results.csv` /
`rules_based_results_flat_target.csv` and their `_summary` files at that commit are the
exact source of every number below). Result, compared directly against the
Fibonacci-target run documented above:

| | Fibonacci target | Flat (3:1) target |
|---|---|---|
| n | 429 | 462 |
| win rate | 5.4% | **16.2%** |
| mean R:R | 10.55 | 3.0 |
| breakeven win rate (`1/(1+RR)`) | 8.7% | 25.0% |
| mean R-multiple | -0.6757 | **-0.3506** |
| t-stat / p-value | -9.525 / p < 0.00005 | -5.104 / p < 0.00005 |

(`analyze_rules_based.py` rounds `p_value` to 4 decimals, so both print as `0.0`; scipy's
underlying value is smaller than that rounding can distinguish from zero, not literally
zero — `p < 0.00005` is the precise bound the printed `0.0` implies.)

The flat target roughly **tripled the win rate** (5.4%→16.2%) and cut the loss in half
(mean R -0.68→-0.35), exactly the direction the target-distance hypothesis predicted —
confirming the Fibonacci extension was a real, substantial part of the problem, not a red
herring. But it's **still significantly negative** (p < 0.00005): a 16.2% win rate
doesn't clear the 25% breakeven bar a 3:1 target requires. So this isolation test lands
on neither of the two clean answers it was designed to distinguish between — it's both.
Fixing the target alone would not make this system profitable.

**SmartScore quintile win rates (flat target): 15.0%, 13.1%, 14.3%, 20.1%** (lowest to
highest bucket) — the top quintile does best, but the ordering is not monotonic (bucket 2
underperforms bucket 1) and even the best bucket is well below the 25% breakeven bar.
Same weak, non-monotonic pattern as the Fibonacci run's quintiles (5.4%, 5.5%, 6.1%,
5.1%) — SmartScore is not cleanly separating good setups from bad ones under either
target scheme.

**By setup type (flat target): Breakout 12.2%, Pullback 18.9%, near_miss_only 21.8%.**
Breakout — the setup SmartScore scores most aggressively (setup_strength + trend_bonus +
base_tightness bonuses all stack for Breakout, see `core/smartscore.py`) — has the
*worst* win rate of the three, and near-misses (setups that didn't even clear the
classification bar) do best. Same inversion as the Fibonacci run. This is the more
concerning half of the picture: even holding target distance constant at a realistic 3:1,
the setups SmartScore is most confident about are not the ones that actually work.

**Bottom line:** this closes the isolation test cleanly, just not with a single-cause
answer. Both hypothesized mechanisms are real:
1. The Fibonacci-extension target was set unrealistically far for a 30-day hold window —
   confirmed, and worth fixing on its own (it more than halved the loss).
2. Independent of target distance, SmartScore's entry/setup classification is not
   correctly ranking quality — Breakout setups underperform near-misses under both target
   schemes, and quintile separation is weak and non-monotonic under both.

Combined with the earlier practitioner research (this doc's prior updates on position
sizing and signal diversification), the actionable path forward is not "revert the target
formula and ship it" — it needs both a more realistic target-distance model *and* a hard
look at whether `classify_setup()`'s Breakout/Pullback thresholds and SmartScore's bonus
weights (`SETUP_THRESHOLDS`, the setup_strength/trend/volume/base/level/fib bonuses in
`core/smartscore.py::compute_smartscore`) are actually correlated with forward returns,
which no test this session has directly checked before now.

## Update 2026-07-12 (target-distance rework, round 2): volatility-scaled target built, not yet run

Picked up mechanism #1 above first (target-distance model). Both the Fibonacci extension
and the flat 3:1 floor share a blind spot: neither target is set relative to how far this
specific ticker actually tends to move in 30 days — Fibonacci extrapolates a recent swing
by a fixed multiple, flat just multiplies whatever the stop distance happens to be.
Neither reflects the ticker's own realized volatility over the actual hold window.

**`research/rules_based_walk_forward.py --target-mode volatility`** (`apply_volatility_target()`):
same entry/stop as `compute_trade_plan()` — only the target changes. Computes this
ticker's trailing daily-return standard deviation over `VOL_LOOKBACK_DAYS=60` bars,
projects it to the actual hold window via random-walk `sqrt(MAX_HOLD_DAYS)` scaling
(expected dispersion grows with the square root of time under a driftless random walk),
and sets the target at `VOL_TARGET_K=1.0` standard deviations of that projected move.
`rr_ratio` is intentionally left unfloored (unlike `compute_trade_plan` and
`apply_flat_target`) so it reports whatever R:R a volatility-scaled target naturally
implies, rather than engineering it back to a fixed number.

**Why k=1.0 specifically:** a pure random walk with zero directional edge would be
expected to close above a 1-standard-deviation target roughly 16% of the time (one-tailed
`P(Z>1)` on a symmetric distribution ≈ 15.9%). That gives this isolation test a cleaner
edge check than an R:R-implied breakeven bar: if SmartScore's entries have real
directional value, realized win rate should clear that ~16% random-walk baseline;
if SmartScore is picking setups no better than chance, win rate should land near it.

**Tested against synthetic data first** (20 seeds of synthetic daily-return-normal random
walks, 900 bars each, drift +0.03%/day / volatility 2%/day, `step_days=3`,
`max_hold_days=30` — same sanity-check pattern used for the flat-target build):
`fibonacci` gave win rate 6.3% (n=847, mean R:R 9.50), `flat` gave win rate 24.2% (n=980,
mean R:R 3.00, exactly by construction), `volatility` gave win rate **25.7%** (n=1051,
mean R:R 4.48). The volatility mode's win rate landing meaningfully above the ~16%
driftless-random-walk baseline is exactly what a small positive drift (0.03%/day × 30 ≈
0.9%, small but nonzero next to ~11% of 30-day volatility) should produce — confirms the
mechanism behaves as designed before spending real API credits on it. (Not a finding —
synthetic data's "edge" here is a manufactured drift, not anything about SmartScore.)
`research/analyze_rules_based.py` needed zero changes to work against the new mode's
output — same reuse-via-matching-column-names design as the flat-target test.

Wired into `ml_confidence_backtest.yml` as two more steps
(`research/rules_based_results_volatility_target.csv` /
`research/rules_based_summary_volatility_target.txt`), same `if [ -s ... ]` guard pattern
as every other optional step in that workflow.

**Not yet run against real data.** Next step: trigger the workflow and compare this
mode's real win rate against the ~16% random-walk baseline (not the R:R-implied breakeven
bar used for the other two modes, since this target's R:R varies per-trade rather than
being fixed). If real win rate clears ~16% with a positive, significant mean R-multiple,
that's the first real edge finding in this entire research effort. If it lands at or
below ~16%, that's further evidence pointing at `core/smartscore.py`'s setup
classification itself, which is the next thing queued up regardless of this result.
