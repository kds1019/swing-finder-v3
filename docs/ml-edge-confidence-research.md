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
