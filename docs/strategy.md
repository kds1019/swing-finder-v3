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

1. **Screen** (`core/universe.py`, + hard technical filters) — necessary conditions that
   rarely change: liquidity, price, market cap, volatility band, 200-EMA rising, 50-EMA
   above 200-EMA, long-term relative strength. Calibrated from broad historical data.
2. **Setup** (`core/pullback_reversal.py`) — the timing shape: pullback depth,
   consolidation, early bounce, not-extended. Small-sample risk is concentrated here.
3. **Decision** (`agents/decision_agent.py`) — fundamentals, catalyst, portfolio fit,
   final ranking and selection.

Keeping these separate means the screen layer gets calibrated against thousands of
instances, and only the setup-shape thresholds carry the one-trade calibration risk.

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
  Remaining trade-plan lever: ~32% of signals still get a weak-RR plan (support refinement
  widens the stop below the R:R floor); those are PF ~1.1 vs ~1.3 for the clean set —
  either skip them or stop the refinement from crossing the floor.
