"""
EMA200 pullback/reversal screener — FRESH DESIGN, replaces core.smartscore's
classify_setup()-driven gating in the live scan path. Not a port of anything in
swing-finder-v2.

Built and calibrated directly against a real trade (EMBJ, bought 2026-06-12 at
$57.60, sold 2026-07-01 at $64.00, +11.3%), pulled from the user's actual Webull
order history and price bars rather than worked from memory of a description:
  - EMA200 rose from $42.74 (1yr before) -> $54.34 (6mo before) -> $61.36 (3mo
    before) -> $60.76 (at entry): a strong, sustained long-term uptrend that had
    only just flattened in the most recent 3 months — exactly what a real pullback
    inside an uptrend looks like, since a 200-day average reacts slowly and will
    read as "flat" during one even when the longer trend is genuinely up. This is
    why the uptrend check below looks 6 months back, not just the last few weeks.
  - Price at entry was $57.60 vs EMA200 $60.76: -5.2%, i.e. "slightly under."
  - The 15 trading days into the entry ranged roughly $53-59 (a ~10% band) with a
    local low around $53.63 two days before entry, then a two-day bounce into the
    $57.60 buy — "stabilized, then started a slight reversal up."
  - Volume profile (60-day window) at entry: POC $56.53, value area $54.28-$63.30.
    Price ($57.60) was actually just *above* the POC (+1.89%), not at/below it —
    an earlier draft of this screener required price <= POC and would have wrongly
    excluded the real reference trade. It was, however, comfortably inside the
    value area, well below value_area_high — that's the condition used below.

Every threshold here is a **percentage relative to that ticker's own price/EMA200/
volume profile**, not an absolute dollar level — this scans for the shape of the
setup (pullback into a rising 200-day average, stabilize, not stretched into thin
volume above) at any price, not for stocks that resemble EMBJ's price level
specifically.

CALIBRATION (2026-08-31): the thresholds below are no longer EMBJ's numbers plus a
margin. They were re-derived from research/build_calibration_dataset.py — ~147k
labelled instances across 461 tickers, 2021-2026 — by binning each measurement
against the realised R-multiple (research/analyze_calibration.py,
research/calibration_findings.md) and putting each cutoff where an edge actually
appears. Headline: the EMBJ-fit gates barely beat "rising 200-EMA + roughly near
it" (profit factor 1.19 vs 1.18); the re-derived gates reach ~1.27 (train 1.24 /
test 1.33 on a 2021-2024 vs 2025-2026 split). What changed and why:
  - price-vs-EMA200 band -12%/+8%  ->  -20%/+3%. The edge is monotonic in pullback
    DEPTH (the -25%..-12% bins were the best); the +4%..+15% region above EMA200
    was dead weight (PF ~1.06). This is the deep-pullback trade, not a shallow dip.
  - "not above value_area_high" (price_vs_vah <= 0)  ->  price_vs_vah <= -4%. The
    edge was already gone by -2%; a real margin below the value-area high is what
    matters, not merely "not above it."
  - consolidation range <= 15%  ->  <= 20%. Wider recent ranges did slightly
    BETTER, not worse — the tight-range requirement wasn't helping.
  - min bounce off the 15-day low 3%  ->  0% (gate effectively removed). The best
    outcomes were with price STILL AT the low (0-1% off); requiring a 3%+ bounce
    just bought a worse entry after the move had already started.
Things the calibration said NOT to add, against prior expectation: a
relative-strength floor (RS laggards did mildly better for this setup — a deep
pullback IS relative weakness), a 50-EMA > 200-EMA gate (50 < 200 bins were
better — that cross is the pullback's signature), a near-52-week-high filter
(within 4% of the high LOSES money here), and a SPY-trend regime gate (no help;
2022's damage is already covered by the VIX <= 20 gate in pipeline.py).

Caveats: ~one market cycle of IEX history; survivorship-biased to today's
universe; weak-RR trade plans excluded from the calibration numbers. Treat the
output as a modest-edge candidate filter, re-run the calibration as more history
and resolved live picks (pick_outcomes.csv) accumulate. See docs/strategy.md.
"""

from __future__ import annotations

import pandas as pd

from core.volume_profile import compute_volume_profile

# How far back to check whether EMA200 itself is trending up — long enough that a
# genuine multi-week pullback (which flattens the recent EMA200 slope even inside a
# real uptrend) doesn't get misread as "no uptrend."
EMA200_TREND_LOOKBACK_DAYS = 126  # ~6 months of trading days
# Kept at 5.0: the calibration showed only a mild, roughly monotonic benefit to a
# steeper long-term trend, and 5% vs 3% made little difference — not worth tightening.
EMA200_MIN_UPTREND_PCT = 5.0

# Price must sit within this band of EMA200. Calibrated: the realised edge is
# monotonic in pullback DEPTH (deeper is better, all the way down to ~-25%), and
# fades to nothing above ~+3%. This is deliberately a deep-pullback filter.
PRICE_VS_EMA200_MIN_PCT = -20.0
PRICE_VS_EMA200_MAX_PCT = 3.0

# Consolidation window. Calibrated: a wider recent range was, if anything, slightly
# better — so this is a loose sanity bound, not a "must be quiet" gate.
CONSOLIDATION_LOOKBACK_DAYS = 15
CONSOLIDATION_MAX_RANGE_PCT = 20.0
# Calibrated to 0.0 (gate effectively off): the best outcomes were with price still
# at the 15-day low; requiring an early bounce bought a worse entry. Kept as a
# constant (not deleted) so a future calibration can re-enable it cleanly.
MIN_BOUNCE_OFF_LOW_PCT = 0.0

# Volume profile window.
VOLUME_PROFILE_WINDOW_DAYS = 60
# "Not extended" gate: price must sit at least this far BELOW the value-area high
# (where 70% of recent volume traded), not merely "not above it". Calibrated: the
# edge was already gone by -2%; -4% is where the good bins start.
MAX_PRICE_VS_VALUE_AREA_HIGH_PCT = -4.0

# Minimum bars this screener needs to run at all — defined here (not duplicated as a
# magic number by callers) so a future change to either lookback constant can't
# silently desynchronize from whatever pre-filter a caller applies before invoking
# this function. agents/market_data_agent.py imports this directly for that reason —
# a hardcoded `len(df) < 60` here previously (a leftover from core.smartscore's own
# minimum) silently rejected almost every ticker as "insufficient_data" once this
# screener's real requirement grew past 60.
MIN_BARS_FOR_SCREENER = max(EMA200_TREND_LOOKBACK_DAYS, CONSOLIDATION_LOOKBACK_DAYS) + 1


def measure_pullback_reversal(df: pd.DataFrame) -> dict | None:
    """Every raw measurement the screener's gates are applied to, for the most
    recent bar of `df` (must already have compute_indicators() applied — needs
    EMA200). NO thresholds applied: returns the continuous values so that
    research/ calibration can bin each one against realised outcomes, while
    detect_pullback_reversal() layers the actual gates on top of this. Returns
    None only when there genuinely isn't enough history (or a degenerate EMA200)
    to compute the measurements at all. See docs/strategy.md."""
    if df is None or len(df) < MIN_BARS_FOR_SCREENER:
        return None

    close = df["Close"]
    ema200 = df["EMA200"]
    current_close = float(close.iloc[-1])
    current_ema200 = float(ema200.iloc[-1])
    if pd.isna(current_ema200) or current_ema200 <= 0:
        return None

    ema200_then = float(ema200.iloc[-1 - EMA200_TREND_LOOKBACK_DAYS])
    if pd.isna(ema200_then) or ema200_then <= 0:
        return None

    ema200_uptrend_pct = round((current_ema200 - ema200_then) / ema200_then * 100, 2)
    price_vs_ema200_pct = round((current_close - current_ema200) / current_ema200 * 100, 2)

    window = close.tail(CONSOLIDATION_LOOKBACK_DAYS)
    window_low = float(window.min())
    window_high = float(window.max())
    consolidation_range_pct = round((window_high - window_low) / current_close * 100, 2)
    bounce_off_low_pct = (
        round((current_close - window_low) / window_low * 100, 2) if window_low > 0 else 0.0
    )

    vp = compute_volume_profile(df, window=VOLUME_PROFILE_WINDOW_DAYS)
    poc = value_area_low = value_area_high = None
    price_vs_poc_pct = price_vs_value_area_high_pct = None
    if vp is not None:
        poc = vp["poc"]
        value_area_low = vp["value_area_low"]
        value_area_high = vp["value_area_high"]
        if poc:
            price_vs_poc_pct = round((current_close - poc) / poc * 100, 2)
        if value_area_high:
            price_vs_value_area_high_pct = round(
                (current_close - value_area_high) / value_area_high * 100, 2
            )

    return {
        "close": current_close,
        "ema200_uptrend_pct": ema200_uptrend_pct,
        "price_vs_ema200_pct": price_vs_ema200_pct,
        "consolidation_range_pct": consolidation_range_pct,
        "bounce_off_low_pct": bounce_off_low_pct,
        "poc": poc,
        "price_vs_poc_pct": price_vs_poc_pct,
        "value_area_low": value_area_low,
        "value_area_high": value_area_high,
        "price_vs_value_area_high_pct": price_vs_value_area_high_pct,
        "volume_profile_available": vp is not None,
    }


# How far back to look for the low of the current pullback.
STABILIZATION_LOOKBACK_DAYS = 20


def measure_stabilization(df: pd.DataFrame) -> dict:
    """Recent price-action read for the most recent bar of `df` (needs
    compute_indicators() applied). This is NOT a screener gate — the calibration
    found a hard "must have bounced" requirement HURT expectancy, because the
    trailing stop already caps a failed entry at -1R. It is structured context for
    the Decision Agent's judgment call: has this pullback stabilized and found
    support, or is the stock still in an active decline (a falling knife)? See
    docs/strategy.md.

    Returns {} if there isn't enough history. Fields:
      last_10d_return_pct / last_20d_return_pct
          recent trend. Both sharply negative + days_since_pullback_low == 0 => still falling.
      days_since_pullback_low
          bars since the lowest low of the last STABILIZATION_LOOKBACK_DAYS. Higher = a
          base is forming.
      higher_low_pct
          the last-3-day low vs that pullback low, as %. > 0 => a higher low is in
          (an actual reversal signal), <= 0 => still probing / making new lows.
      range_contraction_ratio
          mean daily range of the last 5 bars / that of the 15 before. < 1 => settling
          down, > 1 => still volatile / accelerating.
      down_up_volume_ratio
          avg volume on down-close days / up-close days over the last 12 bars.
          < 1 => selling pressure is fading relative to buying.
    """
    if df is None or len(df) < STABILIZATION_LOOKBACK_DAYS + 20:
        return {}

    close, low, high, vol = df["Close"], df["Low"], df["High"], df["Volume"]
    px = float(close.iloc[-1])

    last_10d_return_pct = round((px / float(close.iloc[-11]) - 1) * 100, 2)
    last_20d_return_pct = round((px / float(close.iloc[-21]) - 1) * 100, 2)

    win_low = low.tail(STABILIZATION_LOOKBACK_DAYS)
    window_low = float(win_low.min())
    days_since_pullback_low = int(len(win_low) - 1 - win_low.to_numpy().argmin())

    recent_low_3d = float(low.tail(3).min())
    higher_low_pct = round((recent_low_3d - window_low) / window_low * 100, 2) if window_low > 0 else 0.0

    rng = (high - low)
    recent_rng = float(rng.tail(5).mean())
    prior_rng = float(rng.iloc[-20:-5].mean())
    range_contraction_ratio = round(recent_rng / prior_rng, 2) if prior_rng > 0 else None

    r12 = df.tail(12)
    up_v = r12.loc[r12["Close"] > r12["Close"].shift(1), "Volume"]
    dn_v = r12.loc[r12["Close"] < r12["Close"].shift(1), "Volume"]
    up_mean = float(up_v.mean()) if len(up_v) else 0.0
    dn_mean = float(dn_v.mean()) if len(dn_v) else 0.0
    down_up_volume_ratio = round(dn_mean / up_mean, 2) if up_mean > 0 else None

    return {
        "last_10d_return_pct": last_10d_return_pct,
        "last_20d_return_pct": last_20d_return_pct,
        "days_since_pullback_low": days_since_pullback_low,
        "higher_low_pct": higher_low_pct,
        "range_contraction_ratio": range_contraction_ratio,
        "down_up_volume_ratio": down_up_volume_ratio,
    }


def detect_pullback_reversal(df: pd.DataFrame) -> dict:
    """Detects the EMA200 pullback + stabilization/reversal setup for the most
    recent bar of `df` (must already have compute_indicators() applied — needs
    EMA200). Returns {"detected": False} if there isn't enough history or the
    setup's criteria aren't met; otherwise returns "detected": True plus the raw
    measurements (not a 0-100 score) so callers can rank/filter on whichever
    dimension matters most — this only gates whether the pattern is present.

    Thin threshold layer over measure_pullback_reversal() — the module constants
    are the gates, that function is the measurements. The return shape (detected,
    reason, and the per-gate measurement keys) is unchanged from before the split."""
    m = measure_pullback_reversal(df)
    if m is None:
        return {"detected": False, "reason": "insufficient_data"}

    partial = {
        "ema200_uptrend_pct": m["ema200_uptrend_pct"],
        "price_vs_ema200_pct": m["price_vs_ema200_pct"],
        "consolidation_range_pct": m["consolidation_range_pct"],
        "bounce_off_low_pct": m["bounce_off_low_pct"],
    }

    if m["ema200_uptrend_pct"] < EMA200_MIN_UPTREND_PCT:
        return {"detected": False, "reason": "no_long_term_uptrend",
                "ema200_uptrend_pct": m["ema200_uptrend_pct"]}

    if not (PRICE_VS_EMA200_MIN_PCT <= m["price_vs_ema200_pct"] <= PRICE_VS_EMA200_MAX_PCT):
        return {"detected": False, "reason": "price_too_far_from_ema200",
                "ema200_uptrend_pct": m["ema200_uptrend_pct"],
                "price_vs_ema200_pct": m["price_vs_ema200_pct"]}

    if m["consolidation_range_pct"] > CONSOLIDATION_MAX_RANGE_PCT:
        return {"detected": False, "reason": "not_consolidating",
                "ema200_uptrend_pct": m["ema200_uptrend_pct"],
                "price_vs_ema200_pct": m["price_vs_ema200_pct"],
                "consolidation_range_pct": m["consolidation_range_pct"]}

    if m["bounce_off_low_pct"] < MIN_BOUNCE_OFF_LOW_PCT:
        return {"detected": False, "reason": "no_reversal_yet", **partial}

    if not m["volume_profile_available"]:
        return {"detected": False, "reason": "insufficient_data", **partial}

    vah_pct = m["price_vs_value_area_high_pct"]
    if vah_pct is None or vah_pct > MAX_PRICE_VS_VALUE_AREA_HIGH_PCT:
        return {"detected": False, "reason": "extended_above_value_area",
                **partial, "poc": m["poc"]}

    return {
        "detected": True,
        "ema200_uptrend_pct": m["ema200_uptrend_pct"],
        "price_vs_ema200_pct": m["price_vs_ema200_pct"],
        "consolidation_range_pct": m["consolidation_range_pct"],
        "bounce_off_low_pct": m["bounce_off_low_pct"],
        "poc": m["poc"],
        "price_vs_poc_pct": m["price_vs_poc_pct"],
    }
