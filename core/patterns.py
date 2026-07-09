"""
Chart pattern detection — ported from swing-finder-v2's utils/indicators.py
(bull flag, bear flag, cup and handle, double top/bottom, ascending/descending
triangle, head and shoulders). Live in that app's scan path but purely
informational there (rendered as a badge, never affecting score). Here it also
feeds a SmartScore bonus/penalty via evaluate_pattern_score() — same post-hoc-
adjustment pattern as core/deep_discount_filter.py and core/volume_profile.py.

Two deliberate fixes vs the reference app, not carried forward as-is:
  1. detect_head_and_shoulders's docstring there claims "every combination of
     3 pivots" but the code only iterates consecutive triples
     (phs[i], phs[i+1], phs[i+2]), silently missing valid non-adjacent
     shoulder/head combinations. This port actually checks every ordered
     triple via itertools.combinations.
  2. The reference app's detect_patterns() silently overrides each detector's
     own default lookback with a shorter one (e.g. bull flag's default 40
     becomes 20), undocumented and inconsistent with the individual
     functions' own design. This port lets each detector use its own
     default — the deep-history bars this runs on (~750 days, see
     pipeline.py's DEEP_HISTORY_LOOKBACK_DAYS) have no shortage of data to
     justify shortening it.

find_pivot_points is not re-implemented here — core/indicators.py already has
an identical port, reused directly.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from core.indicators import find_pivot_points


def detect_bull_flag(df: pd.DataFrame, lookback: int = 40) -> dict:
    """
    Bull Flag: a pivot high (pole top) followed by a tight, slightly-downward
    consolidation channel before price breaks higher.

      1. Find the most recent pivot high within lookback bars.
      2. Verify the move INTO that pivot high was >= 5% (the pole).
      3. After the pivot high, check that price consolidates in a narrow
         range (< 8% of the pivot-high price) with no new pivot high.
      4. Volume should dry up during the flag.
    """
    if len(df) < lookback:
        return {"detected": False}

    recent = df.tail(lookback)
    pivots = find_pivot_points(recent, left_bars=3, right_bars=3)
    phs = pivots["pivot_highs"]

    if not phs:
        return {"detected": False}

    pole_top = phs[-1]
    pole_bar = pole_top["bar"]
    pole_price = pole_top["price"]

    if pole_bar < 3 or (len(recent) - 1 - pole_bar) < 5:
        return {"detected": False}

    pole_start_price = float(recent["Close"].iloc[max(0, pole_bar - 5)])
    initial_gain = (pole_price - pole_start_price) / pole_start_price if pole_start_price > 0 else 0
    strong_pole = initial_gain >= 0.05

    flag_bars = recent.iloc[pole_bar:]
    flag_high = float(flag_bars["High"].max())
    flag_low = float(flag_bars["Low"].min())
    flag_range = (flag_high - flag_low) / pole_price if pole_price > 0 else 1

    tight_flag = flag_range < 0.08
    no_new_high = flag_high <= pole_price * 1.01

    pole_bars_slice = recent.iloc[max(0, pole_bar - 5):pole_bar + 1]
    vol_pole = pole_bars_slice["Volume"].mean() if len(pole_bars_slice) > 0 else 1
    vol_flag = flag_bars["Volume"].mean() if len(flag_bars) > 0 else vol_pole
    volume_decrease = vol_flag < vol_pole * 0.85

    detected = strong_pole and tight_flag and no_new_high
    confidence = 0
    if detected:
        confidence = 65
        if volume_decrease:
            confidence += 15
        if flag_range < 0.05:
            confidence += 10
        if initial_gain >= 0.10:
            confidence += 10

    return {
        "detected": detected,
        "confidence": min(100, confidence),
        "initial_gain": round(initial_gain * 100, 1),
        "consolidation_range": round(flag_range * 100, 1),
    }


def detect_cup_and_handle(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Cup and Handle: a real left-rim high, cup-bottom low, and right-rim
    recovery before a small handle pullback.

      1. Find a significant pivot high (left rim) and the deepest pivot low
         after it (cup bottom).
      2. Verify the right side recovers to >= 92% of the left rim.
      3. Verify cup depth is 12-40%.
      4. The last 20% of bars form a tight handle (< 10% range).
    """
    if len(df) < lookback:
        return {"detected": False}

    recent = df.tail(lookback)
    pivots = find_pivot_points(recent, left_bars=3, right_bars=3)
    phs = pivots["pivot_highs"]
    pls = pivots["pivot_lows"]

    if len(phs) < 1 or len(pls) < 1:
        return {"detected": False}

    left_rim = phs[0]
    left_bar = left_rim["bar"]
    left_price = left_rim["price"]

    lows_after = [p for p in pls if p["bar"] > left_bar]
    if not lows_after:
        return {"detected": False}
    cup_bottom = min(lows_after, key=lambda p: p["price"])
    bottom_bar = cup_bottom["bar"]
    bottom_price = cup_bottom["price"]

    cup_depth = (left_price - bottom_price) / left_price if left_price > 0 else 0
    valid_depth = 0.12 <= cup_depth <= 0.40

    right_slice = recent.iloc[bottom_bar:]
    if right_slice.empty:
        return {"detected": False}
    right_high = float(right_slice["High"].max())
    recovery = right_high >= left_price * 0.92

    handle_start = max(bottom_bar, len(recent) - max(5, lookback // 5))
    handle_bars = recent.iloc[handle_start:]
    if handle_bars.empty:
        return {"detected": False}
    handle_range = (float(handle_bars["High"].max()) - float(handle_bars["Low"].min())) / left_price
    small_handle = handle_range < 0.10

    detected = valid_depth and recovery and small_handle
    confidence = 0
    if detected:
        confidence = 70
        if 0.15 <= cup_depth <= 0.30:
            confidence += 15
        if right_high >= left_price * 0.97:
            confidence += 15

    return {
        "detected": detected,
        "confidence": min(100, confidence),
        "cup_depth": round(cup_depth * 100, 1),
        "recovery_pct": round((right_high / left_price - 1) * 100, 1),
    }


def detect_double_bottom(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Double Bottom: two real pivot lows at similar price levels separated by a
    meaningful bounce, with current price approaching or above the peak
    between them.

      1. Find all pivot lows in the window.
      2. Identify any two pivot lows within 4% of each other separated by
         >= 5 bars.
      3. Verify a meaningful peak (>= 5%) exists between the two lows.
      4. Breakout = current price > that peak.
    """
    if len(df) < lookback:
        return {"detected": False}

    recent = df.tail(lookback)
    pivots = find_pivot_points(recent, left_bars=3, right_bars=3)
    pls = pivots["pivot_lows"]

    if len(pls) < 2:
        return {"detected": False}

    best = None
    for i in range(len(pls) - 1):
        for j in range(i + 1, len(pls)):
            p1, p2 = pls[i], pls[j]
            if p2["bar"] - p1["bar"] < 5:
                continue
            similarity = abs(p1["price"] - p2["price"]) / p1["price"]
            if similarity >= 0.04:
                continue

            between = recent.iloc[p1["bar"]:p2["bar"] + 1]
            if between.empty:
                continue
            peak = float(between["High"].max())
            peak_height = (peak - min(p1["price"], p2["price"])) / min(p1["price"], p2["price"])
            if peak_height < 0.05:
                continue

            if best is None or similarity < best[0]:
                best = (similarity, p1, p2, peak)

    if best is None:
        return {"detected": False}

    similarity, p1, p2, peak = best
    current_price = float(recent["Close"].iloc[-1])
    breakout = current_price > peak * 1.005

    confidence = 65
    if similarity < 0.02:
        confidence += 15
    if breakout:
        confidence += 20

    return {
        "detected": True,
        "confidence": min(100, confidence),
        "low_similarity": round(similarity * 100, 1),
        "breakout": breakout,
    }


def detect_ascending_triangle(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Ascending Triangle: flat pivot highs (resistance) with progressively
    higher pivot lows (rising support).

      1. Find pivot highs within 2.5% of each other (flat resistance).
      2. Require >= 2 such pivot highs.
      3. Confirm pivot lows are trending upward.
      4. Bonus: volume contraction.
    """
    if len(df) < lookback:
        return {"detected": False}

    recent = df.tail(lookback)
    pivots = find_pivot_points(recent, left_bars=3, right_bars=3)
    phs = pivots["pivot_highs"]
    pls = pivots["pivot_lows"]

    if len(phs) < 2 or len(pls) < 2:
        return {"detected": False}

    resistance = max(phs, key=lambda p: p["price"])["price"]
    flat_highs = [p for p in phs if abs(p["price"] - resistance) / resistance <= 0.025]
    multiple_touches = len(flat_highs) >= 2

    pl_prices = [p["price"] for p in pls]
    rising_lows = len(pl_prices) >= 2 and pl_prices[-1] > pl_prices[0] * 1.01

    mid = len(recent) // 2
    vol_first = recent.iloc[:mid]["Volume"].mean()
    vol_second = recent.iloc[mid:]["Volume"].mean()
    volume_contraction = vol_second < vol_first

    detected = multiple_touches and rising_lows
    confidence = 0
    if detected:
        confidence = 62
        if len(flat_highs) >= 3:
            confidence += 15
        if volume_contraction:
            confidence += 13
        n = len(pl_prices)
        if n >= 2 and pl_prices[-1] > pl_prices[0] * 1.04:
            confidence += 10

    return {
        "detected": detected,
        "confidence": min(100, confidence),
        "resistance_touches": len(flat_highs),
        "support_rising": rising_lows,
    }


def detect_bearish_flag(df: pd.DataFrame, lookback: int = 40) -> dict:
    """
    Bear Flag: a pivot low (pole bottom) followed by a tight, slightly-upward
    drift before price breaks lower again.

      1. Find the most recent pivot low within lookback bars.
      2. Verify the move INTO that pivot low was >= 5% decline.
      3. After the pivot low, check tight consolidation (< 8%) with slight
         upward drift.
      4. Volume dries up during flag.
    """
    if len(df) < lookback:
        return {"detected": False}

    recent = df.tail(lookback)
    pivots = find_pivot_points(recent, left_bars=3, right_bars=3)
    pls = pivots["pivot_lows"]

    if not pls:
        return {"detected": False}

    pole_bottom = pls[-1]
    pole_bar = pole_bottom["bar"]
    pole_price = pole_bottom["price"]

    if pole_bar < 3 or (len(recent) - 1 - pole_bar) < 5:
        return {"detected": False}

    pole_start_price = float(recent["Close"].iloc[max(0, pole_bar - 5)])
    initial_drop = (pole_start_price - pole_price) / pole_start_price if pole_start_price > 0 else 0
    strong_pole = initial_drop >= 0.05

    flag_bars = recent.iloc[pole_bar:]
    flag_high = float(flag_bars["High"].max())
    flag_low = float(flag_bars["Low"].min())
    flag_range = (flag_high - flag_low) / pole_price if pole_price > 0 else 1
    tight_flag = flag_range < 0.08

    upward_drift = float(flag_bars["Close"].iloc[-1]) > float(flag_bars["Close"].iloc[0])
    no_new_low = flag_low >= pole_price * 0.99

    vol_pole = recent.iloc[max(0, pole_bar - 5):pole_bar + 1]["Volume"].mean()
    vol_flag = flag_bars["Volume"].mean() if len(flag_bars) > 0 else vol_pole
    volume_decrease = vol_flag < vol_pole * 0.85

    detected = strong_pole and tight_flag and upward_drift and no_new_low
    confidence = 0
    if detected:
        confidence = 65
        if volume_decrease:
            confidence += 15
        if flag_range < 0.05:
            confidence += 10
        if initial_drop >= 0.10:
            confidence += 10

    return {
        "detected": detected,
        "confidence": min(100, confidence),
        "initial_drop": round(initial_drop * 100, 1),
        "consolidation_range": round(flag_range * 100, 1),
    }


def detect_double_top(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Double Top: two real pivot highs at similar price levels separated by a
    meaningful trough, with current price approaching or below that trough.

      1. Find all pivot highs in the window.
      2. Identify any two pivot highs within 4% of each other separated by
         >= 5 bars.
      3. Verify a meaningful trough (>= 5% drop) exists between them.
      4. Breakdown = current price < that trough.
    """
    if len(df) < lookback:
        return {"detected": False}

    recent = df.tail(lookback)
    pivots = find_pivot_points(recent, left_bars=3, right_bars=3)
    phs = pivots["pivot_highs"]

    if len(phs) < 2:
        return {"detected": False}

    best = None
    for i in range(len(phs) - 1):
        for j in range(i + 1, len(phs)):
            p1, p2 = phs[i], phs[j]
            if p2["bar"] - p1["bar"] < 5:
                continue
            similarity = abs(p1["price"] - p2["price"]) / p1["price"]
            if similarity >= 0.04:
                continue

            between = recent.iloc[p1["bar"]:p2["bar"] + 1]
            if between.empty:
                continue
            trough = float(between["Low"].min())
            trough_depth = (max(p1["price"], p2["price"]) - trough) / max(p1["price"], p2["price"])
            if trough_depth < 0.05:
                continue

            if best is None or similarity < best[0]:
                best = (similarity, p1, p2, trough)

    if best is None:
        return {"detected": False}

    similarity, p1, p2, trough = best
    current_price = float(recent["Close"].iloc[-1])
    breakdown = current_price < trough * 0.995

    confidence = 65
    if similarity < 0.02:
        confidence += 15
    if breakdown:
        confidence += 20

    return {
        "detected": True,
        "confidence": min(100, confidence),
        "high_similarity": round(similarity * 100, 1),
        "breakdown": breakdown,
    }


def detect_head_and_shoulders(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Head & Shoulders: three pivot highs where the middle (head) is highest,
    outer two (shoulders) are at similar levels, and price is near/below the
    neckline.

      1. Require at least 3 pivot highs.
      2. For every ordered combination of 3 pivots (not just consecutive
         ones — see module docstring): check head > both shoulders by >= 2%.
      3. Shoulders within 6% of each other.
      4. Neckline = average of the two troughs between the peaks.
      5. Breakdown = price < neckline.
    """
    if len(df) < lookback:
        return {"detected": False}

    recent = df.tail(lookback)
    pivots = find_pivot_points(recent, left_bars=3, right_bars=3)
    phs = pivots["pivot_highs"]

    if len(phs) < 3:
        return {"detected": False}

    best = None
    for i, j, k in combinations(range(len(phs)), 3):
        ls, head, rs = phs[i], phs[j], phs[k]
        if not (head["price"] > ls["price"] * 1.02 and head["price"] > rs["price"] * 1.02):
            continue
        shoulder_sim = abs(ls["price"] - rs["price"]) / ls["price"]
        if shoulder_sim >= 0.06:
            continue

        left_trough = float(recent.iloc[ls["bar"]:head["bar"] + 1]["Low"].min())
        right_trough = float(recent.iloc[head["bar"]:rs["bar"] + 1]["Low"].min())
        neckline = (left_trough + right_trough) / 2

        if best is None or shoulder_sim < best[0]:
            best = (shoulder_sim, ls, head, rs, neckline)

    if best is None:
        return {"detected": False}

    shoulder_sim, ls, head, rs, neckline = best
    current_price = float(recent["Close"].iloc[-1])
    near_neckline = current_price <= neckline * 1.04
    breakdown = current_price < neckline * 0.99

    if not near_neckline:
        return {"detected": False}

    confidence = 65
    if shoulder_sim < 0.03:
        confidence += 15
    if breakdown:
        confidence += 20

    return {
        "detected": True,
        "confidence": min(100, confidence),
        "shoulder_similarity": round(shoulder_sim * 100, 1),
        "breakdown": breakdown,
        "neckline": round(neckline, 2),
    }


def detect_descending_triangle(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Descending Triangle: flat pivot lows (support zone) with progressively
    lower pivot highs (declining resistance).

      1. Find pivot lows within 2.5% of each other (flat support).
      2. Require >= 2 such pivot lows.
      3. Confirm pivot highs are trending downward.
      4. Bonus: volume contraction.
    """
    if len(df) < lookback:
        return {"detected": False}

    recent = df.tail(lookback)
    pivots = find_pivot_points(recent, left_bars=3, right_bars=3)
    phs = pivots["pivot_highs"]
    pls = pivots["pivot_lows"]

    if len(pls) < 2 or len(phs) < 2:
        return {"detected": False}

    support = min(pls, key=lambda p: p["price"])["price"]
    flat_lows = [p for p in pls if abs(p["price"] - support) / support <= 0.025]
    multiple_touches = len(flat_lows) >= 2

    ph_prices = [p["price"] for p in phs]
    declining_highs = len(ph_prices) >= 2 and ph_prices[-1] < ph_prices[0] * 0.99

    mid = len(recent) // 2
    vol_first = recent.iloc[:mid]["Volume"].mean()
    vol_second = recent.iloc[mid:]["Volume"].mean()
    volume_contraction = vol_second < vol_first

    detected = multiple_touches and declining_highs
    confidence = 0
    if detected:
        confidence = 62
        if len(flat_lows) >= 3:
            confidence += 15
        if volume_contraction:
            confidence += 13
        if ph_prices[-1] < ph_prices[0] * 0.95:
            confidence += 10

    return {
        "detected": detected,
        "confidence": min(100, confidence),
        "support_touches": len(flat_lows),
        "highs_declining": declining_highs,
    }


def detect_patterns(df: pd.DataFrame) -> list[dict]:
    """
    Run every detector at its own default lookback (see module docstring for
    why this deviates from the reference app) and return all detected
    patterns, sorted by confidence descending.
    """
    patterns = []

    bull_flag = detect_bull_flag(df)
    if bull_flag["detected"]:
        patterns.append({
            "type": "Bull Flag",
            "confidence": bull_flag["confidence"],
            "bias": "Bullish",
            "description": f"Strong move up ({bull_flag['initial_gain']}%) followed by tight consolidation",
            "action": "Buy breakout above consolidation high with volume",
        })

    cup_handle = detect_cup_and_handle(df)
    if cup_handle["detected"]:
        patterns.append({
            "type": "Cup and Handle",
            "confidence": cup_handle["confidence"],
            "bias": "Bullish",
            "description": f"U-shaped recovery ({cup_handle['cup_depth']}% depth) with handle",
            "action": "Buy breakout above handle high",
        })

    double_bottom = detect_double_bottom(df)
    if double_bottom["detected"]:
        patterns.append({
            "type": "Double Bottom",
            "confidence": double_bottom["confidence"],
            "bias": "Bullish",
            "description": f"Two lows at similar price ({double_bottom['low_similarity']}% apart)",
            "action": "Already breaking out" if double_bottom["breakout"] else "Buy breakout above middle peak",
        })

    asc_triangle = detect_ascending_triangle(df)
    if asc_triangle["detected"]:
        patterns.append({
            "type": "Ascending Triangle",
            "confidence": asc_triangle["confidence"],
            "bias": "Bullish",
            "description": f"Flat resistance with {asc_triangle['resistance_touches']} touches, rising support",
            "action": "Buy breakout above resistance with volume surge",
        })

    bear_flag = detect_bearish_flag(df)
    if bear_flag["detected"]:
        patterns.append({
            "type": "Bear Flag",
            "confidence": bear_flag["confidence"],
            "bias": "Bearish",
            "description": f"Strong drop ({bear_flag['initial_drop']}%) followed by tight upward drift",
            "action": "Avoid long — breakdown below consolidation low targets further downside",
        })

    double_top = detect_double_top(df)
    if double_top["detected"]:
        patterns.append({
            "type": "Double Top",
            "confidence": double_top["confidence"],
            "bias": "Bearish",
            "description": f"Two highs at similar price ({double_top['high_similarity']}% apart)",
            "action": "Already breaking down" if double_top["breakdown"] else "Avoid long — breakdown below trough confirms reversal",
        })

    hs = detect_head_and_shoulders(df)
    if hs["detected"]:
        patterns.append({
            "type": "Head and Shoulders",
            "confidence": hs["confidence"],
            "bias": "Bearish",
            "description": f"Classic reversal — neckline at ${hs['neckline']}, shoulders within {hs['shoulder_similarity']}%",
            "action": "Neckline broken — high risk" if hs["breakdown"] else "Avoid long — break below neckline signals trend reversal",
        })

    desc_triangle = detect_descending_triangle(df)
    if desc_triangle["detected"]:
        patterns.append({
            "type": "Descending Triangle",
            "confidence": desc_triangle["confidence"],
            "bias": "Bearish",
            "description": f"Flat support with {desc_triangle['support_touches']} touches, declining highs",
            "action": "Avoid long — breakdown below support triggers measured move lower",
        })

    patterns.sort(key=lambda p: p["confidence"], reverse=True)
    return patterns


BULLISH_MAX_BONUS = 15
BEARISH_MAX_PENALTY = -18


def evaluate_pattern_score(patterns: list[dict]) -> dict:
    """
    SmartScore adjustment from the single highest-confidence detected pattern
    (patterns is already confidence-sorted by detect_patterns) — only the
    most prominent signal moves the score, so conflicting simultaneous
    patterns don't cancel out into a meaningless wash. Bonus/penalty scales
    linearly with the pattern's own 0-100 confidence.
    """
    if not patterns:
        return {
            "triggered": False, "score_adjustment": 0, "flag": None,
            "pattern_name": None, "pattern_confidence": None, "pattern_action": None,
        }

    top = patterns[0]
    confidence = top["confidence"]
    flag_slug = top["type"].lower().replace(" ", "_")

    if top["bias"] == "Bullish":
        adjustment = round(BULLISH_MAX_BONUS * confidence / 100)
    else:
        adjustment = round(BEARISH_MAX_PENALTY * confidence / 100)

    return {
        "triggered": True,
        "score_adjustment": adjustment,
        "flag": f"pattern_{flag_slug}",
        "pattern_name": top["type"],
        "pattern_confidence": confidence,
        "pattern_action": top["action"],
    }
