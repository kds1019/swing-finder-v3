"""
RESEARCH ONLY — does not touch the live pipeline.

Bullish harmonic pattern detector (Gartley, Bat) for research/harmonic_ab.py.

Everything here is CAUSAL: the state reported for bar t uses only bars 0..t, so a backtest
sees exactly what the live pipeline would have seen on that day (no repainting).

How it works:
  1. ZigZag swing points. A swing high is CONFIRMED once price has fallen ZIGZAG_ATR_MULT x
     ATR14 below it (mirror for a swing low). Each pivot records the bar it was confirmed
     on, and is only usable from that bar onward. The most recent, still-unconfirmed
     extreme is never used.
  2. Every time a swing HIGH is confirmed, the last four confirmed pivots are checked as
     X(low) A(high) B(low) C(high). If the AB and BC retracements fit a pattern below, a
     candidate is opened and price is followed bar by bar as the C->D leg develops.
  3. D is "today's price": the pattern is in its potential reversal zone (PRZ) when the
     A->price retracement of XA lies in the pattern's D band.

Pattern definitions (Carney's standard ratios; bands fixed BEFORE any test was run — do
not tune them to backtest results, that is exactly the curve-fitting this test guards against):

              AB / XA          BC / AB          D (retracement of XA)
  gartley     0.568 - 0.668    0.382 - 0.886    0.750 - 0.820   (0.786)
  bat         0.350 - 0.520    0.382 - 0.886    0.850 - 0.920   (0.886)

The usual BC-projection (CD/BC) confirmation is not required: it only re-states where D is,
and every extra ratio multiplies the ways a tolerance can be fit.

Candidate lifecycle (also fixed in advance):
  touched   the bar's LOW first reaches the PRZ top (retracement >= band low)
  failed    a LOW below X  (pattern invalidated)  -> ends
  ended     a HIGH above C before the PRZ was reached (the C->D leg never came), or
            close back above C after a touch (the bounce has run), or
            ACTIVE_MAX_BARS bars after the first touch                                -> ends
  in_zone   this bar's CLOSE retracement is inside the D band
  active    touched, not failed, not ended  (in the zone now or bouncing off it)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ZIGZAG_ATR_MULT = 2.5
ATR_PERIOD = 14
ACTIVE_MAX_BARS = 20

PATTERNS = {
    # name: (AB/XA band, BC/AB band, D band as retracement of XA)
    "gartley": ((0.568, 0.668), (0.382, 0.886), (0.750, 0.820)),
    "bat":     ((0.350, 0.520), (0.382, 0.886), (0.850, 0.920)),
}


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = ATR_PERIOD) -> np.ndarray:
    """Wilder ATR (same smoothing as research/supertrend_ab.py). Causal."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - close[:-1]),
                                np.abs(low[1:] - close[:-1])])
    atr = np.empty(n)
    atr[0] = tr[0]
    a = 1.0 / period
    for i in range(1, n):
        atr[i] = a * tr[i] + (1 - a) * atr[i - 1]
    return atr


def zigzag_pivots(high: np.ndarray, low: np.ndarray, atr: np.ndarray,
                  mult: float = ZIGZAG_ATR_MULT) -> list[tuple[int, float, str, int]]:
    """Confirmed pivots as (bar_idx, price, 'H'|'L', confirm_idx), alternating H/L.
    A pivot is confirmed on the first bar whose opposite extreme has moved mult*ATR(that
    bar) away from it; nothing before confirm_idx may use it."""
    n = len(high)
    pivots: list[tuple[int, float, str, int]] = []
    if n == 0:
        return pivots
    hi_i, lo_i = 0, 0          # running extremes while the first direction is unknown
    direction = 0              # +1 = tracking a swing high, -1 = tracking a swing low
    ext_i = 0
    for i in range(1, n):
        thr = mult * atr[i]
        if direction == 0:
            if high[i] > high[hi_i]:
                hi_i = i
            if low[i] < low[lo_i]:
                lo_i = i
            if high[i] - low[lo_i] >= thr and lo_i < i:
                pivots.append((lo_i, float(low[lo_i]), "L", i))
                direction, ext_i = 1, i
            elif high[hi_i] - low[i] >= thr and hi_i < i:
                pivots.append((hi_i, float(high[hi_i]), "H", i))
                direction, ext_i = -1, i
            continue
        if direction == 1:
            if high[i] >= high[ext_i]:
                ext_i = i
            elif high[ext_i] - low[i] >= thr:
                pivots.append((ext_i, float(high[ext_i]), "H", i))
                direction, ext_i = -1, i
        else:
            if low[i] <= low[ext_i]:
                ext_i = i
            elif high[i] - low[ext_i] >= thr:
                pivots.append((ext_i, float(low[ext_i]), "L", i))
                direction, ext_i = 1, i
    return pivots


def match_xabc(x: float, a: float, b: float, c: float) -> str | None:
    """Bullish pattern name whose AB and BC ratios fit X(low) A(high) B(low) C(high), else None."""
    xa = a - x
    if xa <= 0 or not (x < b < a) or not (b < c < a):
        return None
    ab = (a - b) / xa
    bc = (c - b) / (a - b)
    for name, ((ab_lo, ab_hi), (bc_lo, bc_hi), _) in PATTERNS.items():
        if ab_lo <= ab <= ab_hi and bc_lo <= bc <= bc_hi:
            return name
    return None


def harmonic_states(df: pd.DataFrame, mult: float = ZIGZAG_ATR_MULT) -> pd.DataFrame:
    """One row per (bar, open candidate) where a bullish pattern is ACTIVE (see module doc).

    Columns: idx, date, pattern, in_zone, first_touch (True on the first bar the PRZ was
    reached), x, a, b, c, prz_low, prz_high (prices), c_confirm_idx. A bar can carry more
    than one candidate (e.g. a Bat and a Gartley from different swing sets); callers
    usually keep the first per bar."""
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    close = df["Close"].astype(float).to_numpy()
    dates = df["Date"].to_numpy() if "Date" in df.columns else df.index.to_numpy()
    n = len(close)
    if n < 50:
        return pd.DataFrame()
    atr = _atr(high, low, close)
    piv = zigzag_pivots(high, low, atr, mult)

    rows = []
    for k in range(3, len(piv)):
        if piv[k][2] != "H":
            continue
        (xi, x, xt, _), (ai, a, at, _), (bi, b, bt, _), (ci, c, ct, c_conf) = piv[k - 3: k + 1]
        if (xt, at, bt, ct) != ("L", "H", "L", "H"):
            continue
        name = match_xabc(x, a, b, c)
        if name is None:
            continue
        d_lo, d_hi = PATTERNS[name][2]
        xa = a - x
        prz_top, prz_bot = a - d_lo * xa, a - d_hi * xa     # prices: top = shallower edge
        touched_at = None
        # Walk forward from the bar after C. Bars before c_conf are already known on c_conf, so
        # they count toward touch/fail, but nothing is emitted before the candidate is knowable.
        for t in range(ci + 1, n):
            if low[t] < x:
                break                                            # failed
            if touched_at is None:
                if high[t] > c:
                    break                                        # never reached D; leg ended
                if low[t] <= prz_top:
                    touched_at = t
            else:
                if close[t] > c or t - touched_at >= ACTIVE_MAX_BARS:
                    break
            if touched_at is not None and t >= c_conf:
                rows.append({"idx": t, "date": dates[t], "pattern": name,
                             "in_zone": bool(prz_bot <= close[t] <= prz_top),
                             "first_touch": t == max(touched_at, c_conf),
                             "x": x, "a": a, "b": b, "c": c,
                             "prz_low": prz_bot, "prz_high": prz_top, "c_confirm_idx": c_conf})
    return pd.DataFrame(rows)
