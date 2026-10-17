"""research/harmonic.py — pattern recognition on hand-built swings, and causality (the state
reported for bar t must not change when later bars are added: no repainting).

Run: python -m tests.test_harmonic   (or pytest, if installed)
"""

import numpy as np
import pandas as pd

from research.harmonic import harmonic_states

COLS = ["pattern", "in_zone", "first_touch", "x", "a", "b", "c"]


def _path(points, n_per=12):
    px = []
    for a, b in zip(points[:-1], points[1:]):
        px += list(np.linspace(a, b, n_per, endpoint=False))
    px = np.array(px + [points[-1]])
    return pd.DataFrame({"Date": pd.bdate_range("2023-01-02", periods=len(px)),
                         "High": px * 1.005, "Low": px * 0.995, "Close": px})


LEAD = [80, 120, 100]   # X = the swing low at 100


def test_bat_detected():
    s = harmonic_states(_path(LEAD + [150, 128, 140, 105.7, 125]))   # AB 0.44, D 0.886
    assert set(s.pattern) == {"bat"} and s.in_zone.any() and s.first_touch.sum() == 1


def test_gartley_detected():
    s = harmonic_states(_path(LEAD + [150, 119, 135, 110.7, 130]))   # AB 0.62, D 0.786
    assert set(s.pattern) == {"gartley"} and s.in_zone.any()


def test_no_pattern_when_b_too_shallow():
    assert harmonic_states(_path(LEAD + [150, 140, 146, 105, 120])).empty


def test_ends_when_x_breaks():
    d = _path(LEAD + [150, 119, 135, 92, 90])
    s = harmonic_states(d)
    after_c = int(s.c_confirm_idx.iloc[0])
    x_break = after_c + int(np.argmax(d.Low.to_numpy()[after_c:] < s.x.iloc[0]))
    assert x_break > after_c and s.idx.max() < x_break


def _at(s, t):
    return [] if s.empty else sorted(map(tuple, s[s.idx == t][COLS].values.tolist()))


def test_causal():
    rng = np.random.default_rng(1)
    bad = checked = 0
    for k in range(15):
        px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, 600)))
        d = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=600),
                          "High": px * (1 + np.abs(rng.normal(0, 0.008, 600))),
                          "Low": px * (1 - np.abs(rng.normal(0, 0.008, 600))), "Close": px})
        full = harmonic_states(d)
        ts = set(rng.integers(60, 600, 8).tolist())
        if not full.empty:
            ts |= set(full[full.idx >= 60].idx.head(8).tolist())
        for t in ts:
            checked += 1
            bad += _at(full, t) != _at(harmonic_states(d.iloc[: t + 1]), t)
    assert bad == 0, f"{bad}/{checked} bars changed when later bars were added"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
