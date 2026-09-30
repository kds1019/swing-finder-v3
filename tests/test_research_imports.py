"""Guards against research/ drifting from the live screener again.

1. Every research script and runs/*/run.py must import. On 2026-09-30 all 20 research/*_ab.py
   style scripts had been unimportable since the 2026-09-17 screener change removed
   PRICE_VS_EMA200_MAX_PCT, and nobody noticed until a backtest was needed.
2. core.pullback_reversal.screener_gate is the single gate definition:
   detect_pullback_reversal must agree with it on every bar.

Run: python -m tests.test_research_imports   (or pytest, if installed)
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.indicators import compute_indicators
from core.pullback_reversal import detect_pullback_reversal, measure_pullback_reversal, screener_gate

ROOT = Path(__file__).resolve().parent.parent


def _modules():
    for p in sorted((ROOT / "research").glob("*.py")):
        if p.stem != "__init__":
            yield f"research.{p.stem}"
    for p in sorted((ROOT / "runs").glob("*/run.py")):
        yield str(p.relative_to(ROOT).with_suffix("")).replace("/", ".")


def test_every_research_script_imports():
    failures = []
    argv = sys.argv
    try:
        sys.argv = ["x"]
        for name in _modules():
            try:
                importlib.import_module(name)
            except Exception as e:  # noqa: BLE001 - report every failure, not just the first
                failures.append(f"{name}: {type(e).__name__}: {e}")
    finally:
        sys.argv = argv
    assert not failures, "research scripts that no longer import:\n" + "\n".join(failures)


def test_detect_agrees_with_screener_gate():
    rng = np.random.default_rng(3)
    checked = detected = 0
    for k in range(8):
        n = 700
        r = rng.normal(0.0009 if k % 2 else 0.0, 0.02, n)
        c = 40 * np.exp(np.cumsum(r))
        h, lo = c * (1 + abs(rng.normal(0, .01, n))), c * (1 - abs(rng.normal(0, .01, n)))
        df = compute_indicators(pd.DataFrame({
            "Date": pd.bdate_range("2023-01-02", periods=n), "Open": c, "High": h, "Low": lo,
            "Close": c, "Volume": rng.integers(100_000, 1_000_000, n).astype(float)}))
        for i in range(300, n, 7):
            prefix = df.iloc[: i + 1].tail(300)
            m = measure_pullback_reversal(prefix)
            if m is None:
                continue
            gate, det = screener_gate(m), detect_pullback_reversal(prefix)
            assert det["detected"] == (gate is None), (i, gate, det)
            if gate is not None:
                assert det["reason"] == gate, (i, gate, det["reason"])
            checked += 1
            detected += det["detected"]
    assert checked > 100 and detected > 0, (checked, detected)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
