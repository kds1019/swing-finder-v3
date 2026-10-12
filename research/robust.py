"""
Noise-aware evaluation helpers for the research/*_ab.py portfolio A/Bs.

Why this exists (found 2026-10-12): the same stop_floor A/B ran three times on 2026-09-30 and
the live config returned +29% / +63% / -17%; a local rebuild gave +148%. Two causes:

  1. The ticker sample drifted. `--limit 500 --seed 7` sampled from the LIVE FMP universe,
     which changes between runs, so the same seed drew a different ~460 tickers each time.
     Sector labels drifted the same way (tickers missing from today's universe became
     "Unknown" and shared one 3/sector cap slot).
  2. Portfolio results are path dependent. A 6-position book takes ~500 of ~40k signals;
     dropping a random 10% of tickers moved full-period return +56%..+125% on one fixed
     signal set, while per-trade avgR (0.16-0.24) and PF (1.26-1.42) barely moved.

Fixes provided here:
  - frozen_universe(): one committed ticker + sector list (research/frozen_universe.csv),
    so every A/B, every rerun and CI all test the same stocks with the same sector labels.
  - robust_compare(): runs every variant on the full set AND on N random ticker subsets,
    PAIRED (each subset is shared by all variants), and reports median/range plus how often
    each variant beat the baseline on the same subset. Judge variants on avgR / PF and the
    paired win counts, not on one full-sample total return.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

FROZEN_UNIVERSE = Path(__file__).resolve().parent / "frozen_universe.csv"


def frozen_universe(settings=None, limit: int | None = 500, seed: int = 7,
                    refreeze: bool = False) -> pd.DataFrame:
    """Ticker/Sector frame shared by every A/B. Created once from the live universe (a seeded
    sample of `limit` tickers) and committed; later calls just read the file. Pass
    refreeze=True to deliberately draw a new sample (that is a new experiment baseline —
    don't compare results across a refreeze)."""
    if FROZEN_UNIVERSE.exists() and not refreeze:
        return pd.read_csv(FROZEN_UNIVERSE)
    from core.universe import build_universe

    uni = build_universe(settings)[["Ticker", "Sector"]].dropna(subset=["Ticker"])
    uni = uni.drop_duplicates("Ticker").sort_values("Ticker").reset_index(drop=True)
    if limit and len(uni) > limit:
        idx = np.random.default_rng(seed).choice(len(uni), size=limit, replace=False)
        uni = uni.iloc[sorted(idx)].reset_index(drop=True)
    uni["Sector"] = uni["Sector"].fillna("Unknown")
    uni.to_csv(FROZEN_UNIVERSE, index=False)
    print(f"[robust] froze {len(uni)} tickers to {FROZEN_UNIVERSE.name}", file=sys.stderr)
    return uni


def freeze_from(tickers, sectors: dict) -> pd.DataFrame:
    """Freeze an explicit ticker list (e.g. everything already cached in research/data/bars)
    instead of sampling the live universe. Tickers with no sector label become 'Unknown'."""
    uni = pd.DataFrame({"Ticker": sorted(set(tickers))})
    uni["Sector"] = uni["Ticker"].map(sectors).fillna("Unknown")
    uni.to_csv(FROZEN_UNIVERSE, index=False)
    print(f"[robust] froze {len(uni)} tickers to {FROZEN_UNIVERSE.name} "
          f"({(uni.Sector == 'Unknown').sum()} without a sector)", file=sys.stderr)
    return uni


def ticker_subsets(tickers, n: int, frac: float = 0.9, seed: int = 0) -> list[set]:
    tickers = sorted(set(tickers))
    k = max(1, int(round(frac * len(tickers))))
    rng = np.random.default_rng(seed)
    return [set(rng.choice(tickers, size=k, replace=False)) for _ in range(n)]


def robust_compare(variants: dict[str, pd.DataFrame], run: Callable, stat: Callable,
                   bars, spy, spy_above, calendar, settings,
                   windows: list[tuple[str, str, str]], baseline: str | None = None,
                   n_subsamples: int = 10, frac: float = 0.9, seed: int = 0,
                   precomputed: dict | None = None) -> dict[str, pd.DataFrame]:
    """Per window, one row per variant: full-sample stats, subsample median/min/max, and paired
    win counts vs `baseline` (default: first variant). `precomputed` may hold full-sample
    run() results already computed by the caller, keyed by variant name."""
    names = list(variants)
    baseline = baseline or names[0]
    all_tickers = set().union(*(set(v["ticker"].unique()) for v in variants.values()))
    subsets = ticker_subsets(all_tickers, n_subsamples, frac, seed)

    full = {k: (precomputed or {}).get(k) or run(v, bars, spy_above, calendar, settings)
            for k, v in variants.items()}
    sub = {k: [] for k in names}
    for i, keep in enumerate(subsets, 1):
        for k, v in variants.items():
            sub[k].append(run(v[v["ticker"].isin(keep)], bars, spy_above, calendar, settings))
        print(f"[robust] subsample {i}/{len(subsets)} done", file=sys.stderr)

    out = {}
    for wn, lo, hi in windows:
        f = {k: stat(full[k], spy, lo, hi) for k in names}
        s = {k: [stat(r, spy, lo, hi) for r in sub[k]] for k in names}
        rows = []
        for k in names:
            if f[k] is None:
                continue
            ss = [x for x in s[k] if x is not None]
            col = lambda m: np.array([x[m] for x in ss], dtype=float)
            ret, avgr, pf, dd = col("ret"), col("avgR"), col("pf"), col("dd")
            row = {"variant": k, "full_ret": f[k]["ret"], "full_avgR": f[k]["avgR"],
                   "full_pf": f[k]["pf"], "trades": f[k]["n"],
                   "ret_med": np.nanmedian(ret), "ret_min": np.nanmin(ret), "ret_max": np.nanmax(ret),
                   "dd_med": np.nanmedian(dd), "avgR_med": np.nanmedian(avgr),
                   "pf_med": np.nanmedian(pf), "spy_ret": f[k]["spy_ret"]}
            if k != baseline:
                pairs = [(a, b) for a, b in zip(s[k], s[baseline]) if a is not None and b is not None]
                row["beats_ret"] = f"{sum(a['ret'] > b['ret'] for a, b in pairs)}/{len(pairs)}"
                row["beats_avgR"] = f"{sum(a['avgR'] > b['avgR'] for a, b in pairs)}/{len(pairs)}"
            else:
                row["beats_ret"] = row["beats_avgR"] = "baseline"
            rows.append(row)
        out[wn] = pd.DataFrame(rows)
    return out


def robust_markdown(tables: dict[str, pd.DataFrame], n_subsamples: int, frac: float) -> list[str]:
    L = [f"_Noise-aware view (research/robust.py): every variant run on the full frozen set plus "
         f"{n_subsamples} paired random {frac:.0%} ticker subsets. 'beats' = subsets where the "
         f"variant beat the baseline on that SAME subset. Decide on avgR/PF and the beats "
         f"counts; a single total return is mostly noise._", ""]
    for wn, d in tables.items():
        if d.empty:
            continue
        L += [f"### {wn}", "",
              "| variant | full ret | ret median [min..max] | maxDD med | avgR full / med | "
              "PF full / med | trades | beats base (ret) | beats base (avgR) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            L.append(f"| {r.variant} | {r.full_ret*100:+.0f}% | {r.ret_med*100:+.0f}% "
                     f"[{r.ret_min*100:+.0f}..{r.ret_max*100:+.0f}] | {r.dd_med*100:.0f}% | "
                     f"{r.full_avgR:+.2f} / {r.avgR_med:+.2f} | {r.full_pf:.2f} / {r.pf_med:.2f} | "
                     f"{r.trades} | {r.beats_ret} | {r.beats_avgR} |")
        L += [f"| SPY | {d.spy_ret.iloc[0]*100:+.0f}% | | | | | | | |", ""]
    return L
