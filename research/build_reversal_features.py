"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Enriches research/data/calibration_dataset.csv with the "has it stopped falling?"
candidate features requested in the V3 pullback/reversal audit:

  - higher-low / stabilization structure
  - fast-EMA (10/20) reclaim + slope
  - decline character (1d / 3d drops, gaps, ATR-normalised, waterfall concentration)
  - volume behaviour (down/up vol, expansion, effort-vs-result)
  - momentum (RSI level / turn / oversold depth / crude divergence, MACD hist)
  - support proximity (swing low, POC/VAH already in the base dataset)

Every feature is a pure causal transform of bars up to and including the signal bar
`date` (no look-ahead): computed vectorised over each ticker's full cached history,
then joined onto the existing (features -> outcome) rows by (ticker, date). The
outcome / R / MAE / trailing-R columns are taken as-is from the base dataset.

Output: research/data/reversal_features.csv   (base cols + rev_* cols)

Usage:  python -m research.build_reversal_features
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.indicators import compute_indicators

DATA = Path(__file__).resolve().parent / "data"
BASE = DATA / "calibration_dataset.csv"
BARS = DATA / "bars"
OUT = DATA / "reversal_features.csv"

PB_WIN = 15          # "pullback window" for decline-character features
LOW_WIN = 20         # window for the pullback low / higher-low structure


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """df: full cached OHLCV history for one ticker (Date/Open/High/Low/Close/Volume),
    ascending. Returns a frame indexed like df with rev_* columns, each valid at bar i
    using only bars <= i."""
    df = compute_indicators(df.copy())
    close, high, low = df["Close"], df["High"], df["Low"]
    o = pd.DataFrame(index=df.index)

    ema10 = _ema(df["Close"], 10)
    ema20 = df["EMA20"]
    atr = df["ATR14"]
    rsi = _rsi(df["Close"], 14)

    ret1 = df["Close"].pct_change() * 100
    ret3 = (df["Close"] / df["Close"].shift(3) - 1) * 100
    gap = (df["Open"] / df["Close"].shift(1) - 1) * 100

    # --- decline character (over the trailing PB_WIN bars) ---
    o["rev_max_1d_drop_pct"] = ret1.rolling(PB_WIN).min()
    o["rev_max_3d_drop_pct"] = ret3.rolling(PB_WIN).min()
    o["rev_worst_gap_pct"] = gap.rolling(PB_WIN).min()
    hi20 = high.rolling(LOW_WIN).max()
    o["rev_pct_from_20d_high"] = (close / hi20 - 1) * 100
    # pullback measured in ATR units (depth from the recent local peak, normalised)
    o["rev_decline_in_atr"] = (hi20 - close) / atr
    dn_day = (ret1 < 0)
    o["rev_down_days_10"] = dn_day.rolling(10).sum()
    # waterfall concentration: how much of the 15d peak->trough drop came from the 3 worst days
    tot_drop = (hi20 - low.rolling(LOW_WIN).min()).clip(lower=1e-9)
    worst3 = (-ret1.clip(upper=0) / 100 * close.shift(1)).rolling(PB_WIN).apply(
        lambda x: np.sort(x)[-3:].sum() if len(x) >= 3 else np.nan, raw=True
    )
    o["rev_waterfall_ratio"] = worst3 / tot_drop

    # --- higher-low / stabilisation structure ---
    low_win_min = low.rolling(LOW_WIN).min()
    # days since the LOW_WIN-bar low (0 = today is the low)
    def _days_since_min(x):
        return len(x) - 1 - int(np.argmin(x))
    o["rev_days_since_low_20"] = low.rolling(LOW_WIN).apply(_days_since_min, raw=True)
    o["rev_days_since_low_10"] = low.rolling(10).apply(_days_since_min, raw=True)
    recent_low_3 = low.rolling(3).min()
    o["rev_higher_low_pct"] = (recent_low_3 - low_win_min) / low_win_min * 100
    o["rev_made_new_low_3d"] = (low.rolling(3).min() <= low_win_min.shift(0)).astype(float)
    o["rev_close_up_days_5"] = (ret1 > 0).rolling(5).sum()
    o["rev_last_5d_return_pct"] = (close / close.shift(5) - 1) * 100
    o["rev_last_10d_return_pct"] = (close / close.shift(10) - 1) * 100

    # --- fast-EMA behaviour ---
    o["rev_close_vs_ema10_pct"] = (close / ema10 - 1) * 100
    o["rev_close_vs_ema20_pct"] = (close / ema20 - 1) * 100
    below20 = (close < ema20)
    # consecutive bars closed below EMA20 as of today (0 if above today)
    grp = (~below20).cumsum()
    o["rev_days_below_ema20"] = below20.groupby(grp).cumsum()
    o["rev_ema20_slope_5d_pct"] = (ema20 / ema20.shift(5) - 1) * 100
    o["rev_ema10_gt_ema20"] = (ema10 > ema20).astype(float)
    # reclaim: was below EMA20 at least 3 of the prior 6 bars, above now
    o["rev_reclaimed_ema20"] = ((close > ema20) & (below20.shift(1).rolling(6).sum() >= 3)).astype(float)

    # --- volume behaviour ---
    avgvol20 = df["Volume"].rolling(20).mean()
    up_v = df["Volume"].where(ret1 > 0)
    dn_v = df["Volume"].where(ret1 < 0)
    o["rev_down_up_vol_ratio_12"] = (dn_v.rolling(12, min_periods=2).mean()
                                     / up_v.rolling(12, min_periods=2).mean())
    # relative volume ON the day that set the trailing-20 low, as seen from bar i
    relvol = (df["Volume"] / avgvol20).to_numpy()
    lowv = low.to_numpy()
    vlow = np.full(len(df), np.nan)
    for i in range(LOW_WIN - 1, len(df)):
        j = i - LOW_WIN + 1 + int(np.argmin(lowv[i - LOW_WIN + 1: i + 1]))
        vlow[i] = relvol[j]
    o["rev_vol_at_low_rel"] = vlow
    o["rev_up_vol_expansion_5"] = up_v.rolling(5, min_periods=1).mean() / avgvol20
    o["rev_vol_trend_stab"] = df["Volume"].rolling(5).mean() / df["Volume"].shift(5).rolling(10).mean()

    # --- momentum ---
    o["rev_rsi14"] = rsi
    o["rev_rsi_min_10"] = rsi.rolling(10).min()
    o["rev_rsi_turn_up"] = ((rsi.diff() > 0) & (rsi.shift(1).diff() <= 0) & (rsi.rolling(5).min() < 40)).astype(float)
    o["rev_rsi_up_3d"] = (rsi - rsi.shift(3))
    # crude bullish divergence: today within 2% of the 20d closing low, but RSI >= 5 pts above
    # the RSI reading at that prior low
    rsi_at_low = rsi.where(close == close.rolling(LOW_WIN).min()).ffill()
    near_low = (close / close.rolling(LOW_WIN).min() - 1) < 0.02
    o["rev_rsi_bull_div"] = (near_low & (rsi - rsi_at_low >= 5)).astype(float)
    macd = _ema(df["Close"], 12) - _ema(df["Close"], 26)
    macd_hist = macd - _ema(macd, 9)
    o["rev_macd_hist"] = macd_hist / close * 100
    o["rev_macd_hist_up"] = (macd_hist.diff() > 0).astype(float)

    # --- support proximity ---
    o["rev_dist_to_swing_low_10_pct"] = (close / low.rolling(10).min() - 1) * 100
    o["rev_dist_to_swing_low_20_pct"] = (close / low_win_min - 1) * 100

    # --- pre-pullback trend quality ---
    ema50 = df["EMA50"]
    o["rev_ema50_slope_20d_pct"] = (ema50 / ema50.shift(20) - 1) * 100
    o["rev_ema50_slope_10d_pct"] = (ema50 / ema50.shift(10) - 1) * 100
    o["rev_pct_bars_above_ema50_126"] = (close > ema50).rolling(126).mean() * 100
    # how linear was the prior advance: corr(close, time) over the 120 bars ending 15 bars ago
    def _lin(x):
        if np.isnan(x).any():
            return np.nan
        t = np.arange(len(x))
        return np.corrcoef(t, x)[0, 1]
    o["rev_trend_linearity_120"] = close.shift(PB_WIN).rolling(120).apply(_lin, raw=True)

    o["Date"] = df["Date"].values
    return o


def main() -> None:
    if not BASE.exists():
        sys.exit(f"{BASE} not found")
    base = pd.read_csv(BASE, parse_dates=["date"], low_memory=False)
    tickers = sorted(base.ticker.unique())
    print(f"[rev] {len(base):,} base rows, {len(tickers)} tickers", file=sys.stderr)

    feats = []
    for n, t in enumerate(tickers, 1):
        cache = BARS / f"{t}.pkl"
        if not cache.exists():
            continue
        raw = pd.read_pickle(cache)
        if raw is None or len(raw) < 260:
            continue
        raw = raw.sort_values("Date").reset_index(drop=True)
        f = _features_for_ticker(raw)
        f["ticker"] = t
        f = f.rename(columns={"Date": "date"})
        feats.append(f)
        if n % 100 == 0:
            print(f"[rev] {n}/{len(tickers)}", file=sys.stderr)

    fdf = pd.concat(feats, ignore_index=True)
    fdf["date"] = pd.to_datetime(fdf["date"]).dt.normalize()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()

    merged = base.merge(fdf, on=["ticker", "date"], how="left", validate="many_to_one")
    rev_cols = [c for c in merged.columns if c.startswith("rev_")]
    cov = merged[rev_cols].notna().mean().min()
    print(f"[rev] merged {len(merged):,} rows, {len(rev_cols)} rev_ cols, min coverage {cov:.2%}", file=sys.stderr)

    merged.to_csv(OUT, index=False)
    print(f"[rev] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
