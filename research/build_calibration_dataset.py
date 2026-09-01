"""
Build the screener calibration dataset — Stage 1 of docs/strategy.md's plan.

Replays daily bars for a broad ticker list, and at every bar in a wide "net"
around the pullback region records:

  - the raw screener measurements (core.pullback_reversal.measure_pullback_reversal
    — continuous, NO thresholds applied)
  - candidate features not yet in the screener: ATR%, relative strength vs SPY
    (63 / 126 / 252 session), distance from the 252-session high, 50-EMA vs 200-EMA
  - whether the CURRENT thresholds would fire (`detected`) and, if not, why
  - the trade plan (core.trade_plan.compute_trade_plan) that would have been taken
  - the realised outcome over the next MAX_HOLD_DAYS bars
    (core.trade_plan.resolve_trade_plan_outcome) and the R-multiple

Output is one parquet: (features -> outcome), one row per (ticker, bar). Stage 2
then bins each feature against the R-multiple to place the thresholds from data
instead of from the single EMBJ reference trade.

Indicators are computed ONCE on each ticker's full history and then prefix-sliced
— every indicator here (ewm / rolling) is causal, so the value at bar i is
identical whether computed on the full series or on bars[:i+1]; there is no
look-ahead. Each per-bar screener/plan call is passed only a trailing
WINDOW_BARS slice, exactly mirroring what the live pipeline passes
(settings.bars_lookback_days).

Data caveat: Alpaca's free IEX feed only reaches back to ~mid-2020, so this is
~one full cycle. Enough for thousands of pattern instances; not enough to claim
regime-robustness. Same limitation the runs/ backtests carry.

Usage:
    python -m research.build_calibration_dataset --limit 400 --start 2020-01-01
    python -m research.build_calibration_dataset --tickers AAPL,MSFT,NVDA --start 2019-01-01
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config.settings import load_settings
from core.indicators import compute_indicators
from core.pullback_reversal import (
    CONSOLIDATION_MAX_RANGE_PCT,
    EMA200_MIN_UPTREND_PCT,
    MAX_PRICE_VS_VALUE_AREA_HIGH_PCT,
    MIN_BARS_FOR_SCREENER,
    MIN_BOUNCE_OFF_LOW_PCT,
    PRICE_VS_EMA200_MAX_PCT,
    PRICE_VS_EMA200_MIN_PCT,
    measure_pullback_reversal,
)
from core.trade_plan import compute_trade_plan, resolve_trade_plan_outcome
from core.universe import build_universe

MAX_HOLD_DAYS = 30  # matches core.pick_tracking.MAX_HOLD_DAYS (the triple-barrier horizon)
WINDOW_BARS = 300   # trailing bars handed to each screener/plan call — matches settings.bars_lookback_days

DATA_DIR = Path(__file__).resolve().parent / "data"
BARS_CACHE_DIR = DATA_DIR / "bars"
DEFAULT_OUT = DATA_DIR / "calibration_dataset.csv"

# Wide net: only bars in a rising-200-EMA name somewhere near the pullback region are
# labelled. Deliberately looser than the live screener's -12%/+8% / 15% / 3% gates so
# Stage 2 has room to move every cutoff in either direction — but not so loose that the
# dataset fills with bars nowhere near the pattern (and the per-bar trade-plan cost
# explodes). The current thresholds sit comfortably inside this net.
NET_MIN_EMA200_UPTREND_PCT = 0.0
NET_PRICE_VS_EMA200_MIN_PCT = -22.0
NET_PRICE_VS_EMA200_MAX_PCT = 14.0
NET_MAX_CONSOLIDATION_RANGE_PCT = 25.0
NET_MIN_BOUNCE_OFF_LOW_PCT = 0.0


def _current_verdict(m: dict) -> tuple[bool, str | None]:
    """Apply the CURRENT core.pullback_reversal thresholds to an already-computed
    measurement dict — same gate order as detect_pullback_reversal(), without paying
    to recompute the volume profile."""
    if m["ema200_uptrend_pct"] < EMA200_MIN_UPTREND_PCT:
        return False, "no_long_term_uptrend"
    if not (PRICE_VS_EMA200_MIN_PCT <= m["price_vs_ema200_pct"] <= PRICE_VS_EMA200_MAX_PCT):
        return False, "price_too_far_from_ema200"
    if m["consolidation_range_pct"] > CONSOLIDATION_MAX_RANGE_PCT:
        return False, "not_consolidating"
    if m["bounce_off_low_pct"] < MIN_BOUNCE_OFF_LOW_PCT:
        return False, "no_reversal_yet"
    if not m["volume_profile_available"]:
        return False, "insufficient_data"
    vah_pct = m["price_vs_value_area_high_pct"]
    if vah_pct is None or vah_pct > MAX_PRICE_VS_VALUE_AREA_HIGH_PCT:
        return False, "extended_above_value_area"
    return True, None


def _fetch_one(client: StockHistoricalDataClient, symbol: str, start: datetime) -> pd.DataFrame | None:
    """Full daily history for one symbol, IEX feed, split-adjusted — same feed/adjustment
    choice as agents.market_data_agent (this account is IEX-only; split adjustment keeps
    historical levels continuous). alpaca-py paginates internally."""
    alpaca_symbol = symbol.replace("-", ".")  # FMP dash form -> Alpaca dot form
    req = StockBarsRequest(
        symbol_or_symbols=alpaca_symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=datetime.now(timezone.utc),
        feed=DataFeed.IEX,
        adjustment=Adjustment.SPLIT,
    )
    try:
        bars = client.get_stock_bars(req)
    except Exception as e:  # noqa: BLE001 - research script, log and skip
        print(f"[calib] {symbol}: fetch failed: {e}", file=sys.stderr)
        return None
    df = bars.df
    if df is None or df.empty:
        return None
    df = df.reset_index()
    df = df.rename(columns={
        "timestamp": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].sort_values("Date").reset_index(drop=True)


def get_history(client, symbol: str, start: datetime, refresh: bool = False) -> pd.DataFrame | None:
    BARS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = BARS_CACHE_DIR / f"{symbol.replace('/', '_')}.pkl"
    if cache.exists() and not refresh:
        return pd.read_pickle(cache)
    df = _fetch_one(client, symbol, start)
    if df is not None and not df.empty:
        df.to_pickle(cache)
    return df


# Fixed R-multiple targets to sweep per trade (does a target at kR win before the
# -1R stop is hit?) — feeds the "is the fib-extension target too ambitious" analysis.
TARGET_SWEEP_KS = (1.0, 1.5, 2.0, 2.5, 3.0)


def _path_stats(after: pd.DataFrame, entry: float, stop: float, max_hold: int) -> dict:
    """Walk the post-entry bars once. Returns:
      mfe_r / mae_r  — max favourable / adverse excursion in R over the hold window
                       (window = up to the first stop touch, else the full horizon)
      won_{k}r       — would a fixed target at entry + k*risk have been hit strictly
                       BEFORE the -1R stop? Same-bar stop+target counts as the stop
                       (conservative, matches core.trade_plan.resolve_trade_plan_outcome)
    """
    risk = entry - stop
    ks = TARGET_SWEEP_KS
    if risk <= 0 or after.empty:
        return {"mfe_r": np.nan, "mae_r": np.nan, **{f"won_{k}r": np.nan for k in ks}}

    highs = after["High"].to_numpy(dtype=float)
    lows = after["Low"].to_numpy(dtype=float)
    n = min(len(after), max_hold)
    won = {k: False for k in ks}
    pending = set(ks)
    mfe = mae = 0.0
    for j in range(n):
        stopped = lows[j] <= stop
        hi_r = (highs[j] - entry) / risk
        lo_r = (lows[j] - entry) / risk
        mfe = max(mfe, hi_r)
        mae = min(mae, lo_r)
        for k in list(pending):
            if highs[j] >= entry + k * risk:
                won[k] = not stopped   # same-bar stop wins -> loss
                pending.discard(k)
        if stopped:
            break
    return {"mfe_r": float(mfe), "mae_r": float(mae), **{f"won_{k}r": won[k] for k in ks}}


def _vectorized_features(df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    """Every measurement that is a pure causal transform of the bar series — computed
    once over the whole frame. The per-bar loop only handles what can't be vectorised
    (volume profile, trade plan, forward-looking label)."""
    close, high = df["Close"], df["High"]
    ema200, ema50, atr = df["EMA200"], df["EMA50"], df["ATR14"]

    out = pd.DataFrame(index=df.index)
    out["ema200_uptrend_pct"] = (ema200 / ema200.shift(126) - 1.0) * 100.0
    out["price_vs_ema200_pct"] = (close / ema200 - 1.0) * 100.0
    roll_max_15 = close.rolling(15).max()
    roll_min_15 = close.rolling(15).min()
    out["consolidation_range_pct"] = (roll_max_15 - roll_min_15) / close * 100.0
    out["bounce_off_low_pct"] = (close / roll_min_15 - 1.0) * 100.0
    out["atr_pct"] = atr / close * 100.0
    out["dist_52w_high_pct"] = (close / high.rolling(252).max() - 1.0) * 100.0
    out["ema50_minus_ema200_pct"] = (ema50 / ema200 - 1.0) * 100.0
    out["ema50_gt_ema200"] = ema50 > ema200

    spy_aligned = df["Date"].map(spy_close)
    for k in (63, 126, 252):
        stock_ret = close / close.shift(k) - 1.0
        spy_ret = spy_aligned / spy_aligned.shift(k) - 1.0
        out[f"rs_vs_spy_{k}d_pct"] = (stock_ret - spy_ret) * 100.0
    return out


def label_ticker(symbol: str, df: pd.DataFrame, spy_close: pd.Series, settings) -> list[dict]:
    if df is None or len(df) < MIN_BARS_FOR_SCREENER + MAX_HOLD_DAYS + 5:
        return []

    df = compute_indicators(df.copy())
    vec = _vectorized_features(df, spy_close)

    net = (
        (vec["ema200_uptrend_pct"] > NET_MIN_EMA200_UPTREND_PCT)
        & (vec["price_vs_ema200_pct"].between(NET_PRICE_VS_EMA200_MIN_PCT, NET_PRICE_VS_EMA200_MAX_PCT))
        & (vec["consolidation_range_pct"] <= NET_MAX_CONSOLIDATION_RANGE_PCT)
        & (vec["bounce_off_low_pct"] >= NET_MIN_BOUNCE_OFF_LOW_PCT)
    )
    # Need a full forward horizon to label, and enough real history behind the bar that
    # EMA200 (and its 126-bar-back lookback) is meaningfully warmed up — 300 bars gives
    # ~175 bars of warmup at the lookback point. NOTE: the live pipeline computes
    # indicators on just a 300-bar window (MIN_BARS_FOR_SCREENER=127 minimum), so its
    # EMA200 is less converged than this; worth aligning when the thresholds are retuned.
    first_ok = max(MIN_BARS_FOR_SCREENER - 1, 300)
    last_ok = len(df) - MAX_HOLD_DAYS - 1
    candidate_idx = [i for i in np.where(net.to_numpy())[0] if first_ok <= i <= last_ok]

    rows: list[dict] = []
    for i in candidate_idx:
        prefix = df.iloc[: i + 1].tail(WINDOW_BARS)
        m = measure_pullback_reversal(prefix)
        if m is None:
            continue
        plan = compute_trade_plan(prefix, settings)
        if plan is None:
            continue

        after = df.iloc[i + 1 : i + 1 + MAX_HOLD_DAYS][["Date", "High", "Low", "Close"]].reset_index(drop=True)
        outcome, outcome_price, outcome_date, bars_to = resolve_trade_plan_outcome(
            after, plan["stop"], plan["target"], MAX_HOLD_DAYS
        )
        if outcome is None:
            continue  # not enough forward bars to resolve — shouldn't happen given last_ok, but be safe

        entry, stop = plan["entry"], plan["stop"]
        risk = entry - stop
        r_multiple = (outcome_price - entry) / risk if risk > 0 else np.nan

        path = _path_stats(after, entry, stop, MAX_HOLD_DAYS)
        detected, reject_reason = _current_verdict(m)

        rows.append({
            "ticker": symbol,
            "date": df["Date"].iloc[i],
            "close": m["close"],
            # --- current screener measurements (continuous) ---
            "ema200_uptrend_pct": m["ema200_uptrend_pct"],
            "price_vs_ema200_pct": m["price_vs_ema200_pct"],
            "consolidation_range_pct": m["consolidation_range_pct"],
            "bounce_off_low_pct": m["bounce_off_low_pct"],
            "price_vs_poc_pct": m["price_vs_poc_pct"],
            "price_vs_value_area_high_pct": m["price_vs_value_area_high_pct"],
            "volume_profile_available": m["volume_profile_available"],
            # --- candidate features ---
            "atr_pct": float(vec["atr_pct"].iloc[i]),
            "rs_vs_spy_63d_pct": float(vec["rs_vs_spy_63d_pct"].iloc[i]),
            "rs_vs_spy_126d_pct": float(vec["rs_vs_spy_126d_pct"].iloc[i]),
            "rs_vs_spy_252d_pct": float(vec["rs_vs_spy_252d_pct"].iloc[i]),
            "dist_52w_high_pct": float(vec["dist_52w_high_pct"].iloc[i]),
            "ema50_minus_ema200_pct": float(vec["ema50_minus_ema200_pct"].iloc[i]),
            "ema50_gt_ema200": bool(vec["ema50_gt_ema200"].iloc[i]),
            # --- current-threshold verdict ---
            "detected": detected,
            "reject_reason": reject_reason,
            # --- trade plan ---
            "entry": entry,
            "stop": stop,
            "target": plan["target"],
            "rr_ratio": plan["rr_ratio"],
            "weak_rr": plan["weak_rr"],
            # --- outcome (fib-extension target, floored 3:1 / capped 5:1) ---
            "outcome": outcome,
            "outcome_price": outcome_price,
            "bars_to_resolution": bars_to,
            "r_multiple": r_multiple,
            "return_pct": (outcome_price - entry) / entry * 100.0,
            # --- path stats: what the trade actually offered before it resolved ---
            "mfe_r": path["mfe_r"],
            "mae_r": path["mae_r"],
            **{f"won_{k}r": path[f"won_{k}r"] for k in TARGET_SWEEP_KS},
        })
    return rows


def load_ticker_list(args, settings) -> list[str]:
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.tickers_file:
        return [ln.strip().upper() for ln in Path(args.tickers_file).read_text().splitlines() if ln.strip()]
    universe = build_universe(settings)
    tickers = universe["Ticker"].tolist()
    if args.limit:
        rng = np.random.default_rng(args.seed)
        tickers = list(rng.choice(tickers, size=min(args.limit, len(tickers)), replace=False))
    # SPY is fetched separately in main() for the relative-strength benchmark — not needed here.
    return sorted(t for t in tickers if t != "SPY")


def main() -> None:
    p = argparse.ArgumentParser(description="Build the screener calibration dataset")
    p.add_argument("--tickers", type=str, default=None, help="Comma-separated symbols (overrides universe)")
    p.add_argument("--tickers-file", type=str, default=None, help="File with one symbol per line")
    p.add_argument("--limit", type=int, default=400, help="Random sample of the live universe (ignored with --tickers)")
    p.add_argument("--seed", type=int, default=7, help="Sample seed for --limit")
    p.add_argument("--start", type=str, default="2020-01-01", help="History start (YYYY-MM-DD)")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--refresh-bars", action="store_true", help="Ignore the bar cache and re-fetch")
    args = p.parse_args()

    settings = load_settings()
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        sys.exit("ALPACA_API_KEY / ALPACA_SECRET_KEY required.")

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)

    tickers = load_ticker_list(args, settings)
    print(f"[calib] {len(tickers)} tickers, history from {args.start}", file=sys.stderr)

    spy_df = get_history(client, "SPY", start, refresh=args.refresh_bars)
    if spy_df is None or spy_df.empty:
        sys.exit("Could not fetch SPY history — needed for relative strength.")
    spy_close = spy_df.set_index("Date")["Close"]

    all_rows: list[dict] = []
    t0 = time.time()
    for n, sym in enumerate(tickers, 1):
        if sym == "SPY":
            continue
        df = get_history(client, sym, start, refresh=args.refresh_bars)
        rows = label_ticker(sym, df, spy_close, settings)
        all_rows.extend(rows)
        if n % 25 == 0 or n == len(tickers):
            print(f"[calib] {n}/{len(tickers)} tickers  ({len(all_rows)} rows, {time.time()-t0:.0f}s)", file=sys.stderr)

    if not all_rows:
        sys.exit("No rows produced — check the ticker list / date range.")

    out_df = pd.DataFrame(all_rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    dec = out_df[out_df["detected"]]
    print(f"\n[calib] wrote {len(out_df):,} rows -> {args.out}", file=sys.stderr)
    print(f"[calib] {len(dec):,} match the CURRENT thresholds", file=sys.stderr)
    for label, sub in (("all wide-net rows", out_df), ("current-threshold matches", dec)):
        if sub.empty:
            continue
        wr = (sub["outcome"] == "target_hit").mean() * 100
        print(f"[calib]   {label}: hit-rate {wr:.1f}%  avg R {sub['r_multiple'].mean():.3f}  "
              f"median R {sub['r_multiple'].median():.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
