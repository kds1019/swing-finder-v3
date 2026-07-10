"""
Market Data Agent — Alpaca.

Primary data source for the whole universe scan (quotes, daily bars, intraday).
Batches multi-symbol requests (confirmed working: ~85-87 symbols x 60-day
lookback per call; Alpaca's 1MB response cap is the real constraint, not a
point-count limit — see core/universe.py::batch_tickers).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment

from core.universe import batch_tickers
from core.indicators import compute_indicators
from core.smartscore import compute_smartscore
from core.deep_discount_filter import evaluate_deep_discount_stabilization
from core.trade_plan import compute_trade_plan


def compute_market_bias(spy_df: pd.DataFrame | None) -> str | None:
    """SPY EMA20 vs EMA50 — informational context only (logged and included in
    pipeline.py's output). No longer feeds SmartScore classification — see
    core/smartscore.py's module-level comment for why the buffer that used to
    use this was removed."""
    if spy_df is None or len(spy_df) < 50:
        return None
    df = compute_indicators(spy_df.copy())
    last = df.iloc[-1]
    return "Uptrend" if float(last["EMA20"]) > float(last["EMA50"]) else "Downtrend"


class MarketDataAgent:
    def __init__(self, settings):
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are required for MarketDataAgent. "
                "Add them to your .env."
            )
        self.settings = settings
        self.client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)

    def _fetch_batch(self, symbols: list[str], lookback_days: int) -> pd.DataFrame | None:
        end = datetime.now(timezone.utc)
        # 2.5x calendar-day buffer so weekends/holidays still yield `lookback_days` trading days.
        start = end - timedelta(days=int(lookback_days * 2.5) + 5)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            # This account only has IEX (free-tier) market data access — SIP (all-exchange,
            # paid) returns "subscription does not permit querying recent SIP data".
            # Confirmed via a live 403 during initial testing; see README's feed note.
            feed=DataFeed.IEX,
            # Alpaca defaults to raw (unadjusted) bars when this isn't set — a ticker that
            # splits partway through a fetched window then shows a fake price discontinuity
            # (e.g. ARQQ's 1:25 reverse split in Nov 2024 looked like a +2471% 5-day return
            # in unadjusted bars). Split-adjusting keeps historical price levels continuous;
            # dividend adjustment is deliberately left out — swing entries/stops/targets are
            # actual tradeable prices, and dividend-adjusting would shift them off of that.
            adjustment=Adjustment.SPLIT,
        )
        bars = self.client.get_stock_bars(request)
        df = bars.df
        return df if df is not None and not df.empty else None

    def fetch_universe_bars(self, tickers: list[str], lookback_days: int | None = None) -> dict[str, pd.DataFrame]:
        """Returns {ticker: OHLCV DataFrame with a Date column}, batched per
        core.universe.batch_tickers to stay under Alpaca's response cap."""
        lookback_days = lookback_days or self.settings.bars_lookback_days
        result: dict[str, pd.DataFrame] = {}

        for batch in batch_tickers(tickers, self.settings.alpaca_batch_size):
            df = self._fetch_batch(batch, lookback_days)
            if df is None:
                continue

            for symbol in df.index.get_level_values(0).unique():
                sym_df = df.xs(symbol, level=0).reset_index()
                sym_df = sym_df.rename(columns={
                    "timestamp": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                })
                sym_df["Date"] = pd.to_datetime(sym_df["Date"]).dt.tz_localize(None)
                sym_df = sym_df.tail(lookback_days).reset_index(drop=True)
                result[symbol] = sym_df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        return result

    def fetch_spy_bars(self, lookback_days: int | None = None) -> pd.DataFrame | None:
        return self.fetch_universe_bars(["SPY"], lookback_days).get("SPY")

    def scan_universe(
        self, universe_df: pd.DataFrame, settings
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        SmartScore every ticker in the universe. Returns (ranked_df, bars_by_ticker):
        ranked_df is sorted SmartScore descending with the full factor breakdown;
        bars_by_ticker is the raw indicator-augmented OHLCV per ticker, kept
        separately (rather than embedded in ranked_df) for downstream use by
        the ML forecast / relative strength / multi-timeframe modules.
        """
        tickers = universe_df["Ticker"].tolist()
        sector_lookup = dict(zip(universe_df["Ticker"], universe_df["Sector"]))

        raw_bars = self.fetch_universe_bars(tickers, settings.bars_lookback_days)

        rows = []
        bars_by_ticker: dict[str, pd.DataFrame] = {}

        for ticker, df in raw_bars.items():
            if df is None or len(df) < 60:
                continue

            df = compute_indicators(df.copy())
            result = compute_smartscore(df, settings)
            if result.get("smartscore") is None:
                continue

            dd = evaluate_deep_discount_stabilization(df, result.get("fib_data"), result["rel_vol"])
            smartscore = max(0, min(100, result["smartscore"] + dd["penalty"]))

            bars_by_ticker[ticker] = df
            trade_plan = compute_trade_plan(df, settings)

            rows.append({
                "Ticker": ticker,
                "Sector": sector_lookup.get(ticker, "Unknown"),
                "Price": float(df["Close"].iloc[-1]),
                "Volume": float(df["Volume"].iloc[-1]),
                "SmartScore": smartscore,
                "Setup": result["setup"],
                "NearMiss": result["near_miss"],
                "NearType": result["near_type"],
                "Breakdown": result["breakdown"],
                "RelVolume": result["rel_vol"],
                "HasBase": result["has_base"],
                "AtMeaningfulLevel": result["at_meaningful_level"],
                "FibData": result.get("fib_data"),
                "DeepDiscountFlag": dd["flag"],
                "Stop": trade_plan["stop"] if trade_plan else None,
                "Target": trade_plan["target"] if trade_plan else None,
                "RRRatio": trade_plan["rr_ratio"] if trade_plan else None,
                "WeakRR": trade_plan["weak_rr"] if trade_plan else None,
                "StopSanityFlag": trade_plan["stop_distance_sanity_flag"] if trade_plan else None,
            })

        ranked_df = pd.DataFrame(rows)
        if not ranked_df.empty:
            ranked_df = ranked_df.sort_values("SmartScore", ascending=False).reset_index(drop=True)

        return ranked_df, bars_by_ticker
