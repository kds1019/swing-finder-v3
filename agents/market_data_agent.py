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
from core.pullback_reversal import detect_pullback_reversal
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

    def fetch_news(self, ticker: str, lookback_days: int, limit: int = 1000) -> pd.DataFrame:
        """Historical headlines/summaries for one ticker via Alpaca's free News API
        (Benzinga-sourced) — explicitly documented as usable for sentiment-model training,
        the data source behind core.sentiment's FinBERT scoring. Unlike bars this needs no
        feed/adjustment choice. include_content=False and exclude_contentless=True keep
        this to headline+summary text only, never full article bodies — cheap to score,
        and this repo has no need for more than that.

        An article can tag multiple tickers; NewsRequest(symbols=ticker) filters server-side
        to just this one, so no manual explode-by-symbol is needed the way a multi-symbol
        batched request would require. Per-ticker (not batched across tickers) mirrors
        agents.research_agent.ResearchAgent's get_insider_trades/get_rating_history/
        get_grade_history — one call per ticker, same as those, rather than a single
        multi-symbol call whose page-count cap could skew coverage toward whichever
        tickers happen to have more news.

        limit here is the *total* article count across the whole date range, not a
        per-page size — alpaca-py's NewsClient.get_news already paginates internally via
        next_page_token (confirmed by reading its _get_marketdata source) until either this
        total is reached or the range is exhausted, so this matches the limit=1000 default
        convention already used by get_insider_trades/get_rating_history/get_grade_history,
        not an arbitrarily small per-request page size.

        Returns empty DataFrame (not None) if nothing found, so callers can treat "no
        news" the same as "no insider trades" without a None-check."""
        from alpaca.data.historical import NewsClient
        from alpaca.data.requests import NewsRequest

        cols = ["Date", "headline", "summary"]
        news_client = NewsClient(self.settings.alpaca_api_key, self.settings.alpaca_secret_key)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(lookback_days * 2.5) + 5)

        request = NewsRequest(
            symbols=ticker, start=start, end=end, limit=limit,
            include_content=False, exclude_contentless=True,
        )
        news_set = news_client.get_news(request)
        df = news_set.df
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)

        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)
        for col in ["headline", "summary"]:
            if col not in df.columns:
                df[col] = ""
        return df[cols].sort_values("Date").reset_index(drop=True)

    def scan_universe(
        self, universe_df: pd.DataFrame, settings
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Screens every ticker in the universe for core.pullback_reversal's EMA200
        pullback + stabilization/reversal setup — replaces the old SmartScore
        (classify_setup Breakout/Pullback + ML-edge/volume-profile/chart-pattern
        adjustments) gating, which walk-forward testing found no demonstrated edge
        for (see docs/ml-edge-confidence-research.md). Returns (ranked_df,
        bars_by_ticker): ranked_df is every ticker where the setup was detected,
        sorted by bounce-off-low strength descending as a simple ordering signal —
        NOT a validated ranking, since this screener hasn't been walk-forward
        tested (an explicit choice, not an oversight — see the research doc's
        latest updates); bars_by_ticker is the raw indicator-augmented OHLCV per
        ticker, kept separately for downstream use.
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
            result = detect_pullback_reversal(df)
            if not result.get("detected"):
                continue

            bars_by_ticker[ticker] = df
            trade_plan = compute_trade_plan(df, settings)

            rows.append({
                "Ticker": ticker,
                "Sector": sector_lookup.get(ticker, "Unknown"),
                "Price": float(df["Close"].iloc[-1]),
                "Volume": float(df["Volume"].iloc[-1]),
                "EMA200UptrendPct": result["ema200_uptrend_pct"],
                "PriceVsEMA200Pct": result["price_vs_ema200_pct"],
                "ConsolidationRangePct": result["consolidation_range_pct"],
                "BounceOffLowPct": result["bounce_off_low_pct"],
                "Stop": trade_plan["stop"] if trade_plan else None,
                "Target": trade_plan["target"] if trade_plan else None,
                "RRRatio": trade_plan["rr_ratio"] if trade_plan else None,
                "WeakRR": trade_plan["weak_rr"] if trade_plan else None,
                "StopSanityFlag": trade_plan["stop_distance_sanity_flag"] if trade_plan else None,
            })

        ranked_df = pd.DataFrame(rows)
        if not ranked_df.empty:
            ranked_df = ranked_df.sort_values("BounceOffLowPct", ascending=False).reset_index(drop=True)

        return ranked_df, bars_by_ticker
