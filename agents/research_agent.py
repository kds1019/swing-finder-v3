"""
Research Agent — Financial Modeling Prep (FMP).

Only called for the shortlist that survives SmartScore filtering + sector cap,
never the full 945-ticker universe — keeps API usage sane.

This also fixes a piece of dead code in swing-finder-v2: the reference app's
7-day scanner earnings filter never actually excluded anything because the
underlying per-ticker earnings-date fetch was commented out ("disabled to
avoid rate limiting during scan"). Here, earnings dates are fetched via FMP
for the shortlist only (not the whole universe), so the filter is real.

Uses FMP's newer /stable/ API, not /api/v3/ — confirmed live that v3 is fully
deprecated for keys created after 2025-08-31 ("Legacy Endpoint" 403 on every
v3 path tested, including basic quote/profile). Endpoint names and the
query-param-based symbol convention (?symbol=X, not /symbol/X) were verified
against the real API before writing this, not guessed from older docs.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import pandas as pd
import requests

FMP_BASE_URL = "https://financialmodelingprep.com/stable"


class ResearchAgent:
    def __init__(self, settings):
        if not settings.fmp_api_key:
            raise RuntimeError("FMP_API_KEY is required for ResearchAgent. Add it to your .env.")
        self.settings = settings
        self._api_key = settings.fmp_api_key
        self._session = requests.Session()

    def _get(self, path: str, params: Optional[dict] = None) -> dict | list:
        params = dict(params or {})
        params["apikey"] = self._api_key
        resp = self._session.get(f"{FMP_BASE_URL}/{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_vix_level(self) -> Optional[float]:
        """Current VIX price, via FMP's quote endpoint for the ^VIX index."""
        data = self._get("quote", params={"symbol": "^VIX"})
        if not data:
            return None
        return float(data[0]["price"])

    def get_vix_history(self, from_date: str, to_date: str) -> pd.DataFrame:
        """Historical VIX daily closes as a DataFrame with Date/vix columns —
        feeds core.ml_forecast.prepare_features' vix_df parameter."""
        data = self._get("historical-price-eod/full", params={"symbol": "^VIX", "from": from_date, "to": to_date})
        rows = data if isinstance(data, list) else []
        if not rows:
            return pd.DataFrame(columns=["Date", "vix"])
        df = pd.DataFrame(rows)[["date", "close"]].rename(columns={"date": "Date", "close": "vix"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)

    def get_earnings_calendar(self, tickers: list[str]) -> dict[str, Optional[int]]:
        """{ticker: days_to_earnings} using each ticker's next scheduled earnings date.
        None if no upcoming earnings date is available."""
        result: dict[str, Optional[int]] = {}
        today = pd.Timestamp.now().normalize()

        for ticker in tickers:
            try:
                data = self._get("earnings", params={"symbol": ticker, "limit": 10})
                dates = [pd.to_datetime(row["date"]) for row in data if row.get("date")]
                future_dates = [d for d in dates if d >= today]
                if future_dates:
                    next_date = min(future_dates)
                    result[ticker] = int((next_date - today).days)
                else:
                    result[ticker] = None
            except Exception:
                result[ticker] = None

        return result

    def get_fundamentals(self, ticker: str) -> dict:
        data = self._get("profile", params={"symbol": ticker})
        return data[0] if data else {}

    def get_earnings_history(self, ticker: str, limit: int = 8) -> list[dict]:
        """Trailing `limit` reported quarters' actual vs. estimated EPS/revenue —
        the real beat/met/missed history the research brief needs, not just the
        next-earnings-date lookup get_earnings_calendar() already does. Rows with
        epsActual still null (future/unreported quarters) are dropped — only
        reported history is meaningful for a "has this company been beating or
        missing" read."""
        try:
            data = self._get("earnings", params={"symbol": ticker, "limit": limit + 2})
        except requests.HTTPError as e:
            print(f"[research_agent] get_earnings_history({ticker}) failed: {e}", file=sys.stderr)
            data = []
        rows = data if isinstance(data, list) else []
        reported = [r for r in rows if r.get("epsActual") is not None]
        return reported[:limit]

    def get_income_growth(self, ticker: str, limit: int = 4) -> list[dict]:
        """Trailing `limit` quarters of income-statement growth rates (revenue,
        net income, EPS — quarter-over-quarter, per FMP's own growth convention)
        — the "trending up or down" data the research brief needs, distinct from
        a single-point-in-time profile snapshot."""
        try:
            data = self._get(
                "income-statement-growth", params={"symbol": ticker, "period": "quarter", "limit": limit}
            )
        except requests.HTTPError as e:
            print(f"[research_agent] get_income_growth({ticker}) failed: {e}", file=sys.stderr)
            data = []
        rows = data if isinstance(data, list) else []
        return [
            {
                "date": r.get("date"), "period": r.get("period"), "fiscalYear": r.get("fiscalYear"),
                "growthRevenue": r.get("growthRevenue"), "growthNetIncome": r.get("growthNetIncome"),
                "growthEPS": r.get("growthEPS"),
            }
            for r in rows[:limit]
        ]

    def get_analyst_ratings(self, ticker: str) -> dict:
        """Combines the ratings snapshot (overall rating + factor scores) with
        the analyst buy/hold/sell consensus — FMP splits these into two
        endpoints on the stable API (the old single "rating" endpoint is gone)."""
        snapshot_data = self._get("ratings-snapshot", params={"symbol": ticker})
        consensus_data = self._get("grades-consensus", params={"symbol": ticker})
        snapshot = snapshot_data[0] if snapshot_data else {}
        consensus = consensus_data[0] if consensus_data else {}
        return {**snapshot, **consensus}

    def get_news(self, ticker: str, limit: int = 5) -> list[dict]:
        data = self._get("news/stock", params={"symbols": ticker, "limit": limit})
        return data if isinstance(data, list) else []

    def get_insider_trades(self, ticker: str, limit: int = 1000) -> pd.DataFrame:
        """Form 4 insider transactions — filingDate/transactionType/acquisitionOrDisposition/
        securitiesTransacted/price. Feeds core.ml_forecast.prepare_features' insider_df
        parameter. filingDate (not transactionDate) is the causally correct date to key
        off — insiders can file up to a few days after the actual trade, so the market
        (and this model) only "knows" as of the filing, not the trade itself.

        Path is "insider-trading/search", not "search-insider-trades" — the latter is
        the display name FMP's own docs page (and the FMP MCP tool's internal endpoint
        alias) use, not the actual REST path; verified against a maintained third-party
        Python client's endpoint registry after the display-name guess silently 404'd on
        every call in production (caught by the except below, returning an empty frame
        for all 60 backtested tickers with no visible error)."""
        try:
            data = self._get("insider-trading/search", params={"symbol": ticker, "limit": limit})
        except requests.HTTPError as e:
            print(f"[research_agent] get_insider_trades({ticker}) failed: {e}", file=sys.stderr)
            data = []
        rows = data if isinstance(data, list) else []
        cols = ["filingDate", "transactionType", "acquisitionOrDisposition", "securitiesTransacted", "price"]
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows)
        df["filingDate"] = pd.to_datetime(df["filingDate"])
        return df[cols].sort_values("filingDate").reset_index(drop=True)

    def get_grade_history(self, ticker: str, limit: int = 1000) -> pd.DataFrame:
        """Individual sell-side analyst rating-change events (date/gradingCompany/
        previousGrade/newGrade/action, action in {"upgrade","downgrade","maintain",
        "initiate"} — FMP's own classification, not something this code has to infer from
        the free-text previousGrade/newGrade pair, which vary by grading firm's own scale
        ("Outperform" vs "Buy" vs "Overweight" etc. all mean roughly the same thing but
        aren't directly comparable across firms). Feeds core.ml_forecast.prepare_features'
        grades_df parameter, which only uses the action field for exactly that reason —
        this is genuinely different from get_rating_history()'s FMP-internal daily quant
        score (a fundamentals-ratio composite): this is real, dated sell-side analyst
        revision events, the actual "estimate revision momentum" data category flagged in
        docs/ml-edge-confidence-research.md as untested, not another transform of it.
        Verified live against the real /stable/grades endpoint before writing this — it
        does not honor `limit` server-side (returns full history regardless), so this
        truncates to the most recent `limit` rows client-side."""
        try:
            data = self._get("grades", params={"symbol": ticker})
        except requests.HTTPError as e:
            print(f"[research_agent] get_grade_history({ticker}) failed: {e}", file=sys.stderr)
            data = []
        rows = data if isinstance(data, list) else []
        cols = ["date", "gradingCompany", "previousGrade", "newGrade", "action"]
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows)[cols]
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").tail(limit).reset_index(drop=True)

    def get_rating_history(self, ticker: str, limit: int = 1000) -> pd.DataFrame:
        """Daily FMP quant rating score (overallScore, from historical-ratings — a
        ratio-based daily score, distinct from the monthly analyst buy/hold/sell
        consensus in get_analyst_ratings). Date/overallScore columns. Feeds
        core.ml_forecast.prepare_features' rating_df parameter."""
        try:
            data = self._get("historical-ratings", params={"symbol": ticker, "limit": limit})
        except requests.HTTPError as e:
            print(f"[research_agent] get_rating_history({ticker}) failed: {e}", file=sys.stderr)
            data = []
        rows = data if isinstance(data, list) else []
        if not rows:
            return pd.DataFrame(columns=["Date", "overallScore"])
        df = pd.DataFrame(rows)[["date", "overallScore"]].rename(columns={"date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)

    def enrich_shortlist(self, shortlist_df: pd.DataFrame, market_agent=None, news_lookback_days: int = 270) -> pd.DataFrame:
        """Adds DaysToEarnings, Fundamentals, AnalystRating, EarningsHistory, IncomeGrowth,
        and News columns to the post-screener/post-sector-cap shortlist. Never call this on
        the full universe — it's several FMP calls per ticker.

        News is now a genuine trend window (news_lookback_days, default ~9 months), not a
        5-headline snapshot — DecisionAgent's job changed from "mention News as background
        color" to "read it as the primary research basis," which needs enough history to
        judge a trend, not just whatever's most recent. Fetched via market_agent (Alpaca,
        agents.market_data_agent.MarketDataAgent.fetch_news) rather than FMP's news/stock
        endpoint, reusing the same mechanism research/walk_forward_backtest.py's FinBERT
        pipeline already relies on for exactly this kind of lookback-windowed fetch. Falls
        back to a short FMP-based snapshot (get_news's old 5-headline behavior) if
        market_agent isn't provided, so this still degrades gracefully rather than requiring
        a hard dependency change everywhere enrich_shortlist is called."""
        if shortlist_df.empty:
            return shortlist_df

        tickers = shortlist_df["Ticker"].tolist()
        earnings = self.get_earnings_calendar(tickers)

        enriched = shortlist_df.copy()
        enriched["DaysToEarnings"] = enriched["Ticker"].map(earnings)
        enriched["Fundamentals"] = enriched["Ticker"].apply(lambda t: self.get_fundamentals(t))
        enriched["AnalystRating"] = enriched["Ticker"].apply(lambda t: self.get_analyst_ratings(t))
        enriched["EarningsHistory"] = enriched["Ticker"].apply(lambda t: self.get_earnings_history(t))
        enriched["IncomeGrowth"] = enriched["Ticker"].apply(lambda t: self.get_income_growth(t))

        if market_agent is not None:
            def _fetch_news(ticker: str) -> list[dict]:
                try:
                    news_df = market_agent.fetch_news(ticker, lookback_days=news_lookback_days)
                    return json.loads(news_df.to_json(orient="records")) if not news_df.empty else []
                except Exception as e:
                    print(f"[research_agent] extended news fetch failed for {ticker}: {e}", file=sys.stderr)
                    return []
            enriched["News"] = enriched["Ticker"].apply(_fetch_news)
        else:
            enriched["News"] = enriched["Ticker"].apply(lambda t: self.get_news(t, limit=5))

        return enriched
