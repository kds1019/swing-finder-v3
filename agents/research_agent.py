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
        (and this model) only "knows" as of the filing, not the trade itself."""
        try:
            data = self._get("search-insider-trades", params={"symbol": ticker, "limit": limit})
        except requests.HTTPError:
            data = []
        rows = data if isinstance(data, list) else []
        cols = ["filingDate", "transactionType", "acquisitionOrDisposition", "securitiesTransacted", "price"]
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows)
        df["filingDate"] = pd.to_datetime(df["filingDate"])
        return df[cols].sort_values("filingDate").reset_index(drop=True)

    def get_rating_history(self, ticker: str, limit: int = 1000) -> pd.DataFrame:
        """Daily FMP quant rating score (overallScore, from historical-ratings — a
        ratio-based daily score, distinct from the monthly analyst buy/hold/sell
        consensus in get_analyst_ratings). Date/overallScore columns. Feeds
        core.ml_forecast.prepare_features' rating_df parameter."""
        try:
            data = self._get("historical-ratings", params={"symbol": ticker, "limit": limit})
        except requests.HTTPError:
            data = []
        rows = data if isinstance(data, list) else []
        if not rows:
            return pd.DataFrame(columns=["Date", "overallScore"])
        df = pd.DataFrame(rows)[["date", "overallScore"]].rename(columns={"date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)

    def enrich_shortlist(self, shortlist_df: pd.DataFrame) -> pd.DataFrame:
        """Adds DaysToEarnings, Fundamentals, AnalystRating, News columns to the
        post-SmartScore/post-sector-cap shortlist. Never call this on the full
        universe — it's several FMP calls per ticker."""
        if shortlist_df.empty:
            return shortlist_df

        tickers = shortlist_df["Ticker"].tolist()
        earnings = self.get_earnings_calendar(tickers)

        enriched = shortlist_df.copy()
        enriched["DaysToEarnings"] = enriched["Ticker"].map(earnings)
        enriched["Fundamentals"] = enriched["Ticker"].apply(lambda t: self.get_fundamentals(t))
        enriched["AnalystRating"] = enriched["Ticker"].apply(lambda t: self.get_analyst_ratings(t))
        enriched["News"] = enriched["Ticker"].apply(lambda t: self.get_news(t, limit=5))

        return enriched
