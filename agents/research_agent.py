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

# Short interest: FMP has no equivalent endpoint (confirmed live -- several plausible names
# ["short-interest", "shorts-interest", "short_interest", "stock-short-interest",
# "short-interest-history"] all 404 on this key/plan, and "quote-short" is an unrelated
# abbreviated-quote endpoint, not short-interest data), and Webull's OpenAPI
# financial_alert/financial_indicators don't carry it either. This is Nasdaq's own public
# short-interest API (undocumented, no key required, confirmed live) -- the standard
# bi-weekly FINRA-reported settlement data every US short-interest source ultimately derives
# from. Being undocumented, it could change/break without notice; get_short_interest() below
# degrades to {} on any failure, same as this module's other per-ticker fetches.
NASDAQ_SHORT_INTEREST_URL = "https://api.nasdaq.com/api/quote/{symbol}/short-interest"
_NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json",
}

_ITEM_DATE_KEYS = ("Date", "date", "publishedDate")


def _extract_item_date(item: dict) -> Optional[pd.Timestamp]:
    """Tolerant date extraction across the two News shapes enrich_shortlist can produce
    (Alpaca's "Date" vs FMP's "date"/"publishedDate" fallback), rather than assuming one
    fixed key."""
    for key in _ITEM_DATE_KEYS:
        value = item.get(key)
        if value:
            ts = parse_item_timestamp(value)
            if ts is not None:
                return ts
    return None


def parse_item_timestamp(value) -> Optional[pd.Timestamp]:
    """UTC timestamp from an ISO string/Timestamp, or from a bare number, which is treated
    as epoch MILLISECONDS (pandas' to_json default) — pd.to_datetime alone would read it as
    nanoseconds and land in 1970. None if unparseable."""
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return pd.to_datetime(value, unit="ms", utc=True)
        ts = pd.to_datetime(value, utc=True)
        return None if pd.isna(ts) else ts
    except (ValueError, TypeError, OverflowError):
        return None


def _catalyst_recency(news_items: list[dict]) -> dict:
    """Cheap recency signal derived entirely from News data already fetched (no extra
    API calls) — News carries ~1 quarter of history, so this surfaces "how fresh is
    the most recent item" as an explicit, structured field instead of leaving the
    Decision Agent to spot a handful of recent dates buried in a large chronological
    blob on its own. (An earlier version of this also folded in a separate FMP
    press-releases fetch, but that endpoint 403s under the Starter-tier FMP plan this
    project runs on — confirmed live, not assumed — and there's no working substitute,
    so it was dropped; production already sources News from Alpaca/Benzinga, which
    already carries official press-release-style items, not just aggregated commentary.)"""
    now = pd.Timestamp.now(tz="UTC")
    dates = [d for d in (_extract_item_date(item) for item in news_items) if d is not None]
    if not dates:
        return {"days_since_last_item": None, "items_last_3d": 0, "items_last_7d": 0}
    most_recent = max(dates)
    return {
        "days_since_last_item": int((now - most_recent).days),
        "items_last_3d": sum(1 for d in dates if (now - d).days <= 3),
        "items_last_7d": sum(1 for d in dates if (now - d).days <= 7),
    }


# Only these two transactionType codes are a genuine, voluntary open-market decision by
# the insider. Everything else FMP returns (confirmed live 2026-09-11 across several
# real tickers) is routine noise that would drown out the real signal if counted the same
# way: A-Award (stock granted as compensation, not bought), M-Exempt (option exercise,
# often paired with an immediate same-day sale that isn't itself a market view),
# F-InKind (shares surrendered to cover tax withholding, not a sell decision), G-Gift,
# J-Other, D-Return. A ticker's insider activity can be 80%+ these non-signal types.
INSIDER_PURCHASE_CODE = "P-Purchase"
INSIDER_SALE_CODE = "S-Sale"
INSIDER_LOOKBACK_DAYS = 90  # matches NEWS_LOOKBACK_DAYS's convention (pipeline.py)
INSIDER_TOP_SELLERS = 5

# A reporting person whose name reads like an investment vehicle, or who reports as a 10%
# owner group, is a fund/sponsor — its sales are usually a planned exit (secondary offering,
# block trade), not an operator's view on the business the way an officer's sale can be.
_FUND_NAME_TOKENS = (" LP", " L.P.", " LLC", " L.L.C.", "FUND", "HOLDINGS", "PARTNERS",
                     "CAPITAL", "MANAGEMENT", "INC", "LTD", "GROUP")


def _holder_type(name, role) -> str:
    """'fund_or_10pct_owner' / 'officer' / 'director' / 'other' for one reporting person."""
    n, r = f" {name or ''} ".upper(), (role or "").lower()
    if "10 percent" in r or "10%" in r or any(tok in n for tok in _FUND_NAME_TOKENS):
        return "fund_or_10pct_owner"
    if "officer" in r:
        return "officer"
    if "director" in r:
        return "director"
    return "other"


def _dedupe_insider_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse one real transaction reported by several Form 4 filers into a single row.

    When a fund sells, the fund AND each general-partner director who is deemed an
    indirect owner file their own Form 4 for the SAME shares — FMP returns each filing as a
    separate row. Confirmed live on LTH (2026-09-30): Leonard Green's Green LTF Holdings II
    LP plus directors Danhakl and Galashan each reported the same 5,119,099-share sale on
    8/10 and the same 2,879,154-share sale on 8/26 (the fund split its 8/26 filing across
    three lines summing to that figure). Summing every row tripled a ~$347M exit into
    ~$1.04B and turned 10 real sales into 18.

    Rule: within one (transactionDate, transactionType, price), total each reporting
    person's shares; reporters whose totals are identical are the same shares, so only the
    first (preferring a fund/owner-entity filer, which names the actual seller) is kept.
    Two different insiders independently trading the exact same share count at the exact
    same price on the same day would also be merged — accepted as far rarer than the
    multi-filer case this fixes. Returns one row per surviving (reporter, date, type,
    price) with columns: date, transactionType, price, shares, value, reportingName,
    typeOfOwner, holder_type."""
    if rows.empty:
        return pd.DataFrame(columns=["date", "transactionType", "price", "shares", "value",
                                     "reportingName", "typeOfOwner", "holder_type"])
    d = rows.copy()
    d["date"] = pd.to_datetime(d["transactionDate"].fillna(d["filingDate"]), errors="coerce")
    d["date"] = d["date"].fillna(d["filingDate"])
    d["price"] = pd.to_numeric(d["price"], errors="coerce").fillna(0.0).round(4)
    d["securitiesTransacted"] = pd.to_numeric(d["securitiesTransacted"], errors="coerce").fillna(0)
    d["reporter"] = d["reportingCik"].fillna(d["reportingName"]).astype(str)

    per_reporter = (
        d.groupby(["date", "transactionType", "price", "reporter"], sort=False)
        .agg(shares=("securitiesTransacted", "sum"), reportingName=("reportingName", "first"),
             typeOfOwner=("typeOfOwner", "first"))
        .reset_index()
    )
    per_reporter["holder_type"] = [
        _holder_type(n, r) for n, r in zip(per_reporter["reportingName"], per_reporter["typeOfOwner"])
    ]
    per_reporter["_pref"] = (per_reporter["holder_type"] != "fund_or_10pct_owner").astype(int)
    kept = (
        per_reporter.sort_values("_pref", kind="stable")
        .drop_duplicates(subset=["date", "transactionType", "price", "shares"], keep="first")
    )
    kept = kept.assign(value=kept["shares"] * kept["price"])
    return kept[["date", "transactionType", "price", "shares", "value", "reportingName",
                 "typeOfOwner", "holder_type"]].sort_values("date").reset_index(drop=True)


def summarize_insider_trades(df: pd.DataFrame, lookback_days: int = INSIDER_LOOKBACK_DAYS,
                             now: Optional[pd.Timestamp] = None) -> dict:
    """Genuine open-market insider buying/selling over the trailing `lookback_days`,
    filtered to INSIDER_PURCHASE_CODE/INSIDER_SALE_CODE only (see that comment for why
    the other transaction types are excluded rather than lumped in) and de-duplicated
    across multiple filers of the same shares (_dedupe_insider_rows). Returns {} if there
    is no insider data at all for this ticker (never raises).

    Complementary to short interest, not a restatement of it: an insider buying into a
    heavily-shorted stock is a real "the shorts may be wrong" signal; insiders selling
    alongside heavy shorting is the opposite — real alignment, not misplaced positioning.
    Confirmed live (2026-09-11): WULF (heavily shorted, 31.9% of float) had 2 real
    open-market purchases days before its pullback low; ESTA (already fundamentally
    weak — 2 straight EPS misses) had zero purchases and 24 sales over its recent
    history — insiders steadily selling, reinforcing rather than contradicting the
    already-bearish read.

    Fields:
      window_days
      purchase_count / sale_count — number of distinct open-market transactions (after
          de-duplication; one seller's same-day same-price fills count once)
      purchase_shares / sale_shares — total shares transacted
      purchase_value / sale_value — dollar value (shares * reported price)
      net_value — purchase_value - sale_value (positive = net insider buying)
      most_recent_purchase_date / most_recent_sale_date — None if none in the window
      sale_value_by_holder_type — {"fund_or_10pct_owner"/"officer"/"director"/"other": $}
      top_sellers — up to INSIDER_TOP_SELLERS {name, role, holder_type, shares, value,
          last_date}, largest first: WHO sold, so a sponsor exit reads differently from
          the CEO selling
    """
    if df is None or df.empty:
        return {}
    now = (now or pd.Timestamp.now()).normalize()
    cutoff = now - pd.Timedelta(days=lookback_days)
    recent = df[df["filingDate"] >= cutoff]
    recent = recent[recent["transactionType"].isin([INSIDER_PURCHASE_CODE, INSIDER_SALE_CODE])]
    tx = _dedupe_insider_rows(recent)

    purchases = tx[tx["transactionType"] == INSIDER_PURCHASE_CODE]
    sales = tx[tx["transactionType"] == INSIDER_SALE_CODE]
    purchase_value = float(purchases["value"].sum())
    sale_value = float(sales["value"].sum())

    top = []
    if len(sales):
        by_seller = (
            sales.groupby("reportingName", sort=False)
            .agg(role=("typeOfOwner", "first"), holder_type=("holder_type", "first"),
                 shares=("shares", "sum"), value=("value", "sum"), last_date=("date", "max"))
            .sort_values("value", ascending=False)
            .head(INSIDER_TOP_SELLERS)
            .reset_index()
        )
        top = [{"name": r.reportingName, "role": r.role, "holder_type": r.holder_type,
                "shares": int(r.shares), "value": round(float(r.value), 0),
                "last_date": str(r.last_date.date())} for r in by_seller.itertuples()]

    return {
        "window_days": lookback_days,
        "purchase_count": int(len(purchases)),
        "sale_count": int(len(sales)),
        "purchase_shares": int(purchases["shares"].sum()),
        "sale_shares": int(sales["shares"].sum()),
        "purchase_value": round(purchase_value, 0),
        "sale_value": round(sale_value, 0),
        "net_value": round(purchase_value - sale_value, 0),
        "most_recent_purchase_date": str(purchases["date"].max().date()) if len(purchases) else None,
        "most_recent_sale_date": str(sales["date"].max().date()) if len(sales) else None,
        "sale_value_by_holder_type": {
            k: round(float(v), 0) for k, v in sales.groupby("holder_type")["value"].sum().items()
        },
        "top_sellers": top,
    }


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
        """Combines the ratings snapshot (overall rating + factor scores), the
        analyst buy/hold/sell consensus, and a PRICE-TARGET REVISION signal — FMP
        splits these across endpoints on the stable API.

        The revision signal is the piece the Decision Agent was missing: consensus
        counts and the latest average target say where analysts stand, but not
        which way they're *moving*. `targetRevisionRecentPct` = last-month avg
        target vs last-quarter avg target; `targetRevisionMediumPct` = last-quarter
        vs last-year. Analysts actively cutting targets is a real headwind for a
        pullback-reversal entry regardless of how good the catalyst story reads
        (validated on the 2026-09-01 watchlist: it caught ON slashed -29% in a
        month with only +8% upside left, and NYT priced to its target, both of
        which the fundamentals-only read rated positively). `..._n` is the
        last-month analyst count — 0 or 1 means treat the signal as weak."""
        snapshot_data = self._get("ratings-snapshot", params={"symbol": ticker})
        consensus_data = self._get("grades-consensus", params={"symbol": ticker})
        snapshot = snapshot_data[0] if snapshot_data else {}
        consensus = consensus_data[0] if consensus_data else {}

        pts = {}
        try:
            pt_data = self._get("price-target-summary", params={"symbol": ticker})
            pt = pt_data[0] if isinstance(pt_data, list) and pt_data else {}
            mo, qtr, yr = (pt.get("lastMonthAvgPriceTarget"), pt.get("lastQuarterAvgPriceTarget"),
                           pt.get("lastYearAvgPriceTarget"))
            mo_n, qtr_n = pt.get("lastMonthCount") or 0, pt.get("lastQuarterCount") or 0
            recent = round((mo - qtr) / qtr * 100, 1) if (mo_n and qtr_n and qtr) else None
            medium = round((qtr - yr) / yr * 100, 1) if (qtr_n and yr) else None
            pts = {
                "lastMonthAvgTarget": mo, "lastMonthTargetCount": mo_n,
                "lastQuarterAvgTarget": qtr, "lastYearAvgTarget": yr,
                "targetRevisionRecentPct": recent,   # last month vs last quarter
                "targetRevisionMediumPct": medium,   # last quarter vs last year
            }
        except requests.HTTPError as e:
            print(f"[research_agent] price-target-summary({ticker}) failed: {e}", file=sys.stderr)

        return {**snapshot, **consensus, **pts}

    def get_short_interest(self, ticker: str) -> dict:
        """Bi-weekly FINRA-reported short interest via Nasdaq's own public API (see
        NASDAQ_SHORT_INTEREST_URL above for why FMP/Webull weren't usable) plus FMP's
        shares-float for percent-of-float. Settlement-date data is inherently up to ~2 weeks
        stale (FINRA's own reporting cadence) -- a real limitation of ALL short-interest data,
        not specific to this source; this is a positioning/risk read, not a timing signal.

        CONFIRMED LIVE (2026-09-11) COVERAGE GAP: Nasdaq's endpoint only covers Nasdaq-listed
        tickers -- an NYSE-listed ticker (e.g. KEY, NEE, KMI -- roughly half this project's
        universe, which spans NYSE/NASDAQ/AMEX) returns
        {"data": null, "message": "Short interest is only supported for Nasdaq Listed stocks"}
        and this method correctly degrades to {} for it, exactly like a genuine lookup
        failure. This was NOT caught by earlier testing (AAPL, CRUS) because both happen to be
        Nasdaq-listed. There is no NYSE-equivalent free/no-key public endpoint found so far
        (checked: NYSE's own site has none; financialdata.net's short-interest API requires a
        paid key). See docs/strategy.md for the options considered. Callers (decision_agent.py)
        are already instructed to treat an empty result as "unavailable", never as "no
        shorts" -- important given this gap means it WILL be empty for roughly half of all
        candidates, not just illiquid/obscure ones.

        Returns {} on any failure (network, unexpected shape, non-Nasdaq-listed ticker, ticker
        with no data) -- never raises. Fields:
          short_interest_shares / prior_short_interest_shares -- latest and prior settlement
          settlement_date / prior_settlement_date
          avg_daily_share_volume
          days_to_cover -- shares short / avg daily volume. Higher = more squeeze potential
              AND more downside fuel if the short thesis plays out and volume dries up --
              cuts both ways, not a directional signal by itself.
          short_interest_change_pct -- % change in shares short vs the PRIOR settlement
              (positive = shorts adding, negative = shorts covering/reducing) -- this is the
              trend-direction read: are shorts fighting the current setup or capitulating.
          short_percent_of_float -- shares short / floatShares, None if that lookup fails
        """
        def _num(v):
            try:
                return float(str(v).replace(",", ""))
            except (TypeError, ValueError):
                return None

        try:
            resp = self._session.get(
                NASDAQ_SHORT_INTEREST_URL.format(symbol=ticker),
                params={"assetclass": "stocks"}, headers=_NASDAQ_HEADERS, timeout=15,
            )
            resp.raise_for_status()
            rows = (resp.json().get("data") or {}).get("shortInterestTable", {}).get("rows") or []
        except Exception as e:
            print(f"[research_agent] get_short_interest({ticker}) failed: {e}", file=sys.stderr)
            return {}
        if not rows:
            return {}

        latest = rows[0]
        prior = rows[1] if len(rows) > 1 else None
        shares = _num(latest.get("interest"))
        avg_vol = _num(latest.get("avgDailyShareVolume"))
        try:
            days_to_cover = float(latest.get("daysToCover"))
        except (TypeError, ValueError):
            days_to_cover = (shares / avg_vol) if (shares and avg_vol) else None

        prior_shares = _num(prior.get("interest")) if prior else None
        change_pct = (
            round((shares - prior_shares) / prior_shares * 100, 1)
            if shares is not None and prior_shares else None
        )

        pct_float = None
        try:
            float_data = self._get("shares-float", params={"symbol": ticker})
            float_shares = float_data[0].get("floatShares") if float_data else None
            if shares is not None and float_shares:
                pct_float = round(shares / float_shares * 100, 2)
        except Exception as e:
            print(f"[research_agent] shares-float({ticker}) failed: {e}", file=sys.stderr)

        return {
            "short_interest_shares": shares,
            "prior_short_interest_shares": prior_shares,
            "settlement_date": latest.get("settlementDate"),
            "prior_settlement_date": prior.get("settlementDate") if prior else None,
            "avg_daily_share_volume": avg_vol,
            "days_to_cover": round(days_to_cover, 2) if days_to_cover is not None else None,
            "short_interest_change_pct": change_pct,
            "short_percent_of_float": pct_float,
        }

    def get_news(self, ticker: str, limit: int = 5) -> list[dict]:
        data = self._get("news/stock", params={"symbols": ticker, "limit": limit})
        return data if isinstance(data, list) else []

    def get_insider_trades(self, ticker: str, limit: int = 1000) -> pd.DataFrame:
        """Form 4 insider transactions — filingDate/transactionType/acquisitionOrDisposition/
        securitiesTransacted/price. Was built to feed core.ml_forecast.prepare_features'
        insider_df parameter; that system was deleted (see CLAUDE.md) and this went unused
        until summarize_insider_activity() below picked it up 2026-09-11. filingDate (not
        transactionDate) is the causally correct date to key off — insiders can file up to a
        few days after the actual trade, so the market only "knows" as of the filing, not
        the trade itself.

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
        cols = ["filingDate", "transactionDate", "transactionType", "acquisitionOrDisposition",
                "securitiesTransacted", "price", "reportingCik", "reportingName", "typeOfOwner"]
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows)
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df["filingDate"] = pd.to_datetime(df["filingDate"])
        return df[cols].sort_values("filingDate").reset_index(drop=True)

    INSIDER_PURCHASE_CODE = INSIDER_PURCHASE_CODE
    INSIDER_SALE_CODE = INSIDER_SALE_CODE
    INSIDER_LOOKBACK_DAYS = INSIDER_LOOKBACK_DAYS

    def summarize_insider_activity(self, ticker: str, lookback_days: int = INSIDER_LOOKBACK_DAYS) -> dict:
        """See summarize_insider_trades() — this just fetches the Form 4 rows for `ticker`."""
        return summarize_insider_trades(self.get_insider_trades(ticker), lookback_days)

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

    def enrich_shortlist(self, shortlist_df: pd.DataFrame, market_agent=None, news_lookback_days: int = 90) -> pd.DataFrame:
        """Adds DaysToEarnings, Fundamentals, AnalystRating, EarningsHistory, IncomeGrowth,
        ShortInterest, InsiderActivity, News, and CatalystRecency columns to the
        post-screener/post-sector-cap shortlist. Never call this on the full universe — it's
        several FMP calls per ticker.

        News is a real window (news_lookback_days, ~1 quarter of calendar days — enough for
        the latest earnings reaction and any recent catalyst), not a 5-headline snapshot —
        DecisionAgent's job is to "read it as the primary research basis," not just cite the
        most recent item. Wider windows were tried (270d) but ~doubled the Decision Agent
        prompt for little added signal. Fetched via market_agent (Alpaca,
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
        enriched["ShortInterest"] = enriched["Ticker"].apply(lambda t: self.get_short_interest(t))
        enriched["InsiderActivity"] = enriched["Ticker"].apply(lambda t: self.summarize_insider_activity(t))

        if market_agent is not None:
            def _fetch_news(ticker: str) -> list[dict]:
                try:
                    news_df = market_agent.fetch_news(ticker, lookback_days=news_lookback_days)
                    # date_format="iso": the pandas default ("epoch") serializes Date as epoch
                    # MILLISECONDS, which reached the Decision Agent as bare integers and which
                    # _catalyst_recency then parsed as nanoseconds — every item read as 1970,
                    # so CatalystRecency said ~20,700 days / 0 items in the last 7d for every
                    # ticker (confirmed 2026-09-30).
                    return (json.loads(news_df.to_json(orient="records", date_format="iso"))
                            if not news_df.empty else [])
                except Exception as e:
                    print(f"[research_agent] extended news fetch failed for {ticker}: {e}", file=sys.stderr)
                    return []
            enriched["News"] = enriched["Ticker"].apply(_fetch_news)
        else:
            enriched["News"] = enriched["Ticker"].apply(lambda t: self.get_news(t, limit=5))

        enriched["CatalystRecency"] = enriched["News"].apply(_catalyst_recency)

        return enriched
