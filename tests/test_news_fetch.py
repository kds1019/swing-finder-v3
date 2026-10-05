"""MarketDataAgent.fetch_news on zero articles: alpaca-py's NewsSet.df raises
KeyError("None of ['id'] are in the columns") for an empty set, which surfaced as
"extended news fetch failed" for every no-news ticker. fetch_news must return an empty frame."""

import alpaca.data.historical as historical
from alpaca.data.models.news import NewsSet

from agents.market_data_agent import MarketDataAgent


def _article(i, headline, ts):
    return {"id": i, "headline": headline, "author": "a", "created_at": ts, "updated_at": ts,
            "summary": "s", "content": "", "url": "u", "images": [], "symbols": ["ETR"],
            "source": "benzinga"}


def _agent(monkeypatch, raw):
    class FakeNewsClient:
        def __init__(self, *a, **k):
            pass

        def get_news(self, request):
            return NewsSet(raw)

    monkeypatch.setattr(historical, "NewsClient", FakeNewsClient)
    agent = MarketDataAgent.__new__(MarketDataAgent)
    agent.settings = type("S", (), {"alpaca_api_key": "k", "alpaca_secret_key": "s"})()
    return agent


def test_fetch_news_empty_returns_empty_frame(monkeypatch):
    df = _agent(monkeypatch, {"news": []}).fetch_news("ETR", lookback_days=90)
    assert df.empty and list(df.columns) == ["Date", "headline", "summary"]


def test_fetch_news_parses_and_dedupes(monkeypatch):
    raw = {"news": [_article(2, "Same", "2026-10-01T12:05:00Z"),
                    _article(1, "Same", "2026-10-01T12:00:00Z"),
                    _article(3, "Other", "2026-09-20T09:00:00Z")]}
    df = _agent(monkeypatch, raw).fetch_news("ETR", lookback_days=90)
    assert df["headline"].tolist() == ["Other", "Same"]
    assert df["Date"].dt.tz is None
