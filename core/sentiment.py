"""
FinBERT headline/summary sentiment scoring — the news-sentiment data category flagged in
docs/ml-edge-confidence-research.md as untested and, unlike insider trading/analyst
revisions/rating data, not yet disconfirmed by anything tried this session.

Model: ProsusAI/finbert (HuggingFace) — the standard, most widely-used open-source FinBERT
checkpoint, fine-tuned specifically on financial text rather than a general-purpose
sentiment model. Runs on CPU (no GPU assumed — GitHub Actions runners are CPU-only);
inference for a batch of headlines is seconds, not the "GPU recommended" multi-minute
full-article-body scoring the ml4t reference repo's own docs mention for its heavier
10_text_feature_engineering notebooks. This module only ever scores headlines/summaries,
never full article content, by design (see agents.market_data_agent.MarketDataAgent.
fetch_news's include_content=False).

torch/transformers are lazy-imported inside score_texts(), not at module level — mirrors
the existing lazy-import pattern for sklearn/lightgbm in core/ml_forecast.py and research/
triple_barrier_walk_forward.py, so importing this module doesn't force a multi-hundred-MB
model download/dependency load for callers that never actually score anything.
"""

from __future__ import annotations

import pandas as pd

FINBERT_MODEL = "ProsusAI/finbert"

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline("text-classification", model=FINBERT_MODEL, top_k=None)
    return _pipeline


def score_texts(texts: list[str], batch_size: int = 16) -> list[dict]:
    """One {"positive": p, "negative": n, "neutral": u} dict per input text, in the same
    order, each a full 3-way softmax (p+n+u == 1.0) — not just the top predicted label."""
    if not texts:
        return []
    clf = _get_pipeline()
    raw = clf(texts, batch_size=batch_size, truncation=True)
    return [{item["label"].lower(): float(item["score"]) for item in scores} for scores in raw]


def compute_net_sentiment(texts: list[str], batch_size: int = 16) -> list[float]:
    """positive - negative per text, in [-1, 1] — a single signed score per headline,
    the input build_sentiment_df() aggregates into a per-ticker time series."""
    scored = score_texts(texts, batch_size=batch_size)
    return [s.get("positive", 0.0) - s.get("negative", 0.0) for s in scored]


def build_sentiment_df(news_df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    """Scores each article's headline+summary with FinBERT exactly once, returning (Date,
    net_sentiment) — one row per article. This is the expensive step (a model forward
    pass per article); callers should run it once per ticker up front, the same way
    agents.research_agent.ResearchAgent's get_insider_trades/get_grade_history are fetched
    once per ticker before any walk-forward looping, not re-scored at every step.

    Deliberately headline+summary concatenated, not headline alone — more context per
    article without needing full article content (agents.market_data_agent.MarketDataAgent.
    fetch_news never fetches that). Output feeds core.ml_forecast.build_feature_table's
    sentiment_df parameter, which does the cheap part (rolling-window aggregation) at
    however many as-of dates a caller needs, same division of labor as insider_df's
    already-signed transaction rows."""
    if news_df.empty:
        return pd.DataFrame(columns=["Date", "net_sentiment"])
    texts = (news_df["headline"].fillna("") + ". " + news_df["summary"].fillna("")).tolist()
    net = compute_net_sentiment(texts, batch_size=batch_size)
    return pd.DataFrame({"Date": news_df["Date"].values, "net_sentiment": net})
