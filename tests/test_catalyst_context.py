"""Tests for the catalyst-date plumbing around the Decision Agent.

1. News dates serialized by pandas' default to_json ("epoch" = milliseconds) used to be parsed
   as nanoseconds — every item read as 1970, so CatalystRecency reported ~20,700 days since
   the last item and 0 items in the last 7 days for every ticker (confirmed 2026-09-30).
2. pipeline.enforce_catalyst_recency: a "recent" catalyst_status needs a catalyst_date within
   CATALYST_RECENT_MAX_DAYS, else it's downgraded to "none" + CatalystStale.
3. pipeline.attach_price_context wires core.price_reaction onto the enriched frame.

Run: python -m tests.test_catalyst_context   (or pytest, if installed)
"""

import json

import pandas as pd

from agents.research_agent import _catalyst_recency
from core.price_reaction import annotate_news_with_reaction
from pipeline import attach_price_context, enforce_catalyst_recency
from tests.test_price_reaction import BARS, NOW

EPOCH_MS_ITEM = json.loads(
    pd.DataFrame({"Date": [pd.Timestamp("2026-09-28 14:00:00")], "headline": ["x"]})
    .to_json(orient="records")
)[0]
ISO_ITEM = json.loads(
    pd.DataFrame({"Date": [pd.Timestamp("2026-09-28 14:00:00")], "headline": ["x"]})
    .to_json(orient="records", date_format="iso")
)[0]


def test_epoch_ms_and_iso_dates_both_parse_to_2026():
    assert isinstance(EPOCH_MS_ITEM["Date"], int)
    for item in (EPOCH_MS_ITEM, ISO_ITEM):
        a = annotate_news_with_reaction([item], BARS, now=NOW)[0]
        assert a["age_days"] == 2 and a["reaction_session"] == "2026-09-28", a


def test_catalyst_recency_no_longer_reads_1970():
    r = _catalyst_recency([EPOCH_MS_ITEM])
    assert r["days_since_last_item"] is not None and r["days_since_last_item"] < 30, r
    assert r["items_last_7d"] == 1


def test_enforce_catalyst_recency():
    picks = [
        {"ticker": "SRRK", "catalyst_status": "recent", "catalyst_date": "2026-09-11", "flags": ["X"]},
        {"ticker": "CGNX", "catalyst_status": "recent", "catalyst_date": "2026-09-26", "flags": []},
        {"ticker": "NODATE", "catalyst_status": "recent", "flags": None},
        {"ticker": "FLEX", "catalyst_status": "upcoming", "catalyst_date": "2026-01-01", "flags": []},
    ]
    enforce_catalyst_recency(picks, now=pd.Timestamp("2026-09-30"))
    srrk, cgnx, nodate, flex = picks
    assert srrk["catalyst_status"] == "none" and srrk["catalyst_status_model"] == "recent"
    assert srrk["flags"] == ["X", "CatalystStale"]
    assert cgnx["catalyst_status"] == "recent" and "catalyst_status_model" not in cgnx
    assert nodate["catalyst_status"] == "none" and nodate["flags"] == ["CatalystStale"]
    assert flex["catalyst_status"] == "upcoming"   # only "recent" is date-checked


def test_attach_price_context():
    df = pd.DataFrame({"Ticker": ["SRRK", "MISSING"],
                       "News": [[ISO_ITEM], [ISO_ITEM]]})
    out = attach_price_context(df, {"SRRK": BARS})
    assert len(out.loc[0, "RecentDailyBars"]) == 30 and out.loc[1, "RecentDailyBars"] == []
    assert out.loc[0, "News"][0]["reaction_session"] == "2026-09-28"
    assert out.loc[1, "News"][0]["session_chg_pct"] is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
