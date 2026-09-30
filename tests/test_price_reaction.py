"""Regression test for core.price_reaction, built from SRRK's real daily bars (Webull) and
real news timestamps (FMP/Benzinga), as of the 2026-09-30 scan.

The Decision Agent called SRRK's FDA approval a "recent" catalyst that "drove an ~12%
single-day pop". Real tape: approval printed 9/11 intraday (session -0.5%), the "12%" was a
9/14 premarket headline (session -6.4%), and by 9/30 the stock sat ~11% below its
pre-approval close.

Run: python -m tests.test_price_reaction   (or pytest, if installed)
"""

import pandas as pd

from core.price_reaction import annotate_news_with_reaction, recent_daily_bars

# date, close, volume — SRRK regular sessions 2026-08-12 .. 2026-09-30 (9/30 = intraday at scan)
_BARS = [
    ("2026-08-12", 53.82, 899250), ("2026-08-13", 53.06, 942958), ("2026-08-14", 52.52, 496597),
    ("2026-08-17", 52.05, 778851), ("2026-08-18", 54.72, 1217148), ("2026-08-19", 56.51, 1326268),
    ("2026-08-20", 55.25, 1534089), ("2026-08-21", 56.50, 1156418), ("2026-08-24", 58.20, 1222246),
    ("2026-08-25", 58.25, 1209520), ("2026-08-26", 60.04, 2281289), ("2026-08-27", 59.26, 620744),
    ("2026-08-28", 57.80, 809724), ("2026-08-31", 56.80, 987264), ("2026-09-01", 55.89, 763186),
    ("2026-09-02", 56.10, 1065938), ("2026-09-03", 55.94, 803486), ("2026-09-04", 55.33, 658048),
    ("2026-09-08", 57.42, 1836181), ("2026-09-09", 54.62, 1601822), ("2026-09-10", 55.67, 1175641),
    ("2026-09-11", 55.41, 994404), ("2026-09-14", 51.85, 4048433), ("2026-09-15", 50.00, 5251145),
    ("2026-09-16", 49.37, 3657746), ("2026-09-17", 49.02, 3485708), ("2026-09-18", 46.65, 3253096),
    ("2026-09-21", 48.16, 2904214), ("2026-09-22", 51.19, 3748564), ("2026-09-23", 49.85, 2492616),
    ("2026-09-24", 48.61, 2028236), ("2026-09-25", 48.63, 1875896), ("2026-09-28", 49.13, 1569484),
    ("2026-09-29", 47.75, 1066795), ("2026-09-30", 49.275, 313039),
]
BARS = pd.DataFrame(_BARS, columns=["Date", "Close", "Volume"])
NOW = pd.Timestamp("2026-09-30 16:10", tz="UTC")

# Alpaca-shaped items: "Date" is naive UTC
NEWS = [
    {"Date": "2026-09-11 17:27:56", "headline": "US FDA approves Scholar Rock's muscle weakness drug"},
    {"Date": "2026-09-14 05:50:18", "headline": "Why Scholar Rock Shares Are Trading Higher By Around 12%"},
    {"Date": "2026-09-21 22:30:00", "headline": "Scholar Rock Reports New Employee Inducement Grants"},
    {"Date": "2026-09-30 09:56:00", "headline": "Kuehn Law Encourages Investors of Scholar Rock"},
]


def _by_headline(items):
    return {i["headline"][:20]: i for i in items}


def test_fda_approval_reaction_shows_sell_the_news():
    a = _by_headline(annotate_news_with_reaction(NEWS, BARS, now=NOW))
    fda = a["US FDA approves Scho"]
    assert fda["reaction_session"] == "2026-09-11"      # 13:27 ET, before the close
    assert fda["session_chg_pct"] == -0.47              # 55.41 vs 55.67
    assert fda["age_days"] == 18
    assert fda["chg_since_pct"] == -11.49               # 49.275 vs 55.67


def test_premarket_headline_maps_to_that_days_session_not_its_claim():
    a = _by_headline(annotate_news_with_reaction(NEWS, BARS, now=NOW))
    pm = a["Why Scholar Rock Sha"]
    assert pm["reaction_session"] == "2026-09-14"
    assert pm["session_chg_pct"] == -6.42               # headline says +12%; the session closed -6.4%
    assert pm["session_rel_vol"] > 2.5                  # heavy selling volume


def test_after_close_item_rolls_to_next_session():
    a = _by_headline(annotate_news_with_reaction(NEWS, BARS, now=NOW))
    assert a["Scholar Rock Reports"]["reaction_session"] == "2026-09-22"   # 18:30 ET Mon -> Tue


def test_item_on_latest_session_and_missing_bars():
    a = _by_headline(annotate_news_with_reaction(NEWS, BARS, now=NOW))
    assert a["Kuehn Law Encourages"]["reaction_session"] == "2026-09-30"
    none = annotate_news_with_reaction(NEWS, None, now=NOW)[0]
    assert none["age_days"] == 18 and none["session_chg_pct"] is None


def test_recent_daily_bars_shape():
    rb = recent_daily_bars(BARS, n=5)
    assert [r["date"] for r in rb] == ["2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29", "2026-09-30"]
    assert abs(rb[-1]["close"] - 49.275) < 0.01 and rb[0]["chg_pct"] == -2.49
    assert recent_daily_bars(None) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
