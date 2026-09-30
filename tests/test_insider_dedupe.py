"""Regression test for agents.research_agent.summarize_insider_trades' multi-filer dedupe.

Fixture = LTH's real Form 4 rows (FMP insider-trading/search, pulled 2026-09-30), trimmed to
the fields the summary uses. Before the fix, the three Leonard Green filers (the fund plus
directors Danhakl and Galashan, all reporting the same shares) were summed: 18 sales,
-$1,095.0M. The real figure is 10 distinct sales, about -$400.9M.

Run: python -m tests.test_insider_dedupe   (or pytest, if installed)
"""

import pandas as pd

from agents.research_agent import summarize_insider_trades

FUND = "Green LTF Holdings II LP"
D, O = "director", "officer"


def _row(tdate, fdate, ttype, shares, price, cik, name, role):
    return {"filingDate": pd.Timestamp(fdate), "transactionDate": tdate, "transactionType": ttype,
            "acquisitionOrDisposition": "D", "securitiesTransacted": shares, "price": price,
            "reportingCik": cik, "reportingName": name, "typeOfOwner": role}


LTH_ROWS = [
    # 8/10 sponsor sale: fund split into 3 lines + the same total reported by 2 directors
    _row("2026-08-10", "2026-08-12", "S-Sale", 5025751, 43.16, "1886438", FUND, D),
    _row("2026-08-10", "2026-08-12", "S-Sale", 84836, 43.16, "1886438", FUND, D),
    _row("2026-08-10", "2026-08-12", "S-Sale", 8512, 43.16, "1886438", FUND, D),
    _row("2026-08-10", "2026-08-12", "S-Sale", 5119099, 43.16, "1162644", "DANHAKL JOHN G", D),
    _row("2026-08-10", "2026-08-12", "S-Sale", 5119099, 43.16, "1590836", "Galashan John Kristofer", D),
    # 8/26 sponsor sale, same pattern
    _row("2026-08-26", "2026-08-28", "S-Sale", 2826651, 43.80, "1886438", FUND, D),
    _row("2026-08-26", "2026-08-28", "S-Sale", 47715, 43.80, "1886438", FUND, D),
    _row("2026-08-26", "2026-08-28", "S-Sale", 4788, 43.80, "1886438", FUND, D),
    _row("2026-08-26", "2026-08-28", "S-Sale", 2879154, 43.80, "1162644", "DANHAKL JOHN G", D),
    _row("2026-08-26", "2026-08-28", "S-Sale", 2879154, 43.80, "1590836", "Galashan John Kristofer", D),
    # 8/28 in-kind distribution to LPs — J-Other, not a sale, must be excluded
    _row("2026-08-28", "2026-09-01", "J-Other", 853884, 0, "1886438", FUND, D),
    _row("2026-08-28", "2026-09-01", "J-Other", 853884, 0, "1162644", "DANHAKL JOHN G", D),
    # executives after Q2 earnings (7/31), mostly option exercise-and-sell
    _row("2026-07-31", "2026-08-04", "S-Sale", 433307, 45.0179, "1295679", "Akradi Bahram",
         "director, officer: FOUNDER & CEO"),
    _row("2026-07-31", "2026-08-04", "S-Sale", 4950, 46.0329, "1295679", "Akradi Bahram",
         "director, officer: FOUNDER & CEO"),
    _row("2026-07-31", "2026-08-04", "M-Exempt", 130000, 19.32, "1295458", "Buss Eric J",
         "officer: EVP & CHIEF ADMIN. OFFICER"),
    _row("2026-07-31", "2026-08-04", "S-Sale", 479240, 44.9721, "1295458", "Buss Eric J",
         "officer: EVP & CHIEF ADMIN. OFFICER"),
    _row("2026-07-31", "2026-08-04", "S-Sale", 47748, 44.8, "1923538", "Weaver Erik",
         "officer: EVP & Chief Financial Officer"),
    _row("2026-07-31", "2026-08-04", "S-Sale", 63203, 45.088, "1884062", "Javaheri Parham",
         "officer: EVP &PRESIDENT CLUB OPERATIONS"),
    _row("2026-09-03", "2026-09-08", "S-Sale", 5666, 44, "1884996", "Singh Ritadhwaja Jebens",
         "officer: EVP & CHIEF DIGITAL OFFICER"),
    _row("2026-09-09", "2026-09-11", "S-Sale", 18729, 42.0395, "1884996", "Singh Ritadhwaja Jebens",
         "officer: EVP & CHIEF DIGITAL OFFICER"),
    _row("2026-09-11", "2026-09-11", "S-Sale", 154912, 42, "1884996", "Singh Ritadhwaja Jebens",
         "officer: EVP & CHIEF DIGITAL OFFICER"),
    # outside the 90-day window (filed 6/05) — must be excluded
    _row("2026-06-04", "2026-06-05", "S-Sale", 2208580, 28.6, "1162644", "DANHAKL JOHN G", D),
]

NOW = pd.Timestamp("2026-09-30")


def test_lth_multi_filer_sponsor_sales_counted_once():
    s = summarize_insider_trades(pd.DataFrame(LTH_ROWS), now=NOW)
    assert s["sale_count"] == 10, s["sale_count"]
    assert s["purchase_count"] == 0
    assert abs(s["sale_value"] - 400_900_000) < 200_000, s["sale_value"]
    assert s["net_value"] == -s["sale_value"]
    by = s["sale_value_by_holder_type"]
    assert abs(by["fund_or_10pct_owner"] - 347_000_000) < 200_000, by
    assert abs(by["officer"] - 53_800_000) < 200_000, by
    top = s["top_sellers"]
    assert top[0]["name"] == FUND and top[0]["holder_type"] == "fund_or_10pct_owner", top[0]
    assert top[0]["shares"] == 5119099 + 2879154
    assert s["most_recent_sale_date"] == "2026-09-11"


def test_distinct_sellers_same_day_different_size_both_kept():
    rows = [_row("2026-09-10", "2026-09-11", "S-Sale", 1000, 10.0, "1", "A Person", O),
            _row("2026-09-10", "2026-09-11", "S-Sale", 2000, 10.0, "2", "B Person", O),
            _row("2026-09-12", "2026-09-13", "P-Purchase", 500, 9.0, "3", "C Person", D)]
    s = summarize_insider_trades(pd.DataFrame(rows), now=NOW)
    assert s["sale_count"] == 2 and s["sale_shares"] == 3000
    assert s["purchase_count"] == 1 and s["purchase_value"] == 4500
    assert s["net_value"] == 4500 - 30000


def test_no_rows_in_window_returns_zeros_and_empty_is_empty():
    old = [_row("2026-01-02", "2026-01-03", "S-Sale", 1000, 10.0, "1", "A Person", O)]
    s = summarize_insider_trades(pd.DataFrame(old), now=NOW)
    assert s["sale_count"] == 0 and s["top_sellers"] == [] and s["sale_value"] == 0
    assert summarize_insider_trades(pd.DataFrame(), now=NOW) == {}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
