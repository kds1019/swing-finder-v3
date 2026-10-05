"""core.relative_context (52-week range + sector strength), the Near52wHigh / SectorLagging
flags, and PortfolioAgent.get_open_orders() pagination."""

import pandas as pd

from agents.portfolio_agent import PortfolioAgent
from core.relative_context import (
    attach_sector_strength, compute_52w_position, compute_sector_strength,
)
from pipeline import compute_precomputed_flags


def _bars(closes, high_pad=1.0, low_pad=1.0):
    c = pd.Series(closes, dtype=float)
    return pd.DataFrame({"Close": c, "High": c + high_pad, "Low": c - low_pad})


def test_52w_position_uses_last_252_sessions():
    # An old spike to 500 outside the 252-session window must not count as the high.
    closes = [500.0] + [100.0] * 50 + list(range(100, 300)) + [250.0] * 2
    out = compute_52w_position(_bars(closes))
    assert out["High52w"] == 300.0          # 299 close + 1 pad, inside the window
    assert out["Low52w"] == 99.0
    assert out["PctFrom52wHigh"] == round((250 / 300 - 1) * 100, 2)
    assert 0 < out["Range52wPosition"] < 1


def test_52w_position_short_history_is_all_none():
    assert all(v is None for v in compute_52w_position(_bars([10.0] * 50)).values())
    assert all(v is None for v in compute_52w_position(None).values())


def test_sector_strength_labels():
    spy = _bars([100.0] * 25 + [101.0])               # SPY +1% over 20 sessions
    etfs = {
        "XLK": _bars([100.0] * 25 + [105.0]),         # +5% -> +4pp leading
        "XLU": _bars([100.0] * 25 + [97.0]),          # -3% -> -4pp lagging
        "XLE": _bars([100.0] * 25 + [101.5]),         # +0.5pp inline
    }
    out = compute_sector_strength(etfs, spy)
    assert out["Technology"]["label"] == "leading" and out["Technology"]["rs_20d_pp"] == 4.0
    assert out["Utilities"]["label"] == "lagging"
    assert out["Energy"]["label"] == "inline"
    assert "Healthcare" not in out                    # no XLV bars -> omitted, not guessed
    assert compute_sector_strength(etfs, None) == {}

    df = attach_sector_strength(pd.DataFrame({"Sector": ["Utilities", "Unknown"]}), out)
    assert df.loc[0, "SectorStrength"] == "lagging" and df.loc[0, "SectorETF"] == "XLU"
    assert pd.isna(df.loc[1, "SectorStrength"])


def test_relative_context_flags():
    row = {"Ticker": "X", "Sector": "Utilities", "PctFrom52wHigh": -3.2, "SectorStrength": "lagging"}
    assert compute_precomputed_flags(row, set(), set()) == ["Near52wHigh", "SectorLagging"]
    row = {"Ticker": "X", "PctFrom52wHigh": -12.0, "SectorStrength": "leading"}
    assert compute_precomputed_flags(row, set(), set()) == []
    # Missing values (NaN from a DataFrame round-trip) never produce a flag.
    assert compute_precomputed_flags({"Ticker": "X", "PctFrom52wHigh": float("nan")}, set(), set()) == []


class _Resp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class _OrderV3:
    def __init__(self, pages, fail=False):
        self.pages, self.fail, self.keys_seen = pages, fail, []

    def list_order_open(self, account_id, pagination_key=None):
        if self.fail:
            raise AttributeError("no list_order_open")
        self.keys_seen.append(pagination_key)
        return _Resp(self.pages[pagination_key])

    def get_order_open(self, account_id):
        return _Resp({"data": [{"orders": [{"symbol": "LEGACY"}]}]})


def _agent(order_v3):
    agent = PortfolioAgent.__new__(PortfolioAgent)     # skip the Webull auth in __init__
    agent._account_id = "acct"
    agent.trade = type("T", (), {"order_v3": order_v3})()
    return agent


def test_open_orders_follow_pagination_key():
    pages = {
        None: {"data": [{"orders": [{"symbol": f"S{i}"}]} for i in range(10)], "pagination_key": "p2"},
        "p2": {"data": [{"orders": [{"symbol": "S10"}]}, {"orders": [{"symbol": "S11"}]}]},
    }
    o = _OrderV3(pages)
    df = _agent(o).get_open_orders()
    assert len(df) == 12 and o.keys_seen == [None, "p2"]
    legs = PortfolioAgent.flatten_open_orders(df)
    assert {"S0", "S11"} <= {leg["symbol"] for leg in legs}


def test_open_orders_fall_back_to_legacy_call():
    df = _agent(_OrderV3({}, fail=True)).get_open_orders()
    assert PortfolioAgent.flatten_open_orders(df) == [{
        "symbol": "LEGACY", "side": None, "status": None, "order_type": None,
        "total_quantity": None, "filled_quantity": None, "stop_price": None, "limit_price": None,
    }]
