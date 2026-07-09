"""
Technical indicators — ported verbatim from swing-finder-v2's utils/indicators.py.

Operates on a plain OHLCV DataFrame with columns Open/High/Low/Close/Volume and
a real Date column (no Tiingo/Streamlit coupling — data source is Alpaca bars
via agents/market_data_agent.py).
"""

import pandas as pd
import numpy as np


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    RSI using Wilder's smoothing method (matches Webull, TradingView, most
    platforms). This is the single RSI implementation used everywhere in this
    project — swing-finder-v2 had a second, inconsistent simple-rolling-mean
    RSI in multi_timeframe.py; that inconsistency is intentionally not carried
    over here.
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values.fillna(50)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = (df["High"] - df["Low"]).abs()
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds EMA20, EMA50, EMA200, RSI14, ATR14, BandPos20, HH20, LL20, AvgVol20, RelVolume."""
    if df.empty:
        return df

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    df["EMA20"] = ema(close, 20)
    df["EMA50"] = ema(close, 50)
    df["EMA200"] = ema(close, 200)

    df["RSI14"] = rsi(close, 14)
    df["ATR14"] = atr(df, 14)

    # Bollinger Band position (0-1 between low/high band, not clamped)
    mean = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mean + 2 * std
    lower = mean - 2 * std
    df["BandPos20"] = (close - lower) / (upper - lower)

    df["HH20"] = high.rolling(20).max()
    df["LL20"] = low.rolling(20).min()

    df["AvgVol20"] = df["Volume"].rolling(20).mean()
    df["RelVolume"] = df["Volume"] / df["AvgVol20"]

    return df


def find_pivot_points(df: pd.DataFrame, left_bars: int = 3, right_bars: int = 3) -> dict:
    """
    Find true pivot highs and lows using left/right bar comparison.

    A pivot HIGH at bar i requires:
        High[i] >= High[i-1..i-left_bars]  AND  High[i] >= High[i+1..i+right_bars]
    A pivot LOW at bar i requires:
        Low[i]  <= Low[i-1..i-left_bars]   AND  Low[i]  <= Low[i+1..i+right_bars]

    Returns {"pivot_highs": [{"bar": int, "price": float}], "pivot_lows": [...]}
    (chronological order).
    """
    n = len(df)
    pivot_highs = []
    pivot_lows = []

    highs = df["High"].values
    lows = df["Low"].values

    for i in range(left_bars, n - right_bars):
        h = highs[i]
        lo = lows[i]

        if (all(h >= highs[i - j] for j in range(1, left_bars + 1)) and
                all(h >= highs[i + j] for j in range(1, right_bars + 1))):
            pivot_highs.append({"bar": i, "price": float(h)})

        if (all(lo <= lows[i - j] for j in range(1, left_bars + 1)) and
                all(lo <= lows[i + j] for j in range(1, right_bars + 1))):
            pivot_lows.append({"bar": i, "price": float(lo)})

    return {"pivot_highs": pivot_highs, "pivot_lows": pivot_lows}


def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 20) -> dict | None:
    """
    Calculate Fibonacci retracement levels based on recent swing high/low.

    Uses find_pivot_points() to identify the most recent *significant* swing
    high/low (avoids picking up single-bar spikes as the anchor), falling
    back to window max/min if fewer than 2 pivots are found.

    Returns dict with swing_high, swing_low, fib_levels, current_fib_position
    (0-100, clamped), zone ("discount" <=50 / "premium" >50), optimal_entry,
    price_range. Returns None if there isn't enough data or the range is zero.
    """
    if df.empty or len(df) < lookback:
        return None

    recent_data = df.tail(lookback)

    pivots = find_pivot_points(recent_data, left_bars=2, right_bars=2)
    phs = pivots["pivot_highs"]
    pls = pivots["pivot_lows"]

    if phs:
        swing_high = float(max(phs, key=lambda p: p["bar"])["price"])
    else:
        swing_high = float(recent_data["High"].max())

    if pls:
        swing_low = float(max(pls, key=lambda p: p["bar"])["price"])
    else:
        swing_low = float(recent_data["Low"].min())

    current_price = float(df["Close"].iloc[-1])
    price_range = swing_high - swing_low

    if price_range <= 0:
        return None

    fib_ratios = {
        "0%": 1.000,
        "23.6%": 0.764,
        "38.2%": 0.618,
        "50%": 0.500,
        "61.8%": 0.382,
        "78.6%": 0.214,
        "100%": 0.000,
    }

    fib_levels = {label: swing_low + (price_range * ratio) for label, ratio in fib_ratios.items()}

    current_fib_position = ((current_price - swing_low) / price_range) * 100
    current_fib_position = max(0, min(100, current_fib_position))

    zone = "discount" if current_fib_position <= 50 else "premium"

    entry_levels = [fib_levels["38.2%"], fib_levels["50%"], fib_levels["61.8%"]]
    optimal_entry = min(entry_levels, key=lambda x: abs(x - current_price))

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "fib_levels": fib_levels,
        "current_fib_position": current_fib_position,
        "zone": zone,
        "optimal_entry": optimal_entry,
        "price_range": price_range,
    }


def get_fibonacci_zone_label(fib_position: float) -> str:
    if fib_position <= 23.6:
        return "Deep Discount (0-23.6%)"
    elif fib_position <= 38.2:
        return "Strong Discount (23.6-38.2%)"
    elif fib_position <= 50:
        return "Discount Zone (38.2-50%)"
    elif fib_position <= 61.8:
        return "Equilibrium (50-61.8%)"
    elif fib_position <= 78.6:
        return "Premium Zone (61.8-78.6%)"
    else:
        return "Extended Premium (78.6-100%)"


def analyze_volume(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Relative volume vs N-day average, trend, and accumulation/distribution signal."""
    if len(df) < lookback:
        return {}

    recent = df.tail(lookback)

    avg_volume = recent["Volume"].mean()
    current_volume = float(recent["Volume"].iloc[-1])
    rel_volume = current_volume / avg_volume if avg_volume > 0 else 1.0

    first_half_vol = recent["Volume"].iloc[:lookback // 2].mean()
    second_half_vol = recent["Volume"].iloc[lookback // 2:].mean()
    vol_trend = ("Increasing" if second_half_vol > first_half_vol * 1.1 else
                 "Decreasing" if second_half_vol < first_half_vol * 0.9 else "Stable")

    up_days = recent[recent["Close"] > recent["Close"].shift(1)]
    down_days = recent[recent["Close"] < recent["Close"].shift(1)]

    avg_vol_up = up_days["Volume"].mean() if len(up_days) > 0 else 0
    avg_vol_down = down_days["Volume"].mean() if len(down_days) > 0 else 0

    if avg_vol_up > avg_vol_down * 1.2:
        vol_signal = "Accumulation"
    elif avg_vol_down > avg_vol_up * 1.2:
        vol_signal = "Distribution"
    else:
        vol_signal = "Neutral"

    return {
        "current_volume": int(current_volume),
        "avg_volume": int(avg_volume),
        "relative_volume": round(rel_volume, 2),
        "volume_trend": vol_trend,
        "volume_signal": vol_signal,
        "volume_surge": rel_volume > 1.5,
    }
