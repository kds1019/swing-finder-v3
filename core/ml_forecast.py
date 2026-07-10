"""
ML ensemble forecast (Random Forest + Gradient Boosting) — ported from
swing-finder-v2's utils/ml_models.py, with the VIX-merge bug fixed.

Original bug: the reference code derived a date range from `recent.index` to
fetch VIX history, but by the time that code runs, the index is a plain
RangeIndex (dates live in a `Date` column upstream, reset to 0..N when the
index was built) — so the derived range was always ~1970-01-01 and the VIX
fetch silently returned nothing every time.

Fix here: VIX is never fetched internally. Callers (pipeline.py, via the
Research Agent) pass in a pre-fetched `vix_df` (columns: Date, vix), and this
module builds its DatetimeIndex explicitly from the bars' own `Date` column
before merging — eliminating the index/column mismatch structurally rather
than patching around it.

The large ASCII-art diagnostic print block from the original
`ensemble_ml_forecast` was dropped — it had no effect on return values.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

_MIN_BARS = 60


def prepare_features(
    df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
    spy_df: Optional[pd.DataFrame] = None,
    lookback: int = 1500,
    days_ahead: int = 5,
) -> tuple:
    """
    Prepare features for ML models using up to `lookback` bars of history
    (default 1500, ~6 years; uses all available rows if df is shorter).

    df must have a `Date` column (not just a positional index) plus
    Close/High/Low/Volume, and ideally RSI14/EMA20/EMA50/MACD from
    core.indicators.compute_indicators.

    vix_df, if provided, must have `Date` and `vix` columns — merged by date
    with forward-fill; dropped entirely if it covers less than half the rows
    (models train fine without it; better to omit than to inject mostly-NaN).

    spy_df, if provided, must have `Date` and `Close` columns (an OHLCV frame
    works fine) — used for relative-strength-vs-SPY features. Same
    core.relative_strength.py / core.multi_timeframe.py / core.volume_profile.py
    signals the live pipeline already computes for SmartScore/pattern purposes
    elsewhere, re-derived here in vectorized form (rolling/ewm over every row)
    rather than by calling those row-at-a-time functions in a loop, which would
    be prohibitively slow across a 1500-row training set.

    Returns (X, y, feature_names, y_stats, dates) — dates is the as-of date for
    each row in X/y, aligned 1:1 (needed to pool multiple tickers' feature rows
    into one cross-sectional dataset without misaligning them) — or a 5-tuple
    of Nones if there isn't enough data.
    """
    if len(df) < _MIN_BARS:
        return None, None, None, None, None

    recent = df.tail(lookback).copy()

    # Build the DatetimeIndex explicitly from the Date column — this is the fix:
    # never trust a pre-existing index to already be dates.
    date_idx = pd.to_datetime(recent["Date"]).dt.normalize()
    recent.index = date_idx.dt.tz_localize(None) if date_idx.dt.tz is not None else date_idx

    features = pd.DataFrame(index=recent.index)

    features["close"] = recent["Close"]
    features["high"] = recent["High"]
    features["low"] = recent["Low"]
    features["volume"] = recent["Volume"]

    if "RSI14" in recent.columns:
        features["rsi"] = recent["RSI14"]
    if "MACD" in recent.columns:
        features["macd"] = recent["MACD"]
    if "EMA20" in recent.columns:
        features["ema20"] = recent["EMA20"]
    if "EMA50" in recent.columns:
        features["ema50"] = recent["EMA50"]

    features["price_change"] = recent["Close"].pct_change()
    features["volume_change"] = recent["Volume"].pct_change()
    features["high_low_range"] = (recent["High"] - recent["Low"]) / recent["Close"]

    features["ma5"] = recent["Close"].rolling(5).mean()
    features["ma10"] = recent["Close"].rolling(10).mean()
    features["ma20"] = recent["Close"].rolling(20).mean()

    features["volatility"] = recent["Close"].rolling(10).std()

    for lag in [1, 2, 3, 5]:
        features[f"close_lag_{lag}"] = recent["Close"].shift(lag)
    features["volume_lag_1"] = recent["Volume"].shift(1)

    features["mom_5"] = recent["Close"].pct_change(5)
    features["mom_10"] = recent["Close"].pct_change(10)
    features["mom_20"] = recent["Close"].pct_change(20)

    if "EMA20" in recent.columns:
        features["ema20_slope"] = recent["EMA20"].pct_change(3)
    if "EMA50" in recent.columns:
        features["ema50_slope"] = recent["EMA50"].pct_change(5)

    _high_20 = recent["High"].rolling(20).max()
    _low_20 = recent["Low"].rolling(20).min()
    _range = (_high_20 - _low_20).replace(0, np.nan)
    features["range_position"] = (recent["Close"] - _low_20) / _range

    if "RSI14" in recent.columns:
        features["rsi_slope"] = recent["RSI14"].diff(3)

    features["price_vol_confirm"] = recent["Close"].pct_change(1) * recent["Volume"].pct_change(1)

    # --- Weekly trend alignment (core.multi_timeframe's daily/weekly EMA20-vs-EMA50 check,
    # vectorized): resample to weekly closes, compare weekly EMA20/EMA50 per completed week,
    # then reindex onto the daily index with ffill. ffill only ever pulls a week's label date
    # forward onto later dates, never backward, so this can't leak an in-progress week's data
    # into an earlier date — each day only ever sees the most recently *completed* week. ---
    weekly_close = recent["Close"].resample("W").last().dropna()
    if len(weekly_close) >= 10:
        weekly_ema20 = weekly_close.ewm(span=20, adjust=False).mean()
        weekly_ema50 = weekly_close.ewm(span=50, adjust=False).mean()
        weekly_uptrend = (weekly_ema20 > weekly_ema50).astype(float)
        features["weekly_uptrend"] = weekly_uptrend.reindex(features.index, method="ffill")

    # --- Volume-profile-position proxy (core.volume_profile's point-of-control check,
    # vectorized): rolling dollar-volume-weighted average price stands in for the histogram
    # POC — same "is price stretched above where volume actually concentrated" idea, without
    # needing a per-row histogram rebuild across the whole training set. ---
    _dollar_vol = recent["Close"] * recent["Volume"]
    _rolling_vwap = (
        _dollar_vol.rolling(60, min_periods=20).sum() / recent["Volume"].rolling(60, min_periods=20).sum()
    )
    features["vwap_position_60"] = recent["Close"] / _rolling_vwap - 1

    # --- Relative strength vs SPY (core.relative_strength, vectorized): percentage-point
    # difference between the ticker's and SPY's return over the same trailing window. ---
    if spy_df is not None and not spy_df.empty:
        spy_series = spy_df.set_index(pd.to_datetime(spy_df["Date"]).dt.normalize())["Close"]
        spy_aligned = spy_series.reindex(features.index, method="ffill")
        if spy_aligned.notna().sum() >= len(features) * 0.5:
            features["rs_20"] = recent["Close"].pct_change(19) * 100 - spy_aligned.pct_change(19) * 100
            features["rs_60"] = recent["Close"].pct_change(59) * 100 - spy_aligned.pct_change(59) * 100
        # else: skip RS features silently — models train fine without them, same as VIX below

    # --- Merge VIX (already correctly-dated — no re-derivation from a maybe-broken index) ---
    if vix_df is not None and not vix_df.empty:
        vix_series = vix_df.set_index(pd.to_datetime(vix_df["Date"]).dt.normalize())["vix"]
        vix_aligned = vix_series.reindex(features.index, method="ffill")
        if vix_aligned.notna().sum() >= len(features) * 0.5:
            features["vix"] = vix_aligned.fillna(vix_aligned.median())
        # else: skip VIX silently — models train fine without it

    features = features.dropna()
    dates_aligned = features.index
    features = features.reset_index(drop=True)

    if len(features) < max(20, days_ahead + 1):
        return None, None, None, None, None

    fwd_close = features["close"].shift(-days_ahead)
    fwd_return = fwd_close / features["close"] - 1
    y = fwd_return.iloc[:-days_ahead].values

    _cap = 0.04 * days_ahead
    y_stats = {
        "n_samples": int(len(y)),
        "n_features": int(features.shape[1] - 1),
        "mean_pct": round(float(np.mean(y)) * 100, 3),
        "std_pct": round(float(np.std(y)) * 100, 3),
        "min_pct": round(float(np.min(y)) * 100, 3),
        "p25_pct": round(float(np.percentile(y, 25)) * 100, 3),
        "median_pct": round(float(np.median(y)) * 100, 3),
        "p75_pct": round(float(np.percentile(y, 75)) * 100, 3),
        "max_pct": round(float(np.max(y)) * 100, 3),
        "pct_positive": round(float(np.mean(y > 0)) * 100, 1),
        "pct_capped": round(float(np.mean(np.abs(y) >= _cap)) * 100, 1),
        "days_ahead": days_ahead,
    }

    c = features["close"]

    for col in ["high", "low", "ma5", "ma10", "ma20"]:
        if col in features.columns:
            features[col] = features[col] / c - 1
    for col in ["ema20", "ema50"]:
        if col in features.columns:
            features[col] = features[col] / c - 1
    if "macd" in features.columns:
        features["macd"] = features["macd"] / c
    if "volatility" in features.columns:
        features["volatility"] = features["volatility"] / c
    for lag in [1, 2, 3, 5]:
        col = f"close_lag_{lag}"
        if col in features.columns:
            features[col] = features[col] / c - 1

    vol_ref = features["volume"].rolling(20, min_periods=5).mean().bfill()
    features["volume"] = features["volume"] / vol_ref
    if "volume_lag_1" in features.columns:
        features["volume_lag_1"] = features["volume_lag_1"] / vol_ref

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    features = features.drop(columns=["close"])

    feature_names = features.columns.tolist()
    X = features.iloc[:-days_ahead].values
    dates = dates_aligned[:-days_ahead]
    return X, y, feature_names, y_stats, dates


def random_forest_forecast(
    df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
    spy_df: Optional[pd.DataFrame] = None,
    days_ahead: int = 5,
) -> Dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestRegressor

        X, y, feature_names, y_stats, _dates = prepare_features(
            df, vix_df=vix_df, spy_df=spy_df, lookback=1500, days_ahead=days_ahead
        )

        if X is None or len(X) < 20:
            return {"success": False, "error": "Insufficient data"}

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=20,
            max_features=0.5,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)

        train_score = rf_model.score(X_train, y_train)
        rf_raw_r2 = rf_model.score(X_test, y_test)
        r2_clamped = float(np.clip(rf_raw_r2, 0.0, 1.0))
        rf_preds_test = rf_model.predict(X_test)

        pct_correct_dir = float(np.mean(np.sign(rf_preds_test) == np.sign(y_test)))
        dir_conf = max(pct_correct_dir - 0.5, 0.0) * 50.0
        r2_adj = max(1.0 + rf_raw_r2 * 5.0, 0.3)
        confidence = round(dir_conf * r2_adj, 1)

        last_features = X[-1].reshape(1, -1)
        predicted_return = float(rf_model.predict(last_features)[0])

        max_move = 0.04 * days_ahead
        predicted_return = float(np.clip(predicted_return, -max_move, max_move))

        current_price = float(df["Close"].iloc[-1])
        forecast_price = current_price * (1 + predicted_return)

        importances = rf_model.feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]

        return {
            "success": True,
            "forecast_price": round(forecast_price, 2),
            "predicted_return": round(predicted_return * 100, 2),
            "r2_score": r2_clamped,
            "confidence": confidence,
            "train_score": round(float(np.clip(train_score, 0.0, 1.0)) * 100, 1),
            "top_features": top_features,
            "model_type": "Random Forest",
            "y_stats": y_stats,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def gradient_boosting_forecast(
    df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
    spy_df: Optional[pd.DataFrame] = None,
    days_ahead: int = 5,
) -> Dict[str, Any]:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        X, y, feature_names, y_stats, _dates = prepare_features(
            df, vix_df=vix_df, spy_df=spy_df, lookback=1500, days_ahead=days_ahead
        )

        if X is None or len(X) < 20:
            return {"success": False, "error": "Insufficient data"}

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=20,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        gb_model.fit(X_train, y_train)

        train_score = gb_model.score(X_train, y_train)
        gb_raw_r2 = gb_model.score(X_test, y_test)
        r2_clamped = float(np.clip(gb_raw_r2, 0.0, 1.0))
        gb_preds_test = gb_model.predict(X_test)

        pct_correct_dir = float(np.mean(np.sign(gb_preds_test) == np.sign(y_test)))
        dir_conf = max(pct_correct_dir - 0.5, 0.0) * 50.0
        r2_adj = max(1.0 + gb_raw_r2 * 5.0, 0.3)
        confidence = round(dir_conf * r2_adj, 1)

        last_features = scaler.transform(X[-1].reshape(1, -1))
        predicted_return = float(gb_model.predict(last_features)[0])

        max_move = 0.04 * days_ahead
        predicted_return = float(np.clip(predicted_return, -max_move, max_move))

        current_price = float(df["Close"].iloc[-1])
        forecast_price = current_price * (1 + predicted_return)

        importances = gb_model.feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]

        return {
            "success": True,
            "forecast_price": round(forecast_price, 2),
            "predicted_return": round(predicted_return * 100, 2),
            "r2_score": r2_clamped,
            "confidence": confidence,
            "train_score": round(float(np.clip(train_score, 0.0, 1.0)) * 100, 1),
            "top_features": top_features,
            "model_type": "Gradient Boosting",
            "y_stats": y_stats,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def ensemble_ml_forecast(
    df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
    spy_df: Optional[pd.DataFrame] = None,
    days_ahead: int = 5,
) -> Dict[str, Any]:
    """Combine Random Forest and Gradient Boosting via an R²-weighted blend."""
    rf_result = random_forest_forecast(df, vix_df=vix_df, spy_df=spy_df, days_ahead=days_ahead)
    gb_result = gradient_boosting_forecast(df, vix_df=vix_df, spy_df=spy_df, days_ahead=days_ahead)

    if not rf_result["success"] or not gb_result["success"]:
        return {
            "success": False,
            "error": "One or more models failed",
            "rf_error": rf_result.get("error"),
            "gb_error": gb_result.get("error"),
        }

    rf_r2 = rf_result["r2_score"]
    gb_r2 = gb_result["r2_score"]
    total_r2 = rf_r2 + gb_r2

    if total_r2 < 1e-9:
        rf_weight = gb_weight = 0.5
    else:
        rf_weight = rf_r2 / total_r2
        gb_weight = gb_r2 / total_r2

    ensemble_price = (rf_result["forecast_price"] * rf_weight + gb_result["forecast_price"] * gb_weight)
    ensemble_confidence = (rf_result["confidence"] + gb_result["confidence"]) / 2

    forecast_low = min(rf_result["forecast_price"], gb_result["forecast_price"])
    forecast_high = max(rf_result["forecast_price"], gb_result["forecast_price"])

    return {
        "success": True,
        "ensemble_price": round(ensemble_price, 2),
        "forecast_low": round(forecast_low, 2),
        "forecast_high": round(forecast_high, 2),
        "confidence": round(ensemble_confidence, 1),
        "rf_prediction": rf_result["forecast_price"],
        "gb_prediction": gb_result["forecast_price"],
        "rf_confidence": rf_result["confidence"],
        "gb_confidence": gb_result["confidence"],
        "rf_r2": rf_r2,
        "gb_r2": gb_r2,
        "agreement": round(abs(rf_result["forecast_price"] - gb_result["forecast_price"]) / ensemble_price * 100, 1),
        "y_stats": rf_result.get("y_stats"),
    }


ML_EDGE_POSITIVE_BONUS = 10
ML_EDGE_NEGATIVE_PENALTY = -15

# Ceiling values above apply at full strength once `confidence` reaches this level;
# below it, the adjustment scales down proportionally. Provisional — no calibration
# data yet establishes what confidence level actually corresponds to reliable edge
# (see docs/ml-edge-confidence-research.md); revisit once a walk-forward backtest
# quantifies the real relationship between confidence and forward accuracy.
ML_EDGE_CONFIDENCE_SATURATION = 10.0
# A near-zero-confidence call still means the ensemble picked a direction, however
# weakly — floor keeps that from being scaled down to a no-op.
ML_EDGE_MIN_SCALE = 0.2


def evaluate_ml_edge_score(ml_result: dict, current_price: float) -> dict:
    """SmartScore adjustment from the ML ensemble's directional call: rewards
    a positive 5-day forecast (model confirms upside), penalizes a negative
    one — catches cases like a clean technical breakout the model itself
    doesn't confirm (e.g. a 100 SmartScore where the ensemble actually
    forecasts a pullback). Only runs on the post-sector-cap shortlist, since
    the ensemble trains a fresh model per ticker and isn't cheap enough for
    the full universe scan.

    The bonus/penalty scales with the ensemble's own `confidence` (see
    random_forest_forecast/gradient_boosting_forecast) rather than applying at
    full strength regardless of it — a 0.1-confidence call and a 9-confidence
    call were previously treated identically."""
    if not ml_result.get("success") or current_price <= 0:
        return {
            "triggered": False, "score_adjustment": 0,
            "flag": "ml_edge_unavailable" if not ml_result.get("success") else None,
            "edge_pct": None,
        }

    edge_pct = round((ml_result["ensemble_price"] - current_price) / current_price * 100, 2)
    confidence = ml_result.get("confidence") or 0.0
    scale = ML_EDGE_MIN_SCALE + (1.0 - ML_EDGE_MIN_SCALE) * min(confidence / ML_EDGE_CONFIDENCE_SATURATION, 1.0)

    if edge_pct > 0:
        adjustment = round(ML_EDGE_POSITIVE_BONUS * scale)
        return {"triggered": True, "score_adjustment": adjustment, "flag": None, "edge_pct": edge_pct}
    adjustment = round(ML_EDGE_NEGATIVE_PENALTY * scale)
    return {"triggered": True, "score_adjustment": adjustment, "flag": "negative_ml_edge", "edge_pct": edge_pct}
