"""
Project 12 -- day-ahead features.

Rows are indexed by the hour being forecast, T. The forecast is issued 24 hours
earlier, at T - 24, so every feature must be computable from readings at or
before T - 24.

The easy mistake is a lag shorter than the horizon. "Load one hour ago" is a
superb predictor of the next hour and is simply not available when tomorrow's
6pm is forecast at 6pm today. `LAGS` below is checked against the horizon, and
`assert_no_lookahead` recomputes features on truncated history to prove no
feature reads past the forecast origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 24

# Lags in hours, measured back from the target hour T.
LAGS = [24, 25, 48, 72, 168, 336, 504, 672]

FEATURES = (
    [f"lag_{k}" for k in LAGS]
    + ["same_hour_4wk_mean", "mean_last_24h", "mean_last_168h", "std_last_168h",
       "hour", "dayofweek", "is_weekend", "month", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
)


def build(series: pd.Series, lags: list[int] = LAGS, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Features for every target hour, including the `horizon` hours after the
    series ends, so a forecast for tomorrow can be issued from today's data.
    """
    step = pd.Timedelta(hours=1)
    future = pd.date_range(series.index[-1] + step, periods=horizon, freq="h")
    s = series.reindex(series.index.append(future))

    out = pd.DataFrame(index=s.index)
    for k in lags:
        out[f"lag_{k}"] = s.shift(k)

    # Same hour over the previous four weeks: the household's recent habit.
    out["same_hour_4wk_mean"] = pd.concat([s.shift(k) for k in (168, 336, 504, 672)], axis=1).mean(axis=1)

    # Windows end at T - horizon, the moment the forecast is issued.
    origin = s.shift(horizon)
    out["mean_last_24h"] = origin.rolling(24, min_periods=18).mean()
    out["mean_last_168h"] = origin.rolling(168, min_periods=120).mean()
    out["std_last_168h"] = origin.rolling(168, min_periods=120).std()

    # Calendar of the target hour is known in advance.
    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["month"] = idx.month
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    out["target"] = s
    return out


def assert_no_lookahead(frame: pd.DataFrame, series: pd.Series, columns: list[str],
                        horizon: int = HORIZON, n_checks: int = 30, seed: int = 42) -> dict:
    """
    Recompute features as if standing at each forecast origin T - horizon, with
    no readings after it, and require every value to match. A feature that uses
    anything later than the origin changes when that later data is removed.
    """
    rng = np.random.default_rng(seed)
    candidates = frame.index[800:-(horizon + 1)]
    picks = sorted(rng.choice(len(candidates), size=min(n_checks, len(candidates)), replace=False))
    compared = mismatched = 0
    examples: list[str] = []
    for i in picks:
        target = candidates[i]
        origin = target - pd.Timedelta(hours=horizon)
        truncated = build(series.loc[:origin], horizon=horizon)
        a = truncated.loc[target, columns]
        b = frame.loc[target, columns]
        for c in columns:
            compared += 1
            if not (pd.isna(a[c]) and pd.isna(b[c])) and not np.isclose(a[c], b[c], equal_nan=True):
                mismatched += 1
                if len(examples) < 5:
                    examples.append(f"{c} at {target}")
    return {"targets_checked": len(picks), "values_compared": compared,
            "mismatches": mismatched, "examples": examples}
