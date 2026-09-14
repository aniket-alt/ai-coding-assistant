"""
Project 15 -- feature engineering.

Every feature at row t is computed from information available at the close of
day t, and the target is the return of day t+1. That single rule is the whole
discipline of this module, and it is easy to break by accident: a rolling mean
centred instead of trailing, a `shift()` in the wrong direction, or a scaler
fitted on the full series will each silently import the future.

`assert_no_lookahead` at the bottom checks the rule empirically rather than
trusting the code to be right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Horizons to forecast, in trading days.
HORIZONS = (1, 5)


def build(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach backward-looking features and forward targets.

    Returns a frame indexed by date, where row t carries features known at the
    close of t and targets for t+1 and t+5.
    """
    out = pd.DataFrame(index=df.index)
    r = df["log_return"]

    # -- lagged returns: what just happened ------------------------------
    for lag in (1, 2, 3, 5, 10):
        out[f"ret_lag{lag}"] = r.shift(lag - 1) if lag == 1 else r.rolling(lag).sum()

    # -- realised volatility over trailing windows ------------------------
    for w in (5, 10, 21, 63):
        out[f"vol_{w}"] = r.rolling(w).std() * np.sqrt(252)

    # Volatility of volatility: regime instability.
    out["vol_of_vol"] = out["vol_21"].rolling(21).std()
    # Is short-term vol elevated relative to longer-term? Regime change signal.
    out["vol_ratio"] = out["vol_5"] / out["vol_63"]

    # -- Parkinson range estimator, which uses the day's high/low ---------
    hl = np.log(df["spy_high"] / df["spy_low"])
    out["parkinson_21"] = (
        (hl ** 2).rolling(21).mean() / (4 * np.log(2))
    ) ** 0.5 * np.sqrt(252)

    # -- momentum and trend -----------------------------------------------
    for w in (21, 63, 126):
        out[f"mom_{w}"] = r.rolling(w).sum()
    ma200 = df["spy"].rolling(200).mean()
    out["dist_ma200"] = df["spy"] / ma200 - 1.0
    ma50 = df["spy"].rolling(50).mean()
    out["ma50_over_ma200"] = ma50 / ma200 - 1.0

    # -- implied volatility and rates -------------------------------------
    out["vix"] = df["vix"]
    out["vix_chg"] = df["vix"].pct_change()
    out["vix_rel_21"] = df["vix"] / df["vix"].rolling(21).mean() - 1.0
    # The variance risk premium: implied vol above realised vol.
    out["vrp"] = df["vix"] / 100.0 - out["vol_21"]
    out["tnx"] = df["tnx"]
    out["tnx_chg_21"] = df["tnx"] - df["tnx"].shift(21)

    # -- volume -----------------------------------------------------------
    vol = df["spy_volume"].replace(0, np.nan)
    out["volume_rel_21"] = vol / vol.rolling(21).mean() - 1.0

    # -- calendar ----------------------------------------------------------
    out["dayofweek"] = df.index.dayofweek
    out["month"] = df.index.month

    # -- targets: strictly forward ----------------------------------------
    for h in HORIZONS:
        # Sum of the next h daily log returns, i.e. the return realised from
        # the close of t to the close of t+h.
        out[f"target_h{h}"] = (
            r.shift(-1).rolling(h).sum().shift(-(h - 1)) if h > 1 else r.shift(-1)
        )

    return out


FEATURE_COLUMNS = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "ret_lag10",
    "vol_5", "vol_10", "vol_21", "vol_63", "vol_of_vol", "vol_ratio",
    "parkinson_21",
    "mom_21", "mom_63", "mom_126", "dist_ma200", "ma50_over_ma200",
    "vix", "vix_chg", "vix_rel_21", "vrp", "tnx", "tnx_chg_21",
    "volume_rel_21", "dayofweek", "month",
]


def assert_no_lookahead(frame: pd.DataFrame, df: pd.DataFrame,
                        n_checks: int = 40, seed: int = 42) -> dict:
    """
    Verify empirically that no feature depends on the future.

    Recomputes the feature block on a truncated history -- as if standing at
    day t with nothing after it -- and checks that row t is identical to the
    value produced when the whole series was available. A feature that peeks
    ahead changes when the future is removed.
    """
    rng = np.random.default_rng(seed)
    usable = frame.index[250:-10]
    if len(usable) == 0:
        raise ValueError("not enough history to check for lookahead")

    picks = rng.choice(len(usable), size=min(n_checks, len(usable)), replace=False)
    mismatches: list[str] = []
    compared = 0

    for i in sorted(picks):
        date = usable[i]
        truncated = build(df.loc[:date])
        if date not in truncated.index:
            continue
        a = truncated.loc[date, FEATURE_COLUMNS]
        b = frame.loc[date, FEATURE_COLUMNS]
        for col in FEATURE_COLUMNS:
            va, vb = a[col], b[col]
            compared += 1
            if pd.isna(va) and pd.isna(vb):
                continue
            if not np.isclose(float(va), float(vb), rtol=1e-9, atol=1e-12,
                              equal_nan=True):
                mismatches.append(f"{col} on {date.date()}: {va} vs {vb}")

    return {
        "dates_checked": len(picks),
        "values_compared": compared,
        "mismatches": mismatches[:10],
        "n_mismatches": len(mismatches),
        "passed": not mismatches,
    }
