"""
Project 01 -- feature engineering.

Every feature here must be computable at the moment a rider requests a cab.
Two consequences shape the whole module:

1. Nothing derived from the meter (distance, fare) or the arrival time is used.
2. The statistics that replace them -- how far a zone pair usually is, how long
   it usually takes -- are fitted on the training window only, and the
   target-derived ones are encoded out-of-fold so a row never contributes to
   its own encoding.

That second point is the difference between a target encoding and a leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Global prior weight for the smoothed target encoder. Higher = more shrinkage
# toward the global mean for rare zone pairs.
SMOOTHING = 20.0


# --------------------------------------------------------------------------
# Calendar features -- knowable at booking, no fitting required
# --------------------------------------------------------------------------

def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-of-request features. Pure functions of the pickup timestamp."""
    ts = df["tpep_pickup_datetime"]
    out = pd.DataFrame(index=df.index)

    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["day"] = ts.dt.day
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Rush hour, defined on weekdays only.
    weekday = ts.dt.dayofweek < 5
    out["is_am_rush"] = (weekday & ts.dt.hour.between(7, 9)).astype(int)
    out["is_pm_rush"] = (weekday & ts.dt.hour.between(16, 19)).astype(int)
    out["is_overnight"] = (~ts.dt.hour.between(6, 22)).astype(int)

    # Cyclical encodings so 23:00 and 00:00 are neighbours rather than extremes.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)

    return out


# --------------------------------------------------------------------------
# Zone-pair geography -- fitted on train, applied to both
# --------------------------------------------------------------------------

def zone_pair_statistics(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Learn, from the training window alone, what each origin/destination pair
    looks like: how far it usually is and how often it is travelled.

    Distance is used rather than duration here -- it is a property of the
    geography, not of the target, so it carries no label information.
    """
    pair = (
        train.groupby(["PULocationID", "DOLocationID"], observed=True)
        .agg(
            zp_median_distance=("trip_distance", "median"),
            zp_trip_count=("trip_distance", "size"),
        )
        .reset_index()
    )

    pu = (
        train.groupby("PULocationID", observed=True)
        .agg(
            pu_median_distance=("trip_distance", "median"),
            pu_trip_count=("trip_distance", "size"),
        )
        .reset_index()
    )

    do = (
        train.groupby("DOLocationID", observed=True)
        .agg(do_trip_count=("trip_distance", "size"))
        .reset_index()
    )

    return {
        "pair": pair,
        "pu": pu,
        "do": do,
        "global_median_distance": float(train["trip_distance"].median()),
    }


def apply_zone_statistics(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Attach the learned zone statistics, falling back for unseen pairs."""
    out = df[["PULocationID", "DOLocationID"]].merge(
        stats["pair"], on=["PULocationID", "DOLocationID"], how="left"
    )
    out = out.merge(stats["pu"], on="PULocationID", how="left")
    out = out.merge(stats["do"], on="DOLocationID", how="left")
    out.index = df.index

    # An unseen zone pair at inference time falls back to the origin zone's
    # typical distance, then to the global median.
    out["zp_median_distance"] = (
        out["zp_median_distance"]
        .fillna(out["pu_median_distance"])
        .fillna(stats["global_median_distance"])
    )
    out["pu_median_distance"] = out["pu_median_distance"].fillna(
        stats["global_median_distance"]
    )
    for c in ("zp_trip_count", "pu_trip_count", "do_trip_count"):
        out[c] = out[c].fillna(0)

    # Rarity is itself informative: unusual routes behave differently.
    out["zp_is_rare"] = (out["zp_trip_count"] < 10).astype(int)
    out["is_same_zone"] = (out["PULocationID"] == out["DOLocationID"]).astype(int)

    return out.drop(columns=["PULocationID", "DOLocationID"])


# --------------------------------------------------------------------------
# Out-of-fold target encoding
# --------------------------------------------------------------------------

class OutOfFoldTargetEncoder:
    """
    Smoothed mean-target encoding for high-cardinality zone keys.

    Fitted on training rows only. Within training, each row's encoding comes
    from folds that exclude it, so no row informs its own feature -- the usual
    way target encoding leaks. Test rows use the full-training statistics.

        encoding = (sum_group + prior * global_mean) / (count_group + prior)
    """

    def __init__(self, keys: list[str], name: str, smoothing: float = SMOOTHING,
                 n_splits: int = 5, seed: int = 42) -> None:
        self.keys = keys
        self.name = name
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.seed = seed
        self.global_mean_: float = 0.0
        self.mapping_: pd.DataFrame | None = None

    def fit_transform(self, train: pd.DataFrame, y: pd.Series) -> pd.Series:
        self.global_mean_ = float(y.mean())
        oof = pd.Series(np.nan, index=train.index, name=self.name)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        frame = train[self.keys].copy()
        frame["_y"] = y.to_numpy()

        for fit_idx, enc_idx in kf.split(frame):
            part = frame.iloc[fit_idx]
            agg = part.groupby(self.keys, observed=True)["_y"].agg(["sum", "count"])
            prior = float(part["_y"].mean())
            smooth = (agg["sum"] + self.smoothing * prior) / (agg["count"] + self.smoothing)

            target_rows = frame.iloc[enc_idx]
            keyed = pd.MultiIndex.from_frame(target_rows[self.keys]) \
                if len(self.keys) > 1 else pd.Index(target_rows[self.keys[0]])
            oof.iloc[enc_idx] = smooth.reindex(keyed).to_numpy()

        oof = oof.fillna(self.global_mean_)

        # Full-training mapping, for transforming test rows and serving.
        agg = frame.groupby(self.keys, observed=True)["_y"].agg(["sum", "count"])
        self.mapping_ = (
            (agg["sum"] + self.smoothing * self.global_mean_)
            / (agg["count"] + self.smoothing)
        ).rename(self.name).reset_index()

        return oof

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.mapping_ is None:
            raise RuntimeError("encoder not fitted")
        merged = df[self.keys].merge(self.mapping_, on=self.keys, how="left")
        merged.index = df.index
        return merged[self.name].fillna(self.global_mean_)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

BASE_COLUMNS = ["PULocationID", "DOLocationID", "passenger_count",
                "RatecodeID", "vendor_fields_missing"]

FEATURE_ORDER = [
    # calendar
    "hour", "dayofweek", "day", "is_weekend",
    "is_am_rush", "is_pm_rush", "is_overnight",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # request
    "PULocationID", "DOLocationID", "passenger_count", "RatecodeID",
    "vendor_fields_missing",
    # learned geography
    "zp_median_distance", "pu_median_distance",
    "zp_trip_count", "pu_trip_count", "do_trip_count",
    "zp_is_rare", "is_same_zone",
    # out-of-fold target encodings
    "te_zone_pair", "te_pu_hour",
]

CATEGORICAL = ["PULocationID", "DOLocationID", "RatecodeID"]


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Build the aligned train/test design matrices.

    Everything that learns from data is fitted on `train` here and merely
    applied to `test`, which is what makes the split honest.
    """
    stats = zone_pair_statistics(train)

    def assemble(df: pd.DataFrame) -> pd.DataFrame:
        parts = [
            calendar_features(df),
            df[BASE_COLUMNS].copy(),
            apply_zone_statistics(df, stats),
        ]
        return pd.concat(parts, axis=1)

    Xtr = assemble(train)
    Xte = assemble(test)

    # Target encodings: out-of-fold within train, full-train mapping for test.
    enc_pair = OutOfFoldTargetEncoder(["PULocationID", "DOLocationID"], "te_zone_pair")
    Xtr["te_zone_pair"] = enc_pair.fit_transform(train, y_train)
    Xte["te_zone_pair"] = enc_pair.transform(test)

    pu_hour_tr = train.assign(_hour=train["tpep_pickup_datetime"].dt.hour)
    pu_hour_te = test.assign(_hour=test["tpep_pickup_datetime"].dt.hour)
    enc_pu_hour = OutOfFoldTargetEncoder(["PULocationID", "_hour"], "te_pu_hour")
    Xtr["te_pu_hour"] = enc_pu_hour.fit_transform(pu_hour_tr, y_train)
    Xte["te_pu_hour"] = enc_pu_hour.transform(pu_hour_te)

    Xtr = Xtr[FEATURE_ORDER]
    Xte = Xte[FEATURE_ORDER]

    # Zone and rate-code identifiers arrive as floats (RatecodeID was imputed).
    # XGBoost rejects float-typed category levels, so the codes are normalised
    # to integers before the categorical dtype is applied.
    for c in CATEGORICAL:
        Xtr[c] = Xtr[c].astype("int32").astype("category")
        Xte[c] = pd.Categorical(Xte[c].astype("int32"), categories=Xtr[c].cat.categories)

    encoders = {
        "zone_stats": stats,
        "te_zone_pair": enc_pair,
        "te_pu_hour": enc_pu_hour,
    }
    return Xtr, Xte, encoders
