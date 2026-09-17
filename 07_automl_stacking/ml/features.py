"""
Project 07 -- preprocessing shared by every hand-built contestant.

`pdays` stores 999 for a client never contacted before. Left as a number, 999
sits at the far end of a scale whose real values run from 0 to 27, so linear
and nearest-neighbour models read "never contacted" as "contacted very long
ago". It becomes an explicit indicator, and the sentinel is replaced by -1.

Everything is fitted inside each model's pipeline on training rows only.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from data import CATEGORICAL, NUMERIC

NEVER_CONTACTED = 999


def add_indicators(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["previously_contacted"] = (X["pdays"] != NEVER_CONTACTED).astype(int)
    X["pdays"] = X["pdays"].where(X["pdays"] != NEVER_CONTACTED, -1)
    return X


def indicator_names(transformer, names):
    return [*names, "previously_contacted"]


def preprocessor(scale: bool, extra_numeric: tuple[str, ...] = ()) -> Pipeline:
    numeric = NUMERIC + ["previously_contacted"] + list(extra_numeric)
    return Pipeline([
        ("indicators", FunctionTransformer(add_indicators, feature_names_out=indicator_names)),
        ("columns", ColumnTransformer([
            ("num", StandardScaler() if scale else "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), CATEGORICAL),
        ], verbose_feature_names_out=False)),
    ])
