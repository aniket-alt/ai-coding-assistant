"""
Project 03 -- customer features and future outcomes.

Customers are described using only their first year of purchases (December
2009 to November 2010). Their behaviour in the following year is the outcome
each segmentation is judged on, and is never used to build a segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The boundary between the year segments are built from and the year they are
# judged on. The snapshot is taken at this instant.
SNAPSHOT = pd.Timestamp("2010-12-01")

FEATURES = [
    "recency_days", "frequency", "monetary", "avg_basket",
    "n_products", "tenure_days", "return_rate",
]

# Heavily right-skewed counts and amounts are modelled on a log scale, so one
# wholesale buyer does not define a cluster by themselves.
LOG_FEATURES = ["frequency", "monetary", "avg_basket", "n_products"]


def split_periods(sales: pd.DataFrame, cancellations: pd.DataFrame) -> dict:
    before = sales["InvoiceDate"] < SNAPSHOT
    return {
        "sales_p1": sales[before],
        "sales_p2": sales[~before],
        "cancel_p1": cancellations[cancellations["InvoiceDate"] < SNAPSHOT],
    }


def customer_features(sales_p1: pd.DataFrame, cancel_p1: pd.DataFrame) -> pd.DataFrame:
    """Recency, frequency, monetary value and four behavioural extras, per customer."""
    g = sales_p1.groupby("CustomerID")
    feats = pd.DataFrame({
        "recency_days": (SNAPSHOT - g["InvoiceDate"].max()).dt.days,
        "frequency": g["Invoice"].nunique(),
        "monetary": g["Revenue"].sum(),
        "n_products": g["StockCode"].nunique(),
        "tenure_days": (SNAPSHOT - g["InvoiceDate"].min()).dt.days,
        "lines": g.size(),
    })
    feats["avg_basket"] = feats["monetary"] / feats["frequency"]

    returned = cancel_p1.groupby("CustomerID").size()
    feats["return_rate"] = (
        returned.reindex(feats.index).fillna(0) / (feats["lines"] + returned.reindex(feats.index).fillna(0))
    )
    return feats[FEATURES]


def future_outcomes(sales_p2: pd.DataFrame, customers: pd.Index) -> pd.DataFrame:
    """What each first-year customer did in the second year. Zero if they never came back."""
    rev = sales_p2.groupby("CustomerID")["Revenue"].sum()
    out = pd.DataFrame(index=customers)
    out["future_revenue"] = rev.reindex(customers).fillna(0.0)
    out["returned"] = (out["future_revenue"] > 0).astype(int)
    return out


def model_matrix(feats: pd.DataFrame) -> pd.DataFrame:
    """Log-transform the skewed columns. Scaling happens inside each pipeline."""
    X = feats.copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c])
    return X
