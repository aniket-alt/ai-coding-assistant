"""
Project 06 -- Credit card fraud detection: data acquisition.

Pulls the real ULB credit-card fraud dataset (284,807 European card
transactions over two days in September 2013, of which 492 are fraudulent).
Features V1..V28 are the published PCA components; only Time and Amount are
in their original units.

    python ml/data.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from dsfs import Provenance, fetch, stamp_frame

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

FRAUD = Provenance(
    slug="ulb-creditcard-fraud",
    name="Credit Card Fraud Detection (ULB)",
    source_url="https://openml.org/data/v1/download/1673544/creditcard.arff",
    license="Public (Open Database License) — OpenML dataset 1597",
    citation=(
        "Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). "
        "Calibrating probability with undersampling for unbalanced classification. "
        "IEEE Symposium Series on Computational Intelligence. "
        "Université Libre de Bruxelles, MLG."
    ),
    notes=(
        "284,807 transactions by European cardholders over two days in "
        "September 2013; 492 fraudulent (0.172%). V1..V28 are PCA components "
        "published in place of the original features for confidentiality."
    ),
)

# The transaction amount and elapsed time are the only untransformed columns.
RAW_COLUMNS = ["Time", "Amount"]


THE_ACCURACY_TRAP = """
This dataset is 99.828% legitimate transactions.

A model that predicts "not fraud" for every single transaction therefore scores
99.828% accuracy -- higher than most of the models in the tournament below --
while catching zero fraud and being worth precisely nothing. Accuracy on a
problem this imbalanced is not a weak metric; it is an actively misleading one,
and it is the single most common way fraud results are overstated.

This project ranks on **average precision** (the area under the
precision-recall curve), which is sensitive to performance on the rare class
and ignores the vast pile of easy negatives. The all-legitimate baseline is
kept in the leaderboard specifically so its 99.8% accuracy and its ~0.002
average precision can sit on the same row.

The reported product metric is a review budget: an analyst team can only
examine so many flagged transactions a day, so what matters operationally is
how much fraud is caught inside that budget.
"""


def _parse_arff(raw: bytes) -> pd.DataFrame:
    """
    Minimal ARFF reader.

    scipy's loader returns byte-strings for nominal columns and is slower than
    reading the data section directly; the schema here is flat and numeric, so
    the attribute list is enough to name the columns.
    """
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()

    columns: list[str] = []
    data_start = 0
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("@attribute"):
            parts = line.split()
            columns.append(parts[1].strip("'\""))
        elif low.startswith("@data"):
            data_start = i + 1
            break

    body = "\n".join(lines[data_start:])
    df = pd.read_csv(io.StringIO(body), header=None, names=columns)
    return df


def load() -> tuple[pd.DataFrame, Provenance]:
    """Download (once) and return the transaction table."""
    path, prov = fetch(FRAUD, filename="creditcard.arff")
    df = _parse_arff(path.read_bytes())

    # The label arrives as a quoted nominal ('0'/'1') in the ARFF encoding.
    label = "Class" if "Class" in df.columns else df.columns[-1]
    df["Class"] = (
        df[label].astype(str).str.strip().str.strip("'\"").astype(int)
    )
    if label != "Class":
        df = df.drop(columns=[label])

    df = df.sort_values("Time").reset_index(drop=True)
    return df, stamp_frame(prov, df)


def summarise(df: pd.DataFrame) -> dict:
    """Facts the CRISP-DM data-understanding phase reports verbatim."""
    n = len(df)
    n_fraud = int(df["Class"].sum())
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]
    return {
        "n_transactions": n,
        "n_fraud": n_fraud,
        "fraud_rate": n_fraud / n,
        "imbalance_ratio": (n - n_fraud) / max(n_fraud, 1),
        "all_legit_accuracy": (n - n_fraud) / n,
        "hours_covered": float(df["Time"].max()) / 3600.0,
        "median_amount_legit": float(legit["Amount"].median()),
        "median_amount_fraud": float(fraud["Amount"].median()),
        "mean_amount_legit": float(legit["Amount"].mean()),
        "mean_amount_fraud": float(fraud["Amount"].mean()),
        "fraud_value_total": float(fraud["Amount"].sum()),
        "n_features": int(df.shape[1] - 1),
        "zero_amount_fraud": int((fraud["Amount"] == 0).sum()),
    }


def hourly_profile(df: pd.DataFrame) -> list[dict]:
    """Fraud rate by hour of the two-day window, for the EDA chart."""
    hours = (df["Time"] // 3600).astype(int)
    out = []
    for h, grp in df.groupby(hours):
        n = len(grp)
        f = int(grp["Class"].sum())
        out.append({
            "hour": int(h),
            "transactions": n,
            "fraud": f,
            "fraud_rate_pct": round(f / n * 100, 4) if n else 0.0,
        })
    return out


def amount_bands(df: pd.DataFrame) -> list[dict]:
    """Fraud rate by transaction size."""
    edges = [0, 1, 10, 50, 100, 500, 1000, np.inf]
    labels = ["€0", "€1–10", "€10–50", "€50–100", "€100–500", "€500–1k", "€1k+"]
    band = pd.cut(df["Amount"], bins=edges, labels=labels,
                  include_lowest=True, right=False)
    out = []
    for name, grp in df.groupby(band, observed=True):
        n = len(grp)
        f = int(grp["Class"].sum())
        out.append({
            "band": str(name),
            "transactions": n,
            "fraud": f,
            "fraud_rate_pct": round(f / n * 100, 4) if n else 0.0,
        })
    return out


if __name__ == "__main__":
    df, prov = load()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    prov.save(ARTIFACTS / "provenance.json")
    s = summarise(df)

    print(f"\ntransactions      : {s['n_transactions']:,}")
    print(f"fraudulent        : {s['n_fraud']:,}  ({s['fraud_rate']*100:.3f}%)")
    print(f"imbalance ratio   : 1 : {s['imbalance_ratio']:,.0f}")
    print(f"window            : {s['hours_covered']:.1f} hours")
    print(f"features          : {s['n_features']}")
    print()
    print(f"median amount     : legit €{s['median_amount_legit']:.2f}  "
          f"fraud €{s['median_amount_fraud']:.2f}")
    print(f"total fraud value : €{s['fraud_value_total']:,.2f}")
    print()
    print(f"*** predicting 'never fraud' scores "
          f"{s['all_legit_accuracy']*100:.3f}% accuracy ***")
    print(f"provenance -> {ARTIFACTS / 'provenance.json'}")
