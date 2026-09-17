"""
Project 07 -- bank telemarketing calls: acquisition and dating.

41,188 phone calls made by a Portuguese bank between May 2008 and November
2010, each ending in whether the client subscribed to a term deposit. The file
is ordered by date but stores only the month, so the year is reconstructed from
the order: each time the month goes backwards, a new year has begun.

Two properties shape the whole project. `duration` is the length of the call,
which is only known once the call has ended and the answer given; the dataset's
own documentation says to discard it for a realistic model. And the calls span
the financial crisis, so the share of calls that succeed changes enormously
over time.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from dsfs import Provenance, fetch, stamp_frame

BANK = Provenance(
    slug="uci-bank-marketing",
    name="Bank Marketing, with social and economic context (UCI)",
    source_url="https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    license="Creative Commons Attribution 4.0 (CC BY 4.0)",
    citation=(
        "Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the "
        "success of bank telemarketing. Decision Support Systems, 62, 22-31. "
        "https://doi.org/10.1016/j.dss.2014.03.001"
    ),
    notes="bank-additional-full.csv: 41,188 calls ordered by date, May 2008 to November 2010.",
)

TARGET = "subscribed"
LEAK = "duration"
NUMERIC = ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
           "cons.conf.idx", "euribor3m", "nr.employed"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact",
               "month", "day_of_week", "poutcome"]
FEATURES = NUMERIC + CATEGORICAL
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
TEST_YEAR = 2010


def _read(path: Path) -> pd.DataFrame:
    # The download is a zip holding two more zips; the full file is in one of them.
    with zipfile.ZipFile(path) as outer:
        inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
        with inner.open("bank-additional/bank-additional-full.csv") as f:
            return pd.read_csv(f, sep=";")


def load() -> tuple[pd.DataFrame, Provenance]:
    path, prov = fetch(BANK, filename="bank_marketing.zip")
    df = _read(path)
    prov = stamp_frame(prov, df)
    month = df["month"].map({m: i for i, m in enumerate(MONTHS, 1)}).to_numpy()
    if np.isnan(month.astype(float)).any():
        raise ValueError("unrecognised month label")
    df["year"] = 2008 + np.concatenate([[0], np.cumsum(np.diff(month) < 0)])
    df["period"] = [f"{y}-{m:02d}" for y, m in zip(df["year"], month)]
    df["order"] = np.arange(len(df))
    df[TARGET] = (df["y"] == "yes").astype(int)
    df = df.drop(columns=["y"])
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)
    return df, prov


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Everything before 2010 trains; every call in 2010 is the test set."""
    return df[df["year"] < TEST_YEAR].copy(), df[df["year"] >= TEST_YEAR].copy()


if __name__ == "__main__":
    df, prov = load()
    train, test = split(df)
    print(df.shape, prov.sha256[:12])
    print(f"train {len(train):,} rate {train[TARGET].mean():.4f} | test {len(test):,} rate {test[TARGET].mean():.4f}")
    print(df.groupby("period", sort=False)[TARGET].agg(["size", "mean"]).round(3).to_string())
