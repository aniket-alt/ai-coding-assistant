"""
Project 10 -- the Adult census data.

48,842 records from the 1994 US Census, extracted by Barry Becker, with the
classic question of whether a person's income exceeds $50,000 a year. The
dataset is also a standard case study in fairness, because it records sex and
race alongside variables that stand in for them.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from dsfs import Provenance, fetch, stamp_frame

ADULT = Provenance(
    slug="openml-adult",
    name="Adult Census Income (UCI, via OpenML 1590)",
    source_url="https://openml.org/data/v1/download/1595261/adult.arff",
    license="Public (OpenML); originally UCI Machine Learning Repository, CC BY 4.0",
    citation=(
        "Becker, B., & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning "
        "Repository. https://doi.org/10.24432/C5XW20"
    ),
    notes="Extracted from the 1994 US Census; income above or below $50,000.",
)

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation", "relationship",
               "race", "sex", "native-country"]
SENSITIVE = ["sex", "race"]


def _parse_arff(raw: bytes) -> pd.DataFrame:
    text = raw.decode("utf-8", errors="replace").splitlines()
    columns, start = [], 0
    for i, line in enumerate(text):
        low = line.strip().lower()
        if low.startswith("@attribute"):
            columns.append(line.split()[1].strip("'\""))
        elif low.startswith("@data"):
            start = i + 1
            break
    body = "\n".join(text[start:])
    return pd.read_csv(io.StringIO(body), header=None, names=columns, quotechar="'",
                       skipinitialspace=True, na_values=["?"])


def load() -> tuple[pd.DataFrame, Provenance]:
    path, prov = fetch(ADULT, filename="adult.arff")
    df = _parse_arff(path.read_bytes())
    df = df.rename(columns={"class": "income"})
    df["high_income"] = (df["income"].astype(str).str.strip() == ">50K").astype(int)
    # Plain object columns with NaN for missing values: scikit-learn's imputers
    # cannot compare pandas' nullable string type, whose missing marker is pd.NA.
    for c in CATEGORICAL:
        df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else np.nan).astype(object)
    return df, stamp_frame(prov, df)


if __name__ == "__main__":
    df, prov = load()
    print(df.shape, "high income rate", round(df["high_income"].mean(), 4))
    print(df.isna().sum()[df.isna().sum() > 0])
    print(df.groupby("sex")["high_income"].mean().round(3))
