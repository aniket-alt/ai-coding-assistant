"""
Project 05 -- four small, real datasets for the skills catalogue.

Every demonstration in this lab runs on one of these. They are deliberately
small so a skill can be re-run live from the browser, and real so the
differences it measures are differences that occur in practice.

The original version of this project generated its "Kaggle Titanic" with
`numpy.random` and labelled it a benchmark. These are the actual files,
downloaded from OpenML and checksummed like every other dataset in this
repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dsfs import Provenance, fetch, read_arff, stamp_frame

SOURCES = {
    "titanic": Provenance(
        slug="openml-titanic",
        name="Titanic (OpenML 40945)",
        source_url="https://openml.org/data/v1/download/16826755/Titanic.arff",
        license="Public (OpenML)",
        citation="Titanic [Dataset]. OpenML, dataset 40945. https://www.openml.org/d/40945",
        notes="1,309 passengers; survival, class, sex, age, fare, cabin and port of embarkation.",
    ),
    "credit_g": Provenance(
        slug="openml-credit-g",
        name="German Credit (OpenML 31)",
        source_url="https://openml.org/data/v1/download/31/credit-g.arff",
        license="Public (OpenML); originally UCI Statlog (German Credit Data), CC BY 4.0",
        citation=("Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning "
                  "Repository. https://doi.org/10.24432/C5NC77"),
        notes="1,000 loan applicants labelled good or bad credit risk.",
    ),
    "diabetes": Provenance(
        slug="openml-diabetes",
        name="Pima Indians Diabetes (OpenML 37)",
        source_url="https://openml.org/data/v1/download/37/diabetes.arff",
        license="Public (OpenML)",
        citation=("Smith, J. W. et al. (1988). Using the ADAP learning algorithm to forecast the onset "
                  "of diabetes mellitus. Proceedings of the Symposium on Computer Applications in "
                  "Medical Care, 261-265."),
        notes="768 women; eight clinical measurements and a diabetes diagnosis.",
    ),
    "blood": Provenance(
        slug="openml-blood-transfusion",
        name="Blood Transfusion Service Center (OpenML 1464)",
        source_url="https://openml.org/data/v1/download/1586225/blood-transfusion-service-center.arff",
        license="Public (OpenML); originally UCI, CC BY 4.0",
        citation=("Yeh, I.-C., Yang, K.-J., & Ting, T.-M. (2009). Knowledge discovery on RFM model using "
                  "Bernoulli sequence. Expert Systems with Applications, 36(3), 5866-5871."),
        notes="748 donors; recency, frequency, volume and time, and whether they donated again.",
    ),
}

TARGETS = {"titanic": "survived", "credit_g": "class", "diabetes": "class", "blood": "Class"}
POSITIVE = {"titanic": "1", "credit_g": "bad", "diabetes": "tested_positive", "blood": "2"}


def _clean(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if name == "titanic":
        # Surname and ticket identify the group a passenger travelled in. It is
        # kept as `family`, never as a feature, so a split can hold relatives
        # together; `columns()` and the models exclude it.
        surname = df["name"].astype(str).str.split(",").str[0].str.strip().str.lower()
        df["family"] = surname + "|" + df["ticket"].astype(str).str.strip()
        # name, ticket and cabin are near-unique per passenger, and boat and body
        # record what happened after the sinking, so both groups are dropped.
        df = df.drop(columns=[c for c in ("name", "ticket", "cabin", "boat", "body", "home.dest") if c in df])
    target = TARGETS[name]
    df[target] = (df[target].astype(str).str.strip() == POSITIVE[name]).astype(int)
    for c in df.columns:
        if c == target:
            continue
        # Branch on whether the column is numeric, not on whether it is `object`:
        # pandas 3 reads text into a string dtype, so an `== object` test would
        # send every categorical column through to_numeric and blank it.
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            # Plain object with NaN: scikit-learn's imputers cannot compare pandas' pd.NA.
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else np.nan).astype(object)
    return df.rename(columns={target: "target"})


def load(name: str) -> tuple[pd.DataFrame, Provenance]:
    """One dataset, cleaned, with its target renamed to `target` and coded 1 for the positive class."""
    if name not in SOURCES:
        raise KeyError(f"unknown dataset '{name}'; expected one of {sorted(SOURCES)}")
    spec = SOURCES[name]
    path, prov = fetch(spec, filename=f"{name}.arff")
    df = _clean(name, read_arff(path.read_bytes()))
    return df, stamp_frame(prov, df)


def load_all() -> dict[str, tuple[pd.DataFrame, Provenance]]:
    return {name: load(name) for name in SOURCES}


NOT_FEATURES = ("target", "family")


def columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric and categorical feature names, excluding the target and the group key."""
    features = [c for c in df.columns if c not in NOT_FEATURES]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    return numeric, [c for c in features if c not in numeric]


if __name__ == "__main__":
    for name, (df, prov) in load_all().items():
        num, cat = columns(df)
        print(f"{name:10} {df.shape[0]:>5} rows  {len(num)} numeric  {len(cat)} categorical  "
              f"positive {df['target'].mean():.3f}  missing {df.isna().any(axis=1).mean():.3f}  {prov.sha256[:10]}")
