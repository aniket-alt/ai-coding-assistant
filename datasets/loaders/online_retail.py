"""
Online Retail II -- shared loader for projects 03 and 04.

Every transaction of a UK online gift retailer from December 2009 to December
2011, published by UCI. Both the segmentation and market-basket projects start
from the same cleaned lines, so the cleaning rules live here once rather than
diverging in two copies.

The export has quirks that matter:

* It ships as two yearly sheets that **overlap by nine days** (1-9 December
  2010). Concatenating them naively counts those orders twice.
* About a fifth of lines carry no customer ID.
* Cancellations are separate lines whose invoice number starts with "C".
* Several thousand lines are not products at all: postage, bank charges,
  marketplace fees, manual adjustments and gift vouchers.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from dsfs import Provenance, fetch, stamp_frame

RETAIL = Provenance(
    slug="uci-online-retail-ii",
    name="Online Retail II (UCI)",
    source_url="https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
    license="Creative Commons Attribution 4.0 (CC BY 4.0)",
    citation=(
        "Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning "
        "Repository. https://doi.org/10.24432/C5CG6D"
    ),
    notes=(
        "All transactions of a UK-based online gift retailer, 2009-12-01 to "
        "2011-12-09, as two overlapping yearly sheets."
    ),
)

# Product stock codes are five digits with an optional letter suffix
# (85123A). Everything else is postage, fees, adjustments or vouchers.
PRODUCT_CODE = r"^\d{5}[A-Za-z]{0,2}$"


def _parquet_cache(zip_path: Path) -> Path:
    """Reading the Excel file takes about a minute; cache it as parquet once."""
    out = zip_path.with_name("online_retail_ii.parquet")
    if out.exists():
        return out
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        sheets = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=None,
                               dtype={"Invoice": str, "StockCode": str})
    frame = pd.concat(sheets.values(), ignore_index=True)
    # Some descriptions are bare numbers in the spreadsheet; keep them as text.
    frame["Description"] = frame["Description"].astype("string")
    frame.to_parquet(out)
    return out


def load_raw() -> tuple[pd.DataFrame, Provenance]:
    """Download (once) and return every line of both sheets, unmodified."""
    path, prov = fetch(RETAIL, filename="online_retail_ii.zip")
    df = pd.read_parquet(_parquet_cache(path))
    df = df.rename(columns={"Customer ID": "CustomerID"})
    return df, stamp_frame(prov, df)


def clean(raw: pd.DataFrame) -> dict:
    """
    Reduce raw lines to genuine product sales, keeping cancellations aside.

    Returns the sales lines, the cancellation lines (still useful: a customer's
    return rate is behaviour), and an audit trail of what each rule removed.
    """
    n0 = len(raw)
    steps: list[dict] = []

    def record(rule: str, removed: int, remaining: int) -> None:
        steps.append({"rule": rule, "removed": int(removed),
                      "pct": round(removed / n0 * 100, 3), "remaining": int(remaining)})

    d = raw.copy()

    before = len(d)
    d = d.drop_duplicates()
    record("exact duplicate lines (includes the 9-day overlap between sheets)",
           before - len(d), len(d))

    before = len(d)
    d = d[d["CustomerID"].notna()].copy()
    record("no customer ID", before - len(d), len(d))
    d["CustomerID"] = d["CustomerID"].astype("int64")

    is_cancel = d["Invoice"].str.startswith("C")
    cancellations = d[is_cancel].copy()
    d = d[~is_cancel]
    record("cancellation lines (kept aside for return rates)", int(is_cancel.sum()), len(d))

    before = len(d)
    d = d[d["StockCode"].str.match(PRODUCT_CODE)]
    record("non-product codes (postage, fees, adjustments, vouchers)", before - len(d), len(d))

    before = len(d)
    d = d[(d["Quantity"] > 0) & (d["Price"] > 0)].copy()
    record("non-positive quantity or price", before - len(d), len(d))

    d["Revenue"] = d["Quantity"] * d["Price"]
    return {
        "sales": d.sort_values("InvoiceDate").reset_index(drop=True),
        "cancellations": cancellations.reset_index(drop=True),
        "steps": steps,
        "n_raw": n0,
    }
