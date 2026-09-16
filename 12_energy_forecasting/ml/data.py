"""
Project 12 -- household electricity: acquisition and hourly aggregation.

Four years of one household's electricity use in Sceaux, near Paris, recorded
once a minute. The series has real gaps -- the longest lasts five days -- so
aggregation and scoring have to treat missing data explicitly rather than
interpolating over it.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from dsfs import Provenance, fetch, stamp_frame

POWER = Provenance(
    slug="uci-household-power",
    name="Individual Household Electric Power Consumption (UCI)",
    source_url="https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
    license="Creative Commons Attribution 4.0 (CC BY 4.0)",
    citation=(
        "Hebrail, G., & Berard, A. (2006). Individual Household Electric Power "
        "Consumption [Dataset]. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C58K54"
    ),
    notes="One household in Sceaux, France, sampled every minute from December 2006 to November 2010.",
)

# An hour is only trusted if most of its minutes were recorded.
MIN_MINUTES_PER_HOUR = 45


def load_minutes() -> tuple[pd.DataFrame, Provenance]:
    """Download (once) and parse the minute-level file. Missing readings are '?'."""
    path, prov = fetch(POWER, filename="household_power.zip")
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        df = pd.read_csv(z.open(name), sep=";", na_values="?", low_memory=False)
    df.index = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"]).sort_index()
    return df, stamp_frame(prov, df)


def hourly(minutes: pd.DataFrame) -> tuple[pd.Series, dict]:
    """
    Mean active power per hour, in kilowatts.

    An hour with fewer than MIN_MINUTES_PER_HOUR readings is left missing rather
    than averaged from a handful of minutes, and gaps are never interpolated: a
    forecast scored against an invented value would be scored against nothing.
    """
    kw = minutes["Global_active_power"]
    grouped = kw.resample("h")
    mean = grouped.mean()
    count = grouped.count()
    series = mean.where(count >= MIN_MINUTES_PER_HOUR)
    # Drop the partial first and last hours of the recording.
    series = series.iloc[1:-1]

    missing = series.isna()
    runs = (missing != missing.shift()).cumsum()[missing]
    gap_lengths = runs.value_counts()
    facts = {
        "minutes": int(len(kw)),
        "missing_minutes": int(kw.isna().sum()),
        "hours": int(len(series)),
        "missing_hours": int(missing.sum()),
        "longest_gap_hours": int(gap_lengths.max()) if len(gap_lengths) else 0,
        "start": str(series.index.min()),
        "end": str(series.index.max()),
    }
    return series.rename("kw"), facts


def profiles(series: pd.Series) -> dict:
    """Average load by hour of day for weekdays and weekends, and by month."""
    frame = series.dropna().to_frame()
    frame["hour"] = frame.index.hour
    frame["weekend"] = frame.index.dayofweek >= 5
    frame["month"] = frame.index.month
    daily = frame.groupby(["hour", "weekend"])["kw"].mean().unstack()
    monthly = frame.groupby("month")["kw"].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return {
        "by_hour": [{"hour": int(h), "weekday_kw": round(float(daily.loc[h, False]), 3),
                     "weekend_kw": round(float(daily.loc[h, True]), 3)} for h in daily.index],
        "by_month": [{"month": month_names[m - 1], "kw": round(float(v), 3)} for m, v in monthly.items()],
    }


def summarise(series: pd.Series) -> dict:
    s = series.dropna()
    return {
        "mean_kw": float(s.mean()),
        "median_kw": float(s.median()),
        "p99_kw": float(s.quantile(0.99)),
        "winter_to_summer_ratio": float(
            s[s.index.month.isin([12, 1, 2])].mean() / s[s.index.month.isin([6, 7, 8])].mean()
        ),
        "daily_autocorrelation": float(s.autocorr(lag=24)),
        "weekly_autocorrelation": float(s.autocorr(lag=168)),
        "zero_share": float((s < 0.1).mean()),
    }


if __name__ == "__main__":
    minutes, prov = load_minutes()
    series, facts = hourly(minutes)
    print(facts)
    print(summarise(series))
    print(np.round(series.describe(), 3))
