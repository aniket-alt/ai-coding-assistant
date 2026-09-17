"""
Project 13 -- seven months of real NYC taxi trips.

Project 01 trained a booking-time duration model on January 2024. This project
operates that model through the following six months of genuine TLC records,
cleaned with project 01's own rules so the only thing that changes month to
month is the city.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from dsfs import Provenance, fetch, stamp_frame

REPO = Path(__file__).resolve().parents[2]


def project01(module: str):
    """
    Load one of project 01's modules under a distinct name. This project has its
    own data.py, so a plain `import data` would import itself.
    """
    name = f"project01_{module}"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO / "01_nyc_taxi_prediction" / "ml" / f"{module}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Registered before execution so objects defined in the module (the target
    # encoders) can be pickled with the model and found again when loaded.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


taxi = project01("data")          # project 01's loader and cleaning rules

TRAIN_MONTH = "2024-01"
MONITOR_MONTHS = ["2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07"]
ROWS_PER_MONTH = 150_000
SEED = 42


def month_spec(month: str) -> Provenance:
    stamp = pd.Timestamp(f"{month}-01")
    return Provenance(
        slug=f"nyc-tlc-yellow-{month}",
        name=f"NYC TLC Yellow Taxi Trip Records — {stamp:%B %Y}",
        source_url=f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet",
        license=taxi.TRIPS.license,
        citation=taxi.TRIPS.citation,
        notes=f"Every completed yellow-cab trip recorded in New York City in {stamp:%B %Y}.",
    )


def load_month(month: str) -> tuple[pd.DataFrame, dict, Provenance]:
    """
    One month, cleaned with project 01's rules and systematically sampled.

    Sampling takes every n-th trip in time order, which keeps the month's
    hour-by-hour and day-by-day shape intact.
    """
    path, prov = fetch(month_spec(month))
    raw = pd.read_parquet(path)
    prov = stamp_frame(prov, raw)
    cleaned, steps = taxi.clean(raw, month=month)
    cleaned = cleaned.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    step = max(1, len(cleaned) // ROWS_PER_MONTH)
    sample = cleaned.iloc[::step].head(ROWS_PER_MONTH).reset_index(drop=True)
    sample["month"] = month
    facts = {"month": month, "raw_trips": int(len(raw)), "clean_trips": int(len(cleaned)),
             "sampled": int(len(sample)), "median_duration": float(cleaned["duration_min"].median()),
             "median_distance": float(cleaned["trip_distance"].median())}
    return sample, facts, prov


def zones() -> dict[int, str]:
    frame, _ = taxi.load_zones()
    return {int(r.LocationID): f"{r.Zone} ({r.Borough})" for r in frame.itertuples()}
