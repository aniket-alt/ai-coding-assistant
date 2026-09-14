"""
Project 01 -- NYC Taxi trip duration: data acquisition and cleaning.

Pulls the real NYC TLC Yellow Taxi trip records and reduces ~3M raw rows to a
modelling frame. Run directly to fetch and cache:

    python ml/data.py

The important decision in this module is about *what is knowable at booking
time*, which is documented at `BOOKING_TIME_NOTE` below and enforced in
`train.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dsfs import Provenance, fetch, stamp_frame

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

TRIPS = Provenance(
    slug="nyc-tlc-yellow-2024-01",
    name="NYC TLC Yellow Taxi Trip Records — January 2024",
    source_url="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
    license="NYC Taxi & Limousine Commission open data (public domain)",
    citation=(
        "NYC Taxi & Limousine Commission (2024). TLC Trip Record Data. "
        "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
    ),
    notes="Every completed yellow-cab trip recorded in New York City in January 2024.",
)

ZONES = Provenance(
    slug="nyc-tlc-zone-lookup",
    name="NYC TLC Taxi Zone Lookup",
    source_url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
    license="NYC Taxi & Limousine Commission open data (public domain)",
    citation="NYC Taxi & Limousine Commission (2024). Taxi Zone Lookup Table.",
    notes="Maps the 265 LocationIDs to borough and zone names.",
)

# --------------------------------------------------------------------------
# What a booking actually knows
# --------------------------------------------------------------------------

BOOKING_TIME_NOTE = """
A rider requesting a taxi knows: where they are, where they are going, how many
passengers, and the current time. They do not know how far the meter will run,
what the fare will be, or when they will arrive.

The TLC file records `trip_distance`, `fare_amount` and `tpep_dropoff_datetime`,
but all three are measured *during or after* the trip. Training a
duration model on them produces a spectacular score and a useless product --
the classic form of target leakage, and the one this dataset invites.

This project therefore drops them from the feature set and reconstructs the
only distance signal a booking legitimately has: the historical median distance
for that pickup/dropoff zone pair, computed from the training window alone
(see `zone_pair_statistics` in features.py).
"""

# Columns that exist in the raw file but are unknowable when the trip is booked.
POST_TRIP_COLUMNS = [
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
    "payment_type",
    "store_and_fwd_flag",
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_raw() -> tuple[pd.DataFrame, Provenance]:
    """Download (once) and read the January 2024 trip file."""
    path, prov = fetch(TRIPS)
    df = pd.read_parquet(path)
    return df, stamp_frame(prov, df)


def load_zones() -> tuple[pd.DataFrame, Provenance]:
    """Borough and zone names for the 265 LocationIDs."""
    path, prov = fetch(ZONES)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df, stamp_frame(prov, df)


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Filter the raw records to plausible, completed street-hail trips.

    Returns the cleaned frame plus an audit trail of how many rows each rule
    removed, which the CRISP-DM data-understanding phase reports verbatim.
    """
    steps: list[dict] = []
    n0 = len(df)

    def drop(mask: pd.Series, reason: str, frame: pd.DataFrame) -> pd.DataFrame:
        removed = int((~mask).sum())
        out = frame[mask].copy()
        steps.append({
            "rule": reason,
            "removed": removed,
            "pct": round(removed / max(n0, 1) * 100, 3),
            "remaining": len(out),
        })
        return out

    d = df.copy()
    d["tpep_pickup_datetime"] = pd.to_datetime(d["tpep_pickup_datetime"])
    d["tpep_dropoff_datetime"] = pd.to_datetime(d["tpep_dropoff_datetime"])

    # The target, computed before the dropoff column is discarded.
    d["duration_min"] = (
        d["tpep_dropoff_datetime"] - d["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    # The file is titled January 2024 but contains a handful of stray records
    # from other months -- a real quirk of TLC exports.
    in_month = (
        (d["tpep_pickup_datetime"] >= "2024-01-01")
        & (d["tpep_pickup_datetime"] < "2024-02-01")
    )
    d = drop(in_month, "pickup timestamp outside January 2024", d)

    # Duration: 1 minute to 3 hours. Below a minute is a meter misfire; above
    # three hours in a yellow cab is almost always a forgotten meter.
    d = drop(d["duration_min"].between(1.0, 180.0), "duration outside [1, 180] minutes", d)

    # Distance sanity (used only for filtering, never as a feature).
    d = drop(d["trip_distance"].between(0.1, 100.0), "metered distance outside [0.1, 100] miles", d)

    # Fares must be positive; negatives are voided/refunded records.
    d = drop(d["fare_amount"] > 0, "non-positive fare (voided record)", d)

    # Passenger count of 0 is a data-entry artefact; >6 exceeds legal capacity.
    d = drop(d["passenger_count"].fillna(1).between(1, 6), "passenger count outside [1, 6]", d)

    # Location IDs 264/265 are "Unknown"/"Outside NYC" sentinels.
    valid_loc = (
        d["PULocationID"].between(1, 263) & d["DOLocationID"].between(1, 263)
    )
    d = drop(valid_loc, "unknown or out-of-city zone sentinel (264/265)", d)

    # Implied average speed must be physically possible for Manhattan traffic.
    speed = d["trip_distance"] / (d["duration_min"] / 60.0)
    d = drop(speed.between(1.0, 70.0), "implied speed outside [1, 70] mph", d)

    # A minority of records arrive without passenger_count or RatecodeID -- the
    # same rows in both cases, i.e. one vendor's telemetry rather than random
    # loss. Those rows are kept (they are valid trips) but the missingness is
    # preserved as a feature, since "which vendor recorded this" carries signal.
    n_missing = int(d["passenger_count"].isna().sum())
    d["vendor_fields_missing"] = d["passenger_count"].isna().astype(int)
    d["passenger_count"] = d["passenger_count"].fillna(1.0)
    d["RatecodeID"] = d["RatecodeID"].fillna(1.0)
    steps.append({
        "rule": "imputed missing passenger_count / RatecodeID (indicator retained)",
        "removed": 0,
        "pct": 0.0,
        "remaining": len(d),
        "imputed": n_missing,
    })

    return d.reset_index(drop=True), steps


def build_modelling_frame(sample: int | None = None, seed: int = 42) -> dict:
    """
    Full data step: fetch, clean, attach zone names, and return everything the
    CRISP-DM data-understanding phase needs.
    """
    raw, prov = load_raw()
    zones, zone_prov = load_zones()
    cleaned, steps = clean(raw)

    # Chronological order matters: the split in train.py is by time, so the
    # frame is sorted here once and never shuffled.
    cleaned = cleaned.sort_values("tpep_pickup_datetime").reset_index(drop=True)

    if sample and sample < len(cleaned):
        # Systematic sample preserves the time distribution; a random sample
        # would too, but this keeps the chronology exactly proportional.
        step = len(cleaned) / sample
        idx = (np.arange(sample) * step).astype(int)
        cleaned = cleaned.iloc[idx].reset_index(drop=True)

    zone_names = zones.set_index("LocationID")[["Borough", "Zone"]].to_dict("index")

    return {
        "df": cleaned,
        "provenance": stamp_frame(prov, cleaned),
        "zone_provenance": zone_prov,
        "zones": zones,
        "zone_names": zone_names,
        "cleaning_steps": steps,
        "n_raw": len(raw),
    }


if __name__ == "__main__":
    import json

    out = build_modelling_frame()
    df = out["df"]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out["provenance"].save(ARTIFACTS / "provenance.json")

    print(f"\nraw rows      : {out['n_raw']:,}")
    print(f"cleaned rows  : {len(df):,}  ({len(df)/out['n_raw']*100:.1f}% retained)")
    print(f"date range    : {df['tpep_pickup_datetime'].min()} -> {df['tpep_pickup_datetime'].max()}")
    print(f"duration (min): median {df['duration_min'].median():.1f}, "
          f"p95 {df['duration_min'].quantile(0.95):.1f}")
    print("\ncleaning audit trail:")
    for s in out["cleaning_steps"]:
        print(f"  -{s['removed']:>9,} ({s['pct']:>5.2f}%)  {s['rule']}")
    print(f"\nprovenance written to {ARTIFACTS / 'provenance.json'}")
    print(json.dumps({"sha256": out["provenance"].sha256[:32]}, indent=2))
