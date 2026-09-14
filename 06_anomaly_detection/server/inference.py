"""
Project 06 -- serving.

Scores a transaction and, more usefully, says where that score sits. A raw
suspicion score is close to meaningless on its own: what an analyst needs to
know is whether this transaction would be in today's first hundred alerts.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

ARTIFACTS = ROOT / "artifacts"


class ModelNotTrained(RuntimeError):
    """Raised when the API starts before `python ml/train.py` has run."""


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    path = ARTIFACTS / "model.joblib"
    if not path.exists():
        raise ModelNotTrained(
            "artifacts/model.joblib not found -- run `python ml/train.py` first."
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_run() -> dict[str, Any]:
    path = ARTIFACTS / "metrics.json"
    if not path.exists():
        raise ModelNotTrained("artifacts/metrics.json not found.")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _reference_scores() -> np.ndarray:
    """
    Held-out score distribution, used to place a new score in context.

    Recomputed once from the test window rather than stored, then cached; the
    percentile a transaction lands in is only meaningful against the same
    distribution the budget curve was measured on.
    """
    from data import load

    bundle = load_bundle()
    df, _ = load()
    test = df[df["Time"] >= 35 * 3600]
    X = test[bundle["feature_order"]]
    return np.sort(bundle["model"].predict_proba(X)[:, 1])


def sample_transactions(n: int = 12, fraud_ratio: float = 0.5,
                        seed: int = 7) -> list[dict[str, Any]]:
    """
    Real transactions from the held-out window, for the UI to load as presets.

    Deliberately over-samples fraud so the explorer has interesting cases; the
    label travels with each row so the UI can show whether a score was right.
    """
    from data import load

    df, _ = load()
    test = df[df["Time"] >= 35 * 3600]
    rng = np.random.default_rng(seed)

    fraud = test[test["Class"] == 1]
    legit = test[test["Class"] == 0]
    n_fraud = min(int(n * fraud_ratio), len(fraud))
    picked = pd.concat([
        fraud.sample(n=n_fraud, random_state=seed),
        legit.sample(n=n - n_fraud, random_state=seed),
    ])
    picked = picked.iloc[rng.permutation(len(picked))]

    out = []
    for i, row in enumerate(picked.itertuples(index=False)):
        d = row._asdict()
        out.append({
            "id": i,
            "label": (f"€{d['Amount']:.2f} at hour {int(d['Time'] // 3600)}"
                      f"{' · known fraud' if d['Class'] == 1 else ''}"),
            "is_fraud": int(d["Class"]),
            "features": {k: float(v) for k, v in d.items() if k != "Class"},
        })
    return out


def score(features: dict[str, float]) -> dict[str, Any]:
    """Score one transaction and locate it in the held-out distribution."""
    bundle = load_bundle()
    order = bundle["feature_order"]
    budget = int(bundle["review_budget"])

    row = pd.DataFrame([{k: float(features.get(k, 0.0)) for k in order}])
    s = float(bundle["model"].predict_proba(row)[0, 1])

    ref = _reference_scores()
    # Rank from the top: how many held-out transactions look worse than this.
    rank_from_top = int(np.sum(ref > s)) + 1
    percentile = float(np.searchsorted(ref, s) / len(ref) * 100)

    return {
        "score": s,
        "percentile": round(percentile, 4),
        "rank_from_top": rank_from_top,
        "reference_population": int(len(ref)),
        "would_be_reviewed": rank_from_top <= budget,
        "review_budget": budget,
    }


def budget_curve() -> list[dict[str, Any]]:
    """Precision and recall at each review budget, as measured during training."""
    run = load_run()
    return list(run.get("extra", {}).get("budget_curve", []))


def eda() -> dict[str, Any]:
    """Hourly and amount-band profiles recorded by the training run."""
    extra = load_run().get("extra", {})
    return {
        "hourly_profile": extra.get("hourly_profile", []),
        "amount_bands": extra.get("amount_bands", []),
        "dataset": extra.get("dataset", {}),
    }
