"""
Shared evaluation for probabilistic forecasts.

Used by the SPY return forecaster (project 15) and the household energy
forecaster (project 12), so both score quantile forecasts and test significance
in exactly the same way.
"""

from __future__ import annotations

import numpy as np

QUANTILES = (0.1, 0.5, 0.9)


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Quantile (pinball) loss. Asymmetric: under- and over-prediction differ."""
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def pinball_series(y: np.ndarray, q10: np.ndarray, q50: np.ndarray,
                   q90: np.ndarray) -> np.ndarray:
    """Per-observation pinball loss, averaged over the three quantiles."""
    out = np.zeros(len(y))
    for q, pred in zip(QUANTILES, (q10, q50, q90)):
        d = y - pred
        out += np.maximum(q * d, (q - 1) * d)
    return out / len(QUANTILES)


def interval_scores(y: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> dict[str, float]:
    """
    Does the stated 80% interval contain 80% of outcomes? An interval that is
    too narrow is confidently wrong; one that is too wide is merely vague.
    """
    covered = float(np.mean((y >= q10) & (y <= q90)))
    return {
        "coverage_80": covered,
        "coverage_error": abs(covered - 0.8),
        "interval_width": float(np.mean(q90 - q10)),
    }


def paired_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray,
                     n: int = 5000, seed: int = 42) -> dict:
    """
    Is model A's loss genuinely lower than model B's, or is the gap noise?

    Resamples the paired per-observation loss differences. Reporting a winner
    without this check overstates how much was learned whenever good
    forecasters are close together.
    """
    rng = np.random.default_rng(seed)
    diff = loss_a - loss_b
    # Drawn in chunks so memory stays bounded for long hourly series. The
    # generator yields the same stream either way, so results are unchanged.
    means = np.concatenate([
        diff[rng.integers(0, len(diff), size=(min(250, n - start), len(diff)))].mean(axis=1)
        for start in range(0, n, 250)
    ])
    lo, hi = np.quantile(means, [0.025, 0.975])
    # A negative difference means lower loss, i.e. better than the reference.
    # The verdict is only "better" or "worse" when the whole interval agrees.
    verdict = "better" if hi < 0 else "worse" if lo > 0 else "inconclusive"
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "verdict": verdict,
        "significant": verdict != "inconclusive",
        "better_than_reference": verdict == "better",
    }
