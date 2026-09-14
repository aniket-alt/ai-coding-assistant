"""
Project 15 -- walk-forward forecasting tournament.

    python ml/train.py [--horizon 1]

Every number here comes from a walk-forward backtest: the model is refitted
quarterly on data available at that point and then judged on the following
quarter, which it has never seen. There is no single train/test split, because
a single split on eleven years of market data mostly measures which regime
happened to land in the test set.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage

from data import WHY_RETURNS_NOT_PRICES, load, price_history, summarise
from features import FEATURE_COLUMNS, assert_no_lookahead, build

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEED = 42

QUANTILES = (0.1, 0.5, 0.9)
BURN_IN = 750        # ~3 years before the first forecast
REFIT_EVERY = 63     # refit quarterly
COST_BPS = 2.0       # round-trip execution cost, basis points


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Quantile (pinball) loss. Asymmetric: under- and over-prediction differ."""
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def evaluate(y: np.ndarray, q10: np.ndarray, q50: np.ndarray,
             q90: np.ndarray) -> dict[str, float]:
    """Score a set of walk-forward quantile forecasts."""
    losses = [pinball(y, q10, 0.1), pinball(y, q50, 0.5), pinball(y, q90, 0.9)]

    # Does the stated 80% interval actually contain 80% of outcomes? A model
    # whose intervals are wrong is worse than useless: it is confidently wrong.
    covered = float(np.mean((y >= q10) & (y <= q90)))
    width = float(np.mean(q90 - q10))

    # Direction is what a trader acts on.
    nonzero = q50 != 0
    hit = float(np.mean(np.sign(q50[nonzero]) == np.sign(y[nonzero]))) if nonzero.any() else 0.5

    ss_res = float(np.sum((y - q50) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    return {
        "pinball": float(np.mean(losses)),
        "pinball_q10": losses[0],
        "pinball_q50": losses[1],
        "pinball_q90": losses[2],
        "coverage_80": covered,
        "coverage_error": abs(covered - 0.8),
        "interval_width": width,
        "hit_rate": hit,
        "mae": float(np.mean(np.abs(y - q50))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
    }


def pinball_series(y: np.ndarray, q10: np.ndarray, q50: np.ndarray,
                   q90: np.ndarray) -> np.ndarray:
    """Per-observation pinball loss, averaged over the three quantiles."""
    out = np.zeros(len(y))
    for q, pred in zip(QUANTILES, (q10, q50, q90)):
        d = y - pred
        out += np.maximum(q * d, (q - 1) * d)
    return out / len(QUANTILES)


def paired_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray,
                     n: int = 5000, seed: int = SEED) -> dict:
    """
    Is model A's loss genuinely lower than model B's, or is the gap noise?

    Resamples the paired per-day loss differences. With eleven years of daily
    data the differences between good forecasters are small, and reporting a
    winner without this check overstates how much was actually learned.
    """
    rng = np.random.default_rng(seed)
    diff = loss_a - loss_b
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
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


def backtest(y: np.ndarray, q50: np.ndarray, dates: pd.DatetimeIndex,
             cost_bps: float = COST_BPS) -> dict:
    """
    Long-when-positive strategy, with execution cost charged on every change
    of position. Returns the honest Sharpe, not a flattering one.
    """
    position = (q50 > 0).astype(float)
    turnover = np.abs(np.diff(np.concatenate([[0.0], position])))
    cost = turnover * (cost_bps / 10_000.0)
    strat = position * y - cost

    ann = np.sqrt(252)
    sharpe = float(strat.mean() / strat.std() * ann) if strat.std() > 0 else 0.0
    equity = np.exp(np.cumsum(strat))
    dd = float((equity / np.maximum.accumulate(equity)).min() - 1.0)

    bh_sharpe = float(y.mean() / y.std() * ann) if y.std() > 0 else 0.0
    bh_equity = np.exp(np.cumsum(y))

    return {
        "sharpe": sharpe,
        "ann_return_pct": float(strat.mean() * 252 * 100),
        "max_drawdown_pct": dd * 100,
        "time_in_market_pct": float(position.mean() * 100),
        "n_trades": int(turnover.sum()),
        "buy_hold_sharpe": bh_sharpe,
        "buy_hold_return_pct": float(y.mean() * 252 * 100),
        "curve": [
            {"date": str(d.date()), "strategy": round(float(e), 4),
             "buy_hold": round(float(b), 4)}
            for d, e, b in zip(dates[::5], equity[::5], bh_equity[::5])
        ],
    }


# --------------------------------------------------------------------------
# Walk-forward engine
# --------------------------------------------------------------------------

def walk_forward(X: pd.DataFrame, y: pd.Series, horizon: int,
                 fit_predict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, int]:
    """
    Expanding-window walk forward.

    At each refit the model sees only rows strictly before the test block, and
    the final `horizon - 1` training rows are **purged**: their targets overlap
    the test window, so keeping them would leak the future backwards.
    """
    n = len(X)
    q10s, q50s, q90s, ys, idx = [], [], [], [], []
    refits = 0

    start = BURN_IN
    while start < n:
        stop = min(start + REFIT_EVERY, n)

        # Purge the overlap between the end of training and the test block.
        train_end = start - (horizon - 1)
        if train_end < 100:
            start = stop
            continue

        Xtr, ytr = X.iloc[:train_end], y.iloc[:train_end]
        Xte, yte = X.iloc[start:stop], y.iloc[start:stop]

        ok = ytr.notna()
        if ok.sum() < 100:
            start = stop
            continue

        pred = fit_predict(Xtr[ok], ytr[ok], Xte)
        refits += 1

        q10s.append(pred[0]); q50s.append(pred[1]); q90s.append(pred[2])
        ys.append(yte.to_numpy()); idx.append(Xte.index)
        start = stop

    y_all = np.concatenate(ys)
    mask = ~np.isnan(y_all)
    dates = pd.DatetimeIndex(np.concatenate([i.to_numpy() for i in idx]))

    return (np.concatenate(q10s)[mask], np.concatenate(q50s)[mask],
            np.concatenate(q90s)[mask], y_all[mask], dates[mask], refits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1, choices=list((1, 5)))
    args = ap.parse_args()
    h = args.horizon

    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()

    report = CrispDmReport(
        project_slug="15_spy_forecasting",
        project_title="SPY Probabilistic Return Forecasting",
        business_question=(
            "How much can tomorrow's S&P 500 move actually be predicted, and "
            "how wide should the honest uncertainty band be?"
        ),
    )

    # -- 1. business understanding ------------------------------------------
    report.phase(
        "business_understanding",
        objective=(
            "Produce a calibrated distribution for the next day's return, and "
            "establish honestly whether it beats doing nothing."
        ),
        narrative=(
            "The useful output of a market model is not a number but a "
            "**distribution**. A point forecast of +0.03% is worthless without "
            "knowing whether the plausible range around it is ±0.5% or ±5%; "
            "the range is what determines position size and risk.\n\n"
            "So this project forecasts the 10th, 50th and 90th percentiles and "
            "is scored on **pinball loss**, with interval **calibration** "
            "treated as a first-class result: an 80% band that contains 80% of "
            "outcomes is useful, and one that contains 55% is dangerous.\n"
            + WHY_RETURNS_NOT_PRICES
        ),
        code_refs=["ml/data.py:WHY_RETURNS_NOT_PRICES", "ml/train.py:pinball"],
        gate_notes="Target, loss function and calibration criterion fixed before modelling.",
    ).add("Primary metric", "Pinball loss", "lower is better") \
     .add("Calibration target", "80% interval covers 80%") \
     .add("Horizon", f"t+{h} trading day{'s' if h > 1 else ''}")

    # -- 2. data understanding ----------------------------------------------
    print("[1/6] loading real market data ...")
    df, prov = load()
    prov.save(ARTIFACTS / "provenance.json")
    s = summarise(df)

    report.phase(
        "data_understanding",
        objective="Characterise the return series before assuming anything about it.",
        narrative=(
            f"{s['n_days']:,} trading days from {s['start']} to {s['end']}. SPY "
            f"went from \\${s['first_close']:.2f} to \\${s['last_close']:.2f}, "
            f"a total return of {s['total_return_pct']:+.0f}% — "
            f"{s['ann_return_pct']:+.1f}% annualised against "
            f"{s['ann_vol_pct']:.1f}% annualised volatility.\n\n"
            f"**Buy-and-hold Sharpe over this window is "
            f"{s['sharpe_buy_hold']:.2f}.** That is the number any strategy has "
            f"to beat, and it is the reference point that makes reported "
            f"Sharpe ratios of 2 or 3 on daily equity data implausible on "
            f"their face.\n\n"
            f"Returns are not normal. Excess kurtosis is "
            f"**{s['excess_kurtosis']:.1f}** against 0 for a Gaussian, and the "
            f"window contains a {s['worst_day_pct']:.1f}% day "
            f"({s['worst_day_date']}) and a {s['best_day_pct']:+.1f}% day "
            f"({s['best_day_date']}). Any interval derived from a normal "
            f"assumption will be too narrow in exactly the moments that matter. "
            f"Maximum drawdown was {s['max_drawdown_pct']:.0f}%.\n\n"
            f"{s['pct_up_days']:.1f}% of days are positive — the thin edge that "
            f"makes buy-and-hold work, and roughly the hit rate a directional "
            f"model must exceed to add anything."
        ),
        code_refs=["ml/data.py:summarise"],
        gate_notes="Distributional properties established; normality rejected before modelling.",
    ).add("Trading days", s["n_days"]) \
     .add("Buy-hold Sharpe", f"{s['sharpe_buy_hold']:.2f}", "the bar to beat") \
     .add("Annualised vol", f"{s['ann_vol_pct']:.1f}%") \
     .add("Excess kurtosis", f"{s['excess_kurtosis']:.1f}", "normal = 0") \
     .add("Max drawdown", f"{s['max_drawdown_pct']:.0f}%") \
     .add("Up days", f"{s['pct_up_days']:.1f}%")

    # -- 3. data preparation ------------------------------------------------
    print("[2/6] features + lookahead verification ...")
    feats = build(df)
    look = assert_no_lookahead(feats, df, n_checks=30)
    leakage._record(
        "assert_no_lookahead",
        passed=look["passed"],
        detail=(
            f"{look['values_compared']} feature values across "
            f"{look['dates_checked']} dates recomputed on truncated history; "
            f"{look['n_mismatches']} differed from the full-history value."
        ),
        dates_checked=look["dates_checked"],
        values_compared=look["values_compared"],
        mismatches=look["n_mismatches"],
    )
    if not look["passed"]:
        raise RuntimeError(f"lookahead detected: {look['mismatches']}")

    target = f"target_h{h}"
    usable = feats[FEATURE_COLUMNS + [target]].copy()
    usable = usable.iloc[200:]           # discard the 200-day MA warm-up
    X = usable[FEATURE_COLUMNS]
    y = usable[target]

    leakage.assert_no_target_leak(X.iloc[:BURN_IN], y.iloc[:BURN_IN].fillna(0))

    n_folds = int(np.ceil((len(X) - BURN_IN) / REFIT_EVERY))
    report.phase(
        "data_preparation",
        objective="Build features that use only the past, and prove it rather than assert it.",
        narrative=(
            f"{len(FEATURE_COLUMNS)} features: lagged returns, realised "
            f"volatility over four windows, the Parkinson range estimator, "
            f"momentum and trend, the VIX and its relation to realised "
            f"volatility (the variance risk premium), the 10-year yield, "
            f"relative volume, and calendar effects.\n\n"
            "Every one is computed from data available at the close of day $t$, "
            "and the target is the return over $[t, t+h]$. That rule is easy to "
            "break by accident — a centred rolling window, a `shift()` in the "
            "wrong direction — so it is **verified empirically**: the feature "
            "block is recomputed on truncated history, as if standing at each "
            "of 30 randomly chosen dates with nothing after it, and every value "
            "must match what the full-history computation produced.\n\n"
            f"That check compared **{look['values_compared']} values** and "
            f"found {look['n_mismatches']} discrepancies. A feature that peeks "
            f"ahead cannot survive it.\n\n"
            f"Evaluation is walk-forward across **{n_folds} quarterly refits** "
            f"rather than one split, and each refit **purges** the last "
            f"{h - 1 if h > 1 else 0} training rows whose targets overlap the "
            f"test block."
        ),
        code_refs=["ml/features.py:assert_no_lookahead", "ml/train.py:walk_forward"],
        gate_notes=f"Lookahead check passed over {look['values_compared']} recomputed values.",
    ).add("Features", len(FEATURE_COLUMNS)) \
     .add("Walk-forward folds", n_folds) \
     .add("Burn-in", f"{BURN_IN} days") \
     .add("Values re-verified", look["values_compared"]) \
     .add("Lookahead mismatches", look["n_mismatches"])

    # -- 4. modeling ---------------------------------------------------------
    print(f"[3/6] walk-forward tournament (horizon t+{h}) ...")
    run = RunRecord(
        project_slug="15_spy_forecasting",
        project_title="SPY Probabilistic Return Forecasting",
        task="probabilistic time series forecasting",
        primary_metric="pinball",
        higher_is_better=False,
    )
    run.set_provenance(prov)

    results: dict[str, dict] = {}

    def run_model(name: str, family: str, fit_predict, notes: str = "") -> ModelScore:
        t = time.perf_counter()
        q10, q50, q90, yy, dates, refits = walk_forward(X, y, h, fit_predict)
        secs = time.perf_counter() - t
        m = evaluate(yy, q10, q50, q90)
        bt = backtest(yy, q50, dates)
        m["sharpe"] = bt["sharpe"]
        sc = ModelScore(name=name, family=family, metrics=m,
                        fit_seconds=round(secs, 2), notes=notes,
                        params={"refits": refits})
        run.add_model(sc)
        results[name] = {"backtest": bt, "dates": dates, "q50": q50,
                         "q10": q10, "q90": q90, "y": yy,
                         "loss": pinball_series(yy, q10, q50, q90)}
        print(f"      {name:<28} pinball {m['pinball']:.6f}  cover80 "
              f"{m['coverage_80']*100:5.1f}%  hit {m['hit_rate']*100:5.1f}%  "
              f"R2 {m['r2']:+.4f}  Sharpe {bt['sharpe']:+.2f}  ({secs:.1f}s)")
        return sc

    # Baseline: the random walk. Predict zero, and take the interval from
    # trailing realised volatility. In finance this is a hard baseline.
    def _randomwalk(Xtr, ytr, Xte):
        vol = Xte["vol_21"].to_numpy() / np.sqrt(252) * np.sqrt(h)
        z = stats.norm.ppf(0.9)
        zeros = np.zeros(len(Xte))
        return zeros - z * vol, zeros, zeros + z * vol
    base = run_model("Random walk (zero)", "baseline", _randomwalk,
                     "Predicts no move; interval from trailing realised volatility.")
    run.baseline = dict(base.metrics)

    # Baseline: the historical mean drift.
    def _drift(Xtr, ytr, Xte):
        mu = float(ytr.mean())
        vol = Xte["vol_21"].to_numpy() / np.sqrt(252) * np.sqrt(h)
        z = stats.norm.ppf(0.9)
        mid = np.full(len(Xte), mu)
        return mid - z * vol, mid, mid + z * vol
    run_model("Historical drift", "baseline", _drift,
              "Predicts the trailing mean return; same volatility interval.")

    # Empirical quantiles of the trailing return distribution -- no normal
    # assumption, which matters given excess kurtosis of 14.
    def _empirical(Xtr, ytr, Xte):
        recent = ytr.iloc[-500:]
        lo, mid, hi = np.quantile(recent, QUANTILES)
        n = len(Xte)
        return np.full(n, lo), np.full(n, mid), np.full(n, hi)
    run_model("Empirical quantiles", "baseline", _empirical,
              "Trailing 500-day empirical quantiles; no distributional assumption.")

    # Ridge on the feature set, with residual-based intervals.
    def _ridge(Xtr, ytr, Xte):
        p = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=10.0, random_state=SEED))])
        p.fit(Xtr, ytr)
        mid = p.predict(Xte)
        resid = ytr.to_numpy() - p.predict(Xtr)
        lo_off, hi_off = np.quantile(resid, [0.1, 0.9])
        return mid + lo_off, mid, mid + hi_off
    run_model("Ridge + residual quantiles", "linear", _ridge,
              "Scaler fitted per fold inside the pipeline; intervals from in-fold residuals.")

    lgb_holder: dict[str, object] = {}

    # LightGBM fitted directly on the pinball objective, one model per quantile.
    def _lgbm(Xtr, ytr, Xte):
        preds = []
        for q in QUANTILES:
            m = lgb.LGBMRegressor(
                objective="quantile", alpha=q,
                n_estimators=250, learning_rate=0.03, num_leaves=15,
                min_child_samples=40, reg_lambda=1.0,
                random_state=SEED, n_jobs=-1, verbose=-1,
            )
            m.fit(Xtr, ytr)
            preds.append(m.predict(Xte))
            if q == 0.5:
                lgb_holder["m"] = m
        return preds[0], preds[1], preds[2]
    run_model("LightGBM quantile", "gbdt", _lgbm,
              "Three boosters trained directly on the pinball objective.")

    # Volatility-scaled random walk: zero mean, but the interval comes from an
    # EWMA volatility estimate that reacts faster than a 21-day window.
    def _ewma(Xtr, ytr, Xte):
        lam = 0.94
        v5 = Xte["vol_5"].to_numpy() / np.sqrt(252)
        v21 = Xte["vol_21"].to_numpy() / np.sqrt(252)
        vol = np.sqrt(lam * v21 ** 2 + (1 - lam) * v5 ** 2) * np.sqrt(h)
        # Student-t quantile, to respect the fat tails the data showed.
        t_q = stats.t.ppf(0.9, df=4)
        zeros = np.zeros(len(Xte))
        return zeros - t_q * vol, zeros, zeros + t_q * vol
    run_model("EWMA vol + Student-t", "volatility", _ewma,
              "Zero drift; interval from EWMA volatility with t(4) quantiles for fat tails.")

    best = run.best
    assert best is not None

    report.phase(
        "modeling",
        objective="Compare feature-driven models against baselines that assume returns are unpredictable.",
        narrative=(
            "The baselines here are not straw men. A random walk with a "
            "volatility-derived interval is close to the best available answer "
            "for daily equity returns, and beating it is the entire question.\n\n"
            "Two families are represented: models that try to predict the "
            "**direction** (Ridge, LightGBM quantile) and models that concede "
            "direction is unpredictable and concentrate on getting the "
            "**width** right (random walk, EWMA with Student-$t$ quantiles). "
            "Given excess kurtosis of "
            f"{s['excess_kurtosis']:.0f}, the second family is more defensible "
            "than it first sounds."
        ),
        code_refs=["ml/train.py:walk_forward", "ml/train.py:evaluate"],
        gate_notes=f"{len(run.models)} models across {n_folds} walk-forward folds.",
    ).add("Models", len(run.models)) \
     .add("Winner", best.name) \
     .add("Pinball loss", f"{best.metrics['pinball']:.6f}") \
     .add("80% coverage", f"{best.metrics['coverage_80']*100:.1f}%") \
     .add("Hit rate", f"{best.metrics['hit_rate']*100:.1f}%")

    # -- 5. evaluation -------------------------------------------------------
    print("[4/6] evaluation + audit ...")
    findings = leakage.audit_source(ROOT, temporal_modules=("train", "data", "features"))
    leakage.save_report(findings, ARTIFACTS / "audit.md", "SPY Probabilistic Forecasting")
    crit = [f for f in findings if f.severity == "CRITICAL"]
    executed = leakage.ledger()
    run.audit = {
        "critical": len(crit),
        "warnings": len([f for f in findings if f.severity == "WARNING"]),
        "total": len(findings),
        "runtime_assertions": executed,
        "assertions_passed": sum(1 for a in executed if a["passed"]),
        "assertions_run": len(executed),
    }

    # Compare every model against the random walk on paired daily losses.
    rw = results["Random walk (zero)"]["loss"]
    significance = {}
    for name, r in results.items():
        if name == "Random walk (zero)":
            continue
        n = min(len(r["loss"]), len(rw))
        significance[name] = paired_bootstrap(r["loss"][:n], rw[:n])
    best_sig = significance.get(best.name, {})

    bt_best = results[best.name]["backtest"]
    report.phase(
        "evaluation",
        objective="State plainly how much signal there is, and resist the temptation to overclaim.",
        narrative=(
            f"**{best.name}** achieves the lowest pinball loss "
            f"({best.metrics['pinball']:.6f}) across {n_folds} walk-forward "
            f"folds, with an 80% interval covering "
            f"{best.metrics['coverage_80']*100:.1f}% of outcomes.\n\n"
            f"The directional hit rate is "
            f"{best.metrics['hit_rate']*100:.1f}%, against "
            f"{s['pct_up_days']:.1f}% of days being positive — so a model that "
            f"simply said *up* every day would score about the same. $R^2$ on "
            f"returns is {best.metrics['r2']:+.4f}.\n\n"
            f"**That is the finding, and it is not a disappointing one.** Daily "
            f"equity returns are close to unpredictable in their first moment; "
            f"the honest value is in the second moment, where volatility "
            f"clusters and intervals can be calibrated. Models that concede "
            f"the direction and concentrate on the width do well here for that "
            f"reason.\n\n"
            f"Traded naively — long whenever the median forecast is positive, "
            f"{COST_BPS:.0f} bps round trip — the winner produces a Sharpe of "
            f"**{bt_best['sharpe']:.2f}** against **"
            f"{bt_best['buy_hold_sharpe']:.2f}** for simply holding SPY over "
            f"the same period. Any published daily-equity strategy claiming a "
            f"Sharpe above 2 should be read with this comparison in mind."
        ),
        code_refs=["ml/train.py:backtest", "ml/train.py:evaluate"],
        gate_passed=len(crit) == 0,
        gate_notes=("No critical findings; every forecast is out-of-sample by construction."
                    if not crit else f"{len(crit)} critical findings."),
    ).add("Pinball loss", f"{best.metrics['pinball']:.6f}") \
     .add("80% coverage", f"{best.metrics['coverage_80']*100:.1f}%", "target 80%") \
     .add("Hit rate", f"{best.metrics['hit_rate']*100:.1f}%") \
     .add("R² on returns", f"{best.metrics['r2']:+.4f}") \
     .add("Strategy Sharpe", f"{bt_best['sharpe']:.2f}") \
     .add("Buy-and-hold Sharpe", f"{bt_best['buy_hold_sharpe']:.2f}")

    # -- 6. deployment -------------------------------------------------------
    print("[5/6] persisting ...")
    if lgb_holder.get("m") is not None:
        joblib.dump({"model": lgb_holder["m"], "feature_order": FEATURE_COLUMNS,
                     "horizon": h, "model_name": "LightGBM quantile (median)"},
                    ARTIFACTS / "model.joblib", compress=3)

    latest = X.iloc[[-1]]
    forecasts = {}
    for name in ("Random walk (zero)", "EWMA vol + Student-t", "LightGBM quantile"):
        if name in results:
            r = results[name]
            forecasts[name] = {
                "q10": float(r["q10"][-1]), "q50": float(r["q50"][-1]),
                "q90": float(r["q90"][-1]),
            }

    run.extra = {
        "horizon": h,
        "n_folds": n_folds,
        "price_history": price_history(df, tail=400),
        "equity_curves": {n: results[n]["backtest"]["curve"] for n in results},
        "backtests": {n: {k: v for k, v in results[n]["backtest"].items() if k != "curve"}
                      for n in results},
        "latest_forecasts": forecasts,
        "last_close": float(df["spy"].iloc[-1]),
        "last_date": str(df.index[-1].date()),
        "dataset": s,
        "lookahead_check": look,
        "significance_vs_random_walk": significance,
        "calibration": [
            {"model": m.name,
             "coverage_80": round(m.metrics["coverage_80"] * 100, 2),
             "interval_width_pct": round(m.metrics["interval_width"] * 100, 3),
             "pinball": m.metrics["pinball"]}
            for m in run.leaderboard()
        ],
    }
    run.set_split(SplitInfo(
        strategy=f"walk-forward, {n_folds} quarterly refits, purged",
        n_train=BURN_IN, n_test=int(len(X) - BURN_IN),
        embargo=f"{h - 1} days purged per fold",
        notes="Expanding window; every forecast is out-of-sample by construction.",
    ))

    report.phase(
        "deployment",
        objective="Serve a distribution, not a number.",
        narrative=(
            "`GET /api/forecast` returns the next-day quantile band from each "
            "model alongside the last close, so the band can be read in price "
            "terms. `GET /api/backtest` returns the walk-forward equity curve "
            "against buy-and-hold — displayed together deliberately, because "
            "a strategy curve shown alone is the oldest trick in quantitative "
            "marketing."
        ),
        code_refs=["server/main.py", "server/inference.py"],
        gate_notes="Forecasts, calibration and the buy-and-hold comparison all served.",
    ).add("Served model", best.name) \
     .add("Serving port", 8015) \
     .add("Endpoints", 11)

    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")

    print("[6/6] done.\n")
    print(run.leaderboard_markdown(("coverage_80", "hit_rate", "r2", "sharpe")))
    print(f"winner              : {best.name}")
    print(f"strategy Sharpe     : {bt_best['sharpe']:.2f}")
    print(f"buy-and-hold Sharpe : {bt_best['buy_hold_sharpe']:.2f}")
    print(f"critical findings   : {len(crit)}")
    print(f"runtime             : {run.duration_seconds}s")


if __name__ == "__main__":
    main()
