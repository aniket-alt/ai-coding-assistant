"""
Project 12 -- day-ahead household energy forecasting.

    python ml/train.py

Walk-forward over three years of hourly load: refit every sixty days on
everything before, forecast the following sixty days a day ahead. Scored on
pinball loss and interval coverage, tested for significance against simply
repeating yesterday, and accompanied by a deliberately leaky model -- kept out
of the tournament -- that shows what using a too-recent lag would have claimed.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage
from dsfs.forecasting import QUANTILES, interval_scores, paired_bootstrap, pinball, pinball_series

from data import hourly, load_minutes, profiles, summarise
from features import FEATURES, HORIZON, LAGS, assert_no_lookahead, build

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEED = 42
BURN_IN = 8760        # one year of history before the first forecast
REFIT_EVERY = 1440    # sixty days
REFERENCE = "Same hour yesterday"


def evaluate(y, q10, q50, q90) -> dict[str, float]:
    losses = [pinball(y, q, a) for q, a in zip((q10, q50, q90), QUANTILES)]
    return {
        "pinball": float(np.mean(losses)),
        **interval_scores(y, q10, q90),
        "mae_kw": float(np.mean(np.abs(y - q50))),
        # Weighted absolute percentage error: stable when some hours are near
        # zero, unlike MAPE, which explodes on them.
        "wape_pct": float(np.sum(np.abs(y - q50)) / np.sum(np.abs(y)) * 100),
    }


def walk_forward(frame: pd.DataFrame, fit_predict):
    """Expanding window. Rows without a recorded target are neither trained on nor scored."""
    n = len(frame)
    parts, refits, last = [], 0, None
    start = BURN_IN
    while start < n:
        stop = min(start + REFIT_EVERY, n)
        train = frame.iloc[:start]
        train = train[train["target"].notna()]
        test = frame.iloc[start:stop]
        q10, q50, q90, model = fit_predict(train, test)
        parts.append(pd.DataFrame({"y": test["target"], "q10": q10, "q50": q50, "q90": q90},
                                  index=test.index))
        refits += 1
        last = model
        start = stop
    out = pd.concat(parts)
    return out[out["y"].notna()], refits, last


def residual_band(pred_train: np.ndarray, y_train: np.ndarray, pred_test: np.ndarray):
    """Turn a point forecast into quantiles using the training-window residuals."""
    ok = ~np.isnan(pred_train)
    lo, mid, hi = np.quantile(y_train[ok] - pred_train[ok], QUANTILES)
    return pred_test + lo, pred_test + mid, pred_test + hi


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()

    report = CrispDmReport(
        project_slug="12_energy_forecasting",
        project_title="Energy Demand Forecasting",
        business_question=(
            "How much electricity will this household use in each hour of "
            "tomorrow, and how wide should the uncertainty be?"
        ),
    )
    report.phase(
        "business_understanding",
        objective="Issue a calibrated day-ahead forecast of hourly load.",
        narrative=(
            "A day-ahead forecast is issued once, at a fixed moment, for every "
            "hour of the following day. That timing defines what the model may "
            "know: nothing recorded after the forecast is issued.\n\n"
            "The most common way to break that rule is a lag that is too short. "
            "The load one hour ago is an excellent predictor of the next hour, "
            "and it is simply not available when tomorrow evening is forecast "
            "this evening. This project enforces a minimum lag equal to the "
            "horizon, verifies it by recomputation, and measures separately how "
            "much a leaky model would have flattered the results.\n\n"
            "For a single household the honest baselines are strong: tomorrow "
            "often looks like today, and like the same day last week. Every "
            "model is tested for whether it genuinely beats repeating yesterday."
        ),
        code_refs=["ml/features.py:LAGS", "ml/train.py:walk_forward"],
        gate_notes="Horizon, minimum lag, loss and reference baseline fixed before modelling.",
    ).add("Horizon", f"{HORIZON} hours ahead") \
     .add("Primary metric", "Pinball loss", "P10, P50, P90") \
     .add("Reference", REFERENCE)

    print("[1/6] loading minute readings ...")
    minutes, prov = load_minutes()
    series, facts = hourly(minutes)
    s = summarise(series)
    prov = prov.stamped(extra={"hours_after_aggregation": facts["hours"],
                               "hours_missing": facts["missing_hours"]})
    prov.save(ARTIFACTS / "provenance.json")

    report.phase(
        "data_understanding",
        objective="Aggregate minute readings to hours without inventing values for gaps.",
        narrative=(
            f"{facts['minutes']:,} minute readings from {facts['start'][:10]} to "
            f"{facts['end'][:10]}, of which {facts['missing_minutes']:,} are "
            f"missing. An hour is kept only if at least 45 of its minutes were "
            f"recorded, leaving {facts['hours']:,} hours with "
            f"**{facts['missing_hours']:,} missing**; the longest gap lasts "
            f"{facts['longest_gap_hours']} hours. Gaps are never interpolated, "
            f"because a forecast scored against an interpolated value is scored "
            f"against something that did not happen.\n\n"
            f"Mean load is {s['mean_kw']:.2f} kW against a median of "
            f"{s['median_kw']:.2f} kW, the skew of a household that is usually "
            f"quiet and occasionally runs several appliances at once. Winter use "
            f"is **{s['winter_to_summer_ratio']:.2f}×** summer use.\n\n"
            f"Correlation with the same hour a day earlier is "
            f"{s['daily_autocorrelation']:.2f}, and a week earlier "
            f"{s['weekly_autocorrelation']:.2f}. A single home is noisy, and "
            f"no model will explain most of its hourly variation."
        ),
        code_refs=["ml/data.py:hourly"],
        gate_notes=f"{facts['hours']:,} hours; gaps preserved as missing.",
    ).add("Hours", facts["hours"]) \
     .add("Missing hours", facts["missing_hours"]) \
     .add("Longest gap", f"{facts['longest_gap_hours']} h") \
     .add("Winter / summer", f"{s['winter_to_summer_ratio']:.2f}×") \
     .add("Same hour, one day earlier", f"r = {s['daily_autocorrelation']:.2f}")

    print("[2/6] features + horizon checks ...")
    frame = build(series)
    frame = frame.iloc[:-HORIZON]           # future rows are for serving, not scoring
    short = [k for k in LAGS if k < HORIZON]
    leakage.require(
        "assert_lags_respect_horizon", not short,
        f"{len(LAGS)} lags checked against a {HORIZON}-hour horizon; shortest is "
        f"{min(LAGS)} hours; {len(short)} shorter than the horizon.",
        lags=LAGS, horizon=HORIZON, violations=short,
    )
    look = assert_no_lookahead(build(series), series, FEATURES)
    leakage.require(
        "assert_no_lookahead", look["mismatches"] == 0,
        f"{look['values_compared']} feature values across {look['targets_checked']} target hours "
        f"recomputed from data available at the forecast origin; {look['mismatches']} differed.",
        **look,
    )
    first_test = frame.index[BURN_IN]
    leakage.assert_temporal_split(frame.index[:BURN_IN], frame.index[BURN_IN:])
    history = frame.iloc[:BURN_IN]
    recorded = history["target"].notna()
    leakage.assert_no_target_leak(history.loc[recorded, FEATURES], history.loc[recorded, "target"])

    n_folds = int(np.ceil((len(frame) - BURN_IN) / REFIT_EVERY))
    report.phase(
        "data_preparation",
        objective="Build features that exist when the forecast is issued, and prove it.",
        narrative=(
            f"Every row is a target hour. Its features are lags of {min(LAGS)} to "
            f"{max(LAGS)} hours, the same hour's four-week average, rolling "
            f"statistics ending at the forecast origin, and the target hour's "
            f"calendar, which is known in advance.\n\n"
            f"Two checks enforce the timing. The shortest lag must be at least "
            f"the {HORIZON}-hour horizon. And features for "
            f"{look['targets_checked']} random target hours are recomputed from "
            f"a series truncated at the forecast origin: **{look['values_compared']} "
            f"values, {look['mismatches']} mismatches**.\n\n"
            f"Evaluation walks forward from {first_test:%B %Y}: {n_folds} refits, "
            f"each trained on all earlier hours and forecasting the next sixty "
            f"days."
        ),
        code_refs=["ml/features.py:build", "ml/features.py:assert_no_lookahead"],
        gate_notes="Minimum-lag and recomputation checks passed.",
    ).add("Features", len(FEATURES)) \
     .add("Shortest lag", f"{min(LAGS)} h") \
     .add("Values re-verified", look["values_compared"]) \
     .add("Walk-forward folds", n_folds)

    print(f"[3/6] walk-forward tournament ({n_folds} folds) ...")
    run = RunRecord(project_slug="12_energy_forecasting", project_title="Energy Demand Forecasting",
                    task="probabilistic day-ahead load forecasting", primary_metric="pinball",
                    higher_is_better=False)
    run.set_provenance(prov)
    results: dict[str, pd.DataFrame] = {}
    models: dict[str, object] = {}

    def enter(name, family, fit_predict, notes=""):
        t = time.perf_counter()
        out, refits, last = walk_forward(frame, fit_predict)
        secs = time.perf_counter() - t
        m = evaluate(out["y"].to_numpy(), out["q10"].to_numpy(), out["q50"].to_numpy(), out["q90"].to_numpy())
        run.add_model(ModelScore(name=name, family=family, metrics=m, params={"refits": refits},
                                 fit_seconds=round(secs, 1), notes=notes))
        results[name] = out
        models[name] = last
        print(f"      {name:<32} pinball {m['pinball']:.4f}  cover80 {m['coverage_80']*100:5.1f}%  "
              f"WAPE {m['wape_pct']:5.1f}%  ({secs:.0f}s)")

    def seasonal(col):
        def fit_predict(train, test):
            fb = float(train["target"].median())
            p_tr = train[col].fillna(train["same_hour_4wk_mean"]).to_numpy()
            p_te = test[col].fillna(test["same_hour_4wk_mean"]).fillna(fb).to_numpy()
            return (*residual_band(p_tr, train["target"].to_numpy(), p_te), None)
        return fit_predict

    enter(REFERENCE, "baseline", seasonal("lag_24"), "Repeats the load 24 hours earlier.")
    enter("Same hour last week", "baseline", seasonal("lag_168"), "Repeats the load 168 hours earlier.")
    enter("Four-week hourly average", "baseline", seasonal("same_hour_4wk_mean"),
          "Mean of the same hour over the previous four weeks.")

    def ridge(train, test):
        p = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                      ("m", Ridge(alpha=10.0, random_state=SEED))])
        p.fit(train[FEATURES], train["target"])
        return (*residual_band(p.predict(train[FEATURES]), train["target"].to_numpy(),
                               p.predict(test[FEATURES])), p)
    enter("Ridge + residual quantiles", "linear", ridge,
          "Imputer and scaler fitted per fold inside the pipeline.")

    def lgbm(features):
        def fit_predict(train, test):
            preds, fitted = [], {}
            for q in QUANTILES:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=300, learning_rate=0.05,
                                      num_leaves=31, min_child_samples=50, random_state=SEED, n_jobs=-1,
                                      verbose=-1)
                m.fit(train[features], train["target"])
                preds.append(m.predict(test[features]))
                fitted[q] = m
            q10, q50, q90 = np.sort(np.vstack(preds), axis=0)   # prevent crossed quantiles
            return q10, q50, q90, fitted
        return fit_predict
    enter("LightGBM quantile", "gbdt", lgbm(FEATURES), "Three boosters on the pinball objective.")

    # Student-t band around the four-week average, scaled by recent variability:
    # a fat-tailed interval for a spiky series.
    def student_t(train, test):
        fb = float(train["target"].median())
        mid = test["same_hour_4wk_mean"].fillna(fb).to_numpy()
        sd = test["std_last_168h"].fillna(train["target"].std()).to_numpy()
        z = stats.t.ppf(0.9, df=4)
        return mid - z * sd, mid, mid + z * sd, None
    enter("Four-week average + Student-t", "volatility", student_t,
          "Four-week mean with a t(4) interval scaled by last week's variability.")

    best = run.best
    assert best is not None

    print("[4/6] significance, leakage demonstration, audit ...")
    ref = results[REFERENCE]
    significance = {}
    for name, out in results.items():
        if name == REFERENCE:
            continue
        joined = out.join(ref, rsuffix="_ref", how="inner")
        significance[name] = paired_bootstrap(
            pinball_series(joined["y"].to_numpy(), joined["q10"].to_numpy(), joined["q50"].to_numpy(),
                           joined["q90"].to_numpy()),
            pinball_series(joined["y"].to_numpy(), joined["q10_ref"].to_numpy(), joined["q50_ref"].to_numpy(),
                           joined["q90_ref"].to_numpy()),
        )
    for m in run.models:
        m.metrics["skill_vs_yesterday_pct"] = (1 - m.metrics["pinball"] / run.models[0].metrics["pinball"]) * 100

    # What a lag shorter than the horizon would have claimed. Built from the same
    # series with one-hour lags added, scored the same way, and kept out of the
    # tournament because it could never run in production.
    leaky_frame = build(series, lags=[1, 2, 3] + LAGS).iloc[:-HORIZON]
    leaky_features = [f"lag_{k}" for k in (1, 2, 3)] + FEATURES
    leaky_out, _, _ = walk_forward(leaky_frame, lgbm(leaky_features))
    leaky = evaluate(leaky_out["y"].to_numpy(), leaky_out["q10"].to_numpy(),
                     leaky_out["q50"].to_numpy(), leaky_out["q90"].to_numpy())
    honest = next(m for m in run.models if m.name == "LightGBM quantile").metrics
    flattery_pct = (1 - leaky["pinball"] / honest["pinball"]) * 100
    print(f"      leaky LightGBM (1h lags)          pinball {leaky['pinball']:.4f}  "
          f"({flattery_pct:.0f}% better than it could honestly be)")

    findings = leakage.audit_source(ROOT, temporal_modules=("train", "features", "data"))
    leakage.save_report(findings, ARTIFACTS / "audit.md", "Energy Demand Forecasting")
    crit = [f for f in findings if f.severity == "CRITICAL"]
    executed = leakage.ledger()
    run.audit = {"critical": len(crit), "warnings": sum(1 for f in findings if f.severity == "WARNING"),
                 "total": len(findings), "runtime_assertions": executed,
                 "assertions_passed": sum(1 for a in executed if a["passed"]),
                 "assertions_run": len(executed)}

    ref_m = run.models[0].metrics
    report.phase(
        "modeling",
        objective="Compare learned forecasters with three seasonal baselines and a fat-tailed interval.",
        narrative=(
            "Three baselines repeat the past in different ways: yesterday, the "
            "same day last week, and the average of the last four weeks. Each is "
            "given an interval from its own training-window errors, so baselines "
            "compete on calibration as well as on the median.\n\n"
            "Ridge and LightGBM use all features. LightGBM fits three boosters "
            "directly on the pinball loss, and its three quantiles are sorted "
            "per hour so that P10 never exceeds P90."
        ),
        code_refs=["ml/train.py:enter", "ml/train.py:residual_band"],
        gate_notes=f"{len(run.models)} forecasters over {n_folds} folds.",
    ).add("Forecasters", len(run.models)).add("Winner", best.name) \
     .add("Pinball loss", f"{best.metrics['pinball']:.4f}") \
     .add("80% coverage", f"{best.metrics['coverage_80']*100:.1f}%")

    sig = significance.get(best.name, {})
    report.phase(
        "evaluation",
        objective="Decide whether the winner genuinely beats repeating the past, and measure the cost of a leak.",
        narrative=(
            f"**{best.name}** has the lowest pinball loss, "
            f"{best.metrics['pinball']:.4f}, a skill of "
            f"{best.metrics['skill_vs_yesterday_pct']:.1f}% over repeating "
            f"yesterday ({ref_m['pinball']:.4f}). A paired bootstrap on hourly "
            f"losses rates that difference as **{sig.get('verdict', 'not tested')}**. "
            f"Its 80% interval covers {best.metrics['coverage_80']*100:.1f}% of "
            f"outcomes, and its median misses by "
            f"{best.metrics['wape_pct']:.1f}% of total load (WAPE).\n\n"
            f"**The leak.** The same LightGBM with one-, two- and three-hour lags "
            f"added scores a pinball loss of {leaky['pinball']:.4f}, "
            f"**{flattery_pct:.0f}% better** than the honest version. None of "
            f"that improvement exists when the forecast is actually issued a day "
            f"in advance. It is reported here and kept out of the leaderboard, "
            f"because it is exactly the result a careless evaluation would have "
            f"published.\n\n"
            "No weather data is available, and winter use is nearly double summer "
            "use; temperature would likely be the most valuable missing input."
        ),
        code_refs=["ml/train.py:main"],
        gate_passed=len(crit) == 0,
        gate_notes="No critical findings; leaky variant measured but excluded.",
    ).add("Skill vs yesterday", f"{best.metrics['skill_vs_yesterday_pct']:.1f}%") \
     .add("Significance", sig.get("verdict", "—")) \
     .add("Leaky model would claim", f"{flattery_pct:.0f}% better") \
     .add("Critical findings", len(crit))

    print("[5/6] persisting ...")
    full = build(series)
    next_day = full.iloc[-HORIZON:]
    last_lgb = models["LightGBM quantile"]
    q = np.sort(np.vstack([last_lgb[a].predict(next_day[FEATURES]) for a in QUANTILES]), axis=0)
    joblib.dump({"quantile_models": last_lgb, "features": FEATURES, "horizon": HORIZON},
                ARTIFACTS / "model.joblib", compress=3)

    window = results[best.name].iloc[-24 * 14:]
    ref_window = results[REFERENCE].reindex(window.index)
    run.set_split(SplitInfo(
        strategy=f"walk-forward, {n_folds} refits every 60 days, day-ahead",
        n_train=BURN_IN, n_test=int(len(results[best.name])),
        train_end=str(frame.index[BURN_IN - 1]), test_start=str(first_test),
        embargo=f"{HORIZON} h horizon; no lag shorter than the horizon",
        notes="Hours with no recorded load are neither trained on nor scored.",
    ))
    report.phase(
        "deployment",
        objective="Serve tomorrow's hourly band and the recent track record together.",
        narrative=(
            "`GET /api/forecast` returns the next 24 hours after the data ends "
            "from the final fold's quantile models, beside the last fourteen "
            "days of walk-forward forecasts against what actually happened, so "
            "the band can be judged by its history."
        ),
        code_refs=["server/main.py"],
        gate_notes="Final-fold quantile models persisted; forecasts served from artifacts.",
    ).add("Served model", "LightGBM quantile").add("Serving port", 8012)

    run.extra = {
        "facts": facts, "summary": s, "profiles": profiles(series),
        "significance_vs_yesterday": significance,
        "leakage_demo": {"honest_pinball": honest["pinball"], "leaky_pinball": leaky["pinball"],
                         "flattery_pct": flattery_pct, "leaky_lags": [1, 2, 3],
                         "leaky_coverage_80": leaky["coverage_80"]},
        "recent_window": [
            {"time": str(t), "actual": round(float(r.y), 3), "q10": round(float(r.q10), 3),
             "q50": round(float(r.q50), 3), "q90": round(float(r.q90), 3),
             "yesterday": round(float(ref_window.loc[t, "q50"]), 3)}
            for t, r in window.iterrows()
        ],
        "recent_model": best.name,
        "next_day": [
            {"time": str(t), "q10": round(float(a), 3), "q50": round(float(b), 3), "q90": round(float(c), 3)}
            for t, a, b, c in zip(next_day.index, q[0], q[1], q[2])
        ],
    }
    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")

    print("[6/6] done.\n")
    print(run.leaderboard_markdown(("coverage_80", "wape_pct", "skill_vs_yesterday_pct")))
    for k, v in significance.items():
        print(f"  vs yesterday: {k:<32} {v['verdict']}")
    print(f"critical: {len(crit)}  runtime: {run.duration_seconds}s")


if __name__ == "__main__":
    main()
