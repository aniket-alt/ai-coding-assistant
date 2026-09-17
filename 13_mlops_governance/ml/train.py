"""
Project 13 -- operate project 01's model through six months of real drift.

    python ml/train.py

Walks month by month from February to July 2024. For each month, every
retraining policy may use only months that have already finished; its model is
then scored on the new month. Drift is measured with the population stability
index, the served model is explained with TreeSHAP contributions, and its API is
load-tested on this machine.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage

from data import MONITOR_MONTHS, ROWS_PER_MONTH, SEED, TRAIN_MONTH, load_month, project01

features01 = project01("features")
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PSI_ALARM = 0.10


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------

def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population stability index with decile bins taken from the reference."""
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference.dropna(), edges)[0] / reference.notna().sum()
    cur = np.histogram(current.dropna(), edges)[0] / current.notna().sum()
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_categorical(reference: pd.Series, current: pd.Series, top: int = 30) -> float:
    """PSI over the most common categories, with everything else pooled."""
    keep = reference.value_counts().head(top).index
    ref = reference.where(reference.isin(keep), -1).value_counts(normalize=True)
    cur = current.where(current.isin(keep), -1).value_counts(normalize=True)
    cats = ref.index.union(cur.index)
    ref, cur = ref.reindex(cats).fillna(1e-4).clip(1e-4), cur.reindex(cats).fillna(1e-4).clip(1e-4)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


DRIFT_COLUMNS = {
    "duration_min": "numeric",      # the target, known once trips complete
    "trip_distance": "numeric",
    "hour": "numeric",
    "passenger_count": "numeric",
    "PULocationID": "categorical",
}


def drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    ref = reference.assign(hour=reference["tpep_pickup_datetime"].dt.hour)
    cur = current.assign(hour=current["tpep_pickup_datetime"].dt.hour)
    return {c: (psi_numeric if kind == "numeric" else psi_categorical)(ref[c], cur[c])
            for c, kind in DRIFT_COLUMNS.items()}


# --------------------------------------------------------------------------
# Modelling -- project 01's features and settings, unchanged
# --------------------------------------------------------------------------

def fit_lightgbm(train: pd.DataFrame, test: pd.DataFrame):
    Xtr, Xte, encoders = features01.build_features(train, test, train["duration_min"])
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=96, min_child_samples=40,
                              subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
                              random_state=SEED, n_jobs=-1, verbose=-1)
    model.fit(Xtr, train["duration_min"], categorical_feature=features01.CATEGORICAL)
    pred = np.clip(model.predict(Xte), 1.0, 180.0)
    return pred, {"model": model, "encoders": encoders, "feature_order": features01.FEATURE_ORDER,
                  "categorical": features01.CATEGORICAL,
                  "categories": {c: list(Xtr[c].cat.categories) for c in features01.CATEGORICAL}}, Xte


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = y - pred
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "within_5min": float(np.mean(np.abs(err) <= 5) * 100)}


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()

    report = CrispDmReport(
        project_slug="13_mlops_governance", project_title="MLOps & Governance",
        business_question=(
            "Once project 01's taxi model is in service, how fast does it go stale, "
            "and which retraining policy keeps it accurate at the lowest cost?"
        ),
    )
    report.phase(
        "business_understanding",
        objective="Decide how often to retrain, using evidence from real months rather than a rule of thumb.",
        narrative=(
            "A model is trained once and then used on a world that keeps moving. "
            "Two questions follow: how quickly does accuracy decay, and does "
            "retraining earn its cost? The usual answers — retrain monthly, or "
            "retrain when a drift metric crosses a threshold — are conventions, "
            "not findings.\n\n"
            "This project answers them with six real months. Project 01's model "
            "and features are taken unchanged, trained on January 2024, and then "
            "operated from February to July under several retraining policies. "
            "Each month, a policy may train only on months that have already "
            "finished; its model is then scored on the new month.\n\n"
            "Around that experiment sit the pieces an auditor asks for: a model "
            "registry, per-prediction explanations, a measured load test and a "
            "model card."
        ),
        code_refs=["ml/train.py:main"],
        gate_notes="Policies, drift threshold and evaluation fixed before any month was scored.",
    ).add("Training month", "January 2024").add("Monitored months", len(MONITOR_MONTHS)) \
     .add("Drift alarm", f"PSI > {PSI_ALARM}")

    print("[1/7] loading seven months of real trips ...")
    months, facts, provs = {}, [], []
    for m in [TRAIN_MONTH] + MONITOR_MONTHS:
        frame, f, p = load_month(m)
        months[m] = frame
        facts.append(f)
        provs.append(p)
    jan = months[TRAIN_MONTH]
    prov = provs[0].stamped(
        name="NYC TLC Yellow Taxi Trip Records — January to July 2024",
        n_rows=sum(p.n_rows for p in provs),
        notes="Seven monthly files; the SHA-256 and size are January's. Every month's receipt is in extra.",
        extra={"months": [{"month": f["month"], "rows": p.n_rows, "sha256": p.sha256} for f, p in zip(facts, provs)],
               "rows_modelled_per_month": ROWS_PER_MONTH},
    )
    prov.save(ARTIFACTS / "provenance.json")

    drift_rows = []
    for m in MONITOR_MONTHS:
        d = drift(jan, months[m])
        drift_rows.append({"month": m, **{k: round(v, 4) for k, v in d.items()}})
    first, last = facts[0], max(facts[1:], key=lambda f: f["median_duration"])
    report.phase(
        "data_understanding",
        objective="Measure how much the city actually changed between January and July.",
        narrative=(
            "| Month | Trips | Median duration | Median distance |\n|---|---:|---:|---:|\n"
            + "\n".join(f"| {f['month']} | {f['raw_trips']:,} | {f['median_duration']:.2f} min | "
                        f"{f['median_distance']:.2f} mi |" for f in facts)
            + f"\n\nMedian trip duration rises from {first['median_duration']:.2f} minutes in "
            f"January to {last['median_duration']:.2f} in {pd.Timestamp(last['month']):%B}, "
            f"while median distance barely moves. Trips got slower rather than longer: "
            f"the model's target shifted even where its inputs did not.\n\n"
            "Drift against January is measured with the population stability "
            "index. By convention, values below 0.1 are stable and above 0.25 are "
            "a major shift:\n\n"
            "| Month | " + " | ".join(DRIFT_COLUMNS) + " |\n|---|" + "---:|" * len(DRIFT_COLUMNS) + "\n"
            + "\n".join(f"| {r['month']} | " + " | ".join(f"{r[c]:.3f}" for c in DRIFT_COLUMNS) + " |"
                        for r in drift_rows)
        ),
        code_refs=["ml/data.py:load_month", "ml/train.py:psi_numeric"],
        gate_notes="Seven real months cleaned with project 01's rules; drift measured on every monitored month.",
    ).add("Trips across seven months", sum(f["raw_trips"] for f in facts)) \
     .add("Median duration, January", f"{first['median_duration']:.2f} min") \
     .add("Peak median duration", f"{last['median_duration']:.2f} min")

    print("[2/7] checks ...")
    leakage.assert_temporal_split(jan["tpep_pickup_datetime"],
                                  pd.concat([months[m]["tpep_pickup_datetime"] for m in MONITOR_MONTHS]))
    Xjan, _, _ = features01.build_features(jan, months[MONITOR_MONTHS[0]], jan["duration_min"])
    leakage.assert_no_target_leak(Xjan.drop(columns=features01.CATEGORICAL), jan["duration_min"])

    print("[3/7] walking six months under each retraining policy ...")
    order = [TRAIN_MONTH] + MONITOR_MONTHS
    registry, violations = [], []
    results: dict[str, dict] = {}

    def record_fit(policy: str, trained_on: list[str], evaluated: str, rows: int, secs: float, s: dict):
        if any(order.index(t) >= order.index(evaluated) for t in trained_on):
            violations.append(f"{policy}: trained on {trained_on} for {evaluated}")
        registry.append({"version": len(registry) + 1, "policy": policy, "trained_on": trained_on,
                         "rows": rows, "fit_seconds": round(secs, 1), "first_used_for": evaluated,
                         "rmse_on_first_month": round(s["rmse"], 3)})

    def run_policy(name: str, choose_training):
        monthly, fits, current = [], 0, None
        for i, m in enumerate(MONITOR_MONTHS):
            past = order[:order.index(m)]
            decision = choose_training(i, m, past, current)
            test = months[m]
            if decision is not None:                       # retrain
                train = pd.concat([months[t] for t in decision], ignore_index=True)
                t = time.perf_counter()
                pred, bundle, _ = fit_lightgbm(train, test)
                s = scores(test["duration_min"].to_numpy(), pred)
                record_fit(name, decision, m, len(train), time.perf_counter() - t, s)
                current, fits = (decision, bundle), fits + 1
            else:                                          # keep serving the existing model
                trained_on, bundle = current
                _, Xte, _ = features01.build_features(pd.concat([months[t] for t in trained_on], ignore_index=True),
                                                      test, pd.concat([months[t] for t in trained_on])["duration_min"])
                pred = np.clip(bundle["model"].predict(Xte), 1.0, 180.0)
                s = scores(test["duration_min"].to_numpy(), pred)
            monthly.append({"month": m, **{k: round(v, 4) for k, v in s.items()},
                            "trained_on": list(current[0]), "retrained": decision is not None})
        results[name] = {"monthly": monthly, "fits": fits, "bundle": current[1], "trained_on": current[0]}
        print(f"      {name:<34} mean RMSE {np.mean([r['rmse'] for r in monthly]):.3f}  retrains {fits}")

    run_policy("Never retrain", lambda i, m, past, cur: [TRAIN_MONTH] if cur is None else None)
    run_policy("Retrain monthly on last month", lambda i, m, past, cur: [past[-1]])
    run_policy("Retrain monthly on all history", lambda i, m, past, cur: list(past))

    alarms = []

    def on_alarm(i, m, past, cur):
        if cur is None:
            return [TRAIN_MONTH]
        reference = pd.concat([months[t] for t in cur[0]], ignore_index=True)
        latest = months[past[-1]]
        d = drift(reference, latest)
        worst = max(d, key=d.get)
        fired = d[worst] > PSI_ALARM
        alarms.append({"decided_for": m, "compared": past[-1], "worst_column": worst,
                       "psi": round(d[worst], 4), "retrained": fired})
        return list(past) if fired else None
    run_policy(f"Retrain on drift alarm (PSI > {PSI_ALARM})", on_alarm)

    # Baselines that need no model.
    def baseline(name: str, predict):
        monthly = []
        for m in MONITOR_MONTHS:
            past = pd.concat([months[t] for t in order[:order.index(m)]], ignore_index=True)
            s = scores(months[m]["duration_min"].to_numpy(), predict(past, months[m]))
            monthly.append({"month": m, **{k: round(v, 4) for k, v in s.items()},
                            "trained_on": order[:order.index(m)], "retrained": True})
        results[name] = {"monthly": monthly, "fits": 0, "bundle": None, "trained_on": []}
        print(f"      {name:<34} mean RMSE {np.mean([r['rmse'] for r in monthly]):.3f}")

    baseline("January median, never refreshed",
             lambda past, test: np.full(len(test), jan["duration_min"].median()))

    def lookup(past, test):
        table = past.groupby(["PULocationID", "DOLocationID"])["duration_min"].median()
        keys = pd.MultiIndex.from_frame(test[["PULocationID", "DOLocationID"]])
        return table.reindex(keys).fillna(past["duration_min"].median()).to_numpy()
    baseline("Route median lookup, refreshed monthly", lookup)

    leakage.require(
        "assert_policies_train_only_on_finished_months", not violations,
        f"{len(registry)} model fits checked; {len(violations)} used a month at or after the month they served.",
        fits=len(registry), violations=violations,
    )

    run = RunRecord(project_slug="13_mlops_governance", project_title="MLOps & Governance",
                    task="model monitoring and retraining policy", primary_metric="mean_rmse",
                    higher_is_better=False)
    run.set_provenance(prov)
    run.set_split(SplitInfo(strategy="month-by-month, each policy trained only on finished months",
                            n_train=ROWS_PER_MONTH, n_test=ROWS_PER_MONTH * len(MONITOR_MONTHS),
                            train_end=str(jan["tpep_pickup_datetime"].max()),
                            test_start=str(months[MONITOR_MONTHS[0]]["tpep_pickup_datetime"].min()),
                            notes=f"{ROWS_PER_MONTH:,} sampled trips per month."))
    for name, r in results.items():
        rmses = [x["rmse"] for x in r["monthly"]]
        run.add_model(ModelScore(
            name=name, family="baseline" if r["bundle"] is None else "retraining policy",
            metrics={"mean_rmse": float(np.mean(rmses)), "worst_month_rmse": float(np.max(rmses)),
                     "july_rmse": rmses[-1], "retrains": float(r["fits"]),
                     "mean_within_5min": float(np.mean([x["within_5min"] for x in r["monthly"]]))},
            params={"retrains": r["fits"]}))
    best = run.best
    never = next(m for m in run.models if m.name == "Never retrain").metrics
    alarm_name = f"Retrain on drift alarm (PSI > {PSI_ALARM})"
    alarm = next(m for m in run.models if m.name == alarm_name).metrics

    report.phase(
        "data_preparation",
        objective="Guarantee no policy ever trains on the month it is judged on.",
        narrative=(
            "Each policy chooses its training months at the start of a month, "
            "from months that have already finished. Every fit is written to the "
            f"model registry with its training window, and an assertion checks all "
            f"{len(registry)} fits: none may include the month it serves.\n\n"
            "Features and hyperparameters are project 01's, unchanged, so the "
            "only variable is what the model was trained on and when."
        ),
        code_refs=["ml/train.py:run_policy", "ml/train.py:record_fit"],
        gate_notes=f"{len(registry)} registry entries verified against their serving month.",
    ).add("Model fits", len(registry)).add("Policies", 4).add("Baselines", 2)

    report.phase(
        "modeling",
        objective="Compare four retraining policies and two model-free baselines month by month.",
        narrative=(
            "- **Never retrain** serves January's model all year.\n"
            "- **Retrain monthly on last month** follows the most recent month only.\n"
            "- **Retrain monthly on all history** accumulates every finished month.\n"
            f"- **Retrain on drift alarm** retrains on all history only when some "
            f"monitored column's PSI against the model's training data exceeds "
            f"{PSI_ALARM}.\n\n"
            "The two baselines refresh without learning: January's median "
            "duration, and the median duration of each pickup–dropoff pair over "
            "all finished months."
        ),
        code_refs=["ml/train.py:on_alarm", "ml/train.py:baseline"],
        gate_notes=f"{len(run.models)} strategies scored on {len(MONITOR_MONTHS)} months.",
    ).add("Best policy", best.name).add("Mean RMSE", f"{best.metrics['mean_rmse']:.3f} min") \
     .add("Retrains", int(best.metrics["retrains"]))

    print("[4/7] explanations ...")
    served = results["Retrain monthly on all history"]
    july = months[MONITOR_MONTHS[-1]]
    _, Xjul, _ = features01.build_features(
        pd.concat([months[t] for t in served["trained_on"]], ignore_index=True), july,
        pd.concat([months[t] for t in served["trained_on"]])["duration_min"])
    sample = Xjul.iloc[:: max(1, len(Xjul) // 5000)]
    contrib = served["bundle"]["model"].predict(sample, pred_contrib=True)
    names = list(sample.columns)
    global_importance = sorted(
        ({"feature": n, "mean_abs_minutes": round(float(np.abs(contrib[:, i]).mean()), 3)} for i, n in enumerate(names)),
        key=lambda r: -r["mean_abs_minutes"])

    print("[5/7] persisting the served model and load-testing its API ...")
    joblib.dump({**served["bundle"], "policy": "Retrain monthly on all history",
                 "trained_on": served["trained_on"]}, ARTIFACTS / "model.joblib", compress=3)
    load = subprocess.run([sys.executable, str(ROOT / "ml" / "load_test.py")], capture_output=True, text=True)
    if load.returncode != 0:
        raise RuntimeError(f"load test failed:\n{load.stdout}\n{load.stderr}")
    load_test = json.loads((ARTIFACTS / "load_test.json").read_text(encoding="utf-8"))

    print("[6/7] audit ...")
    own = leakage.audit_source(ROOT, temporal_modules=("train", "data"))
    leakage.save_report(own, ARTIFACTS / "audit.md", "MLOps & Governance")
    crit = [f for f in own if f.severity == "CRITICAL"]
    executed = leakage.ledger()
    run.audit = {"critical": len(crit), "warnings": sum(1 for f in own if f.severity == "WARNING"),
                 "total": len(own), "runtime_assertions": executed,
                 "assertions_passed": sum(1 for a in executed if a["passed"]), "assertions_run": len(executed)}

    fired = sum(a["retrained"] for a in alarms)
    peak = max(load_test["levels"], key=lambda r: r["throughput_rps"])
    report.phase(
        "evaluation",
        objective="Decide which policy to run, and whether the drift alarm earns its place.",
        narrative=(
            f"**{best.name}** keeps average error lowest, at {best.metrics['mean_rmse']:.3f} "
            f"minutes RMSE, against {never['mean_rmse']:.3f} for never retraining. Never "
            f"retraining is worst in {pd.Timestamp(max(results['Never retrain']['monthly'], key=lambda r: r['rmse'])['month']):%B}, "
            f"at {never['worst_month_rmse']:.3f}.\n\n"
            f"The drift alarm fired in {fired} of {len(alarms)} monthly checks and "
            f"retrained {int(alarm['retrains'])} times in total, reaching "
            f"{alarm['mean_rmse']:.3f} minutes. PSI thresholds are conventions, and "
            f"this is the evidence for how much this one is worth on this model.\n\n"
            f"Served, the model answers {peak['throughput_rps']:.0f} requests per second at "
            f"concurrency {peak['concurrency']} on this machine, with a 99th-percentile "
            f"latency of {peak['p99_ms']:.1f} ms. That is one uvicorn worker on a laptop, "
            f"measured, not a production claim."
        ),
        code_refs=["ml/train.py:main", "ml/load_test.py"],
        gate_passed=len(crit) == 0,
        gate_notes="No critical findings; every policy judged only on months it had not seen.",
    ).add("Best mean RMSE", f"{best.metrics['mean_rmse']:.3f} min") \
     .add("Never retrain", f"{never['mean_rmse']:.3f} min") \
     .add("Alarm retrains", int(alarm["retrains"])) \
     .add("Peak throughput", f"{peak['throughput_rps']:.0f} req/s")

    report.phase(
        "deployment",
        objective="Serve the model with its lineage, explanations, load profile and model card.",
        narrative=(
            "The served model is the one trained on all history through June. "
            "`POST /api/predict` returns the estimate together with each feature's "
            "TreeSHAP contribution in minutes. `GET /api/registry`, `/api/drift`, "
            "`/api/load-test` and `/api/model-card` publish lineage, monitoring, "
            "capacity and governance information from the same run."
        ),
        code_refs=["server/main.py", "server/inference.py"],
        gate_notes="Model, registry, drift history, load profile and model card persisted together.",
    ).add("Served policy", "Retrain monthly on all history").add("Serving port", 8013)

    run.extra = {
        "facts": facts, "drift_vs_january": drift_rows, "psi_alarm": PSI_ALARM, "alarm_checks": alarms,
        "policy_months": {n: r["monthly"] for n, r in results.items()}, "registry": registry,
        "global_importance": global_importance, "load_test": load_test,
        "served": {"policy": "Retrain monthly on all history", "trained_on": served["trained_on"]},
    }
    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")

    print("[7/7] done.\n")
    print(run.leaderboard_markdown(("worst_month_rmse", "july_rmse", "retrains")))
    print("alarms:", alarms)
    print(f"critical: {len(crit)}  runtime: {run.duration_seconds}s")


if __name__ == "__main__":
    main()
