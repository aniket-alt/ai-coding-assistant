"""
Project 07 -- does automated stacking beat one well-tuned model?

    python ml/train.py
    (run ml/autogluon_run.py in .venv-automl first to include AutoGluon)

Every contestant trains on calls made before 2010 and is scored on every call
made in 2010, the year it would have been deployed into. Each is also asked
what score it expected, from its own validation, so the optimism of that
estimate can be measured against what actually happened.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage

from data import CATEGORICAL, FEATURES, LEAK, NUMERIC, TARGET, TEST_YEAR, load, split
from features import preprocessor

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEED = 42
FOLDS = 5
TUNING_TRIALS = 40
CARUANA_ROUNDS = 30
BOOTSTRAPS = 2000


def lgbm(**params) -> lgb.LGBMClassifier:
    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=8, verbose=-1)
    return lgb.LGBMClassifier(**{**base, **params})


def level_one() -> dict[str, Pipeline]:
    """Five deliberately different learners, so their errors are not all the same errors."""
    return {
        "Logistic regression": Pipeline([("pre", preprocessor(scale=True)),
                                         ("m", LogisticRegression(max_iter=3000, C=0.5))]),
        "LightGBM": Pipeline([("pre", preprocessor(scale=False)), ("m", lgbm())]),
        "XGBoost": Pipeline([("pre", preprocessor(scale=False)),
                             ("m", xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                                     subsample=0.8, colsample_bytree=0.8,
                                                     random_state=SEED, n_jobs=8, verbosity=0))]),
        "Extra trees": Pipeline([("pre", preprocessor(scale=False)),
                                 ("m", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5,
                                                            random_state=SEED, n_jobs=8))]),
        "k-nearest neighbours": Pipeline([("pre", preprocessor(scale=True)),
                                          ("m", KNeighborsClassifier(n_neighbors=50, n_jobs=8))]),
    }


def bag(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame,
        folds: StratifiedKFold) -> tuple[np.ndarray, np.ndarray, list[Pipeline], float]:
    """
    Out-of-fold predictions for the training rows, and the average of the fold
    models' predictions for the test rows -- the bagging AutoGluon itself uses.
    """
    oof = np.zeros(len(X))
    test = np.zeros(len(X_test))
    models = []
    t = time.perf_counter()
    for tr, va in folds.split(X, y):
        m = clone(pipe).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test += m.predict_proba(X_test)[:, 1] / folds.get_n_splits()
        models.append(m)
    return oof, test, models, time.perf_counter() - t


def caruana(oof: dict[str, np.ndarray], y: np.ndarray, rounds: int = CARUANA_ROUNDS) -> tuple[dict[str, float], list[dict]]:
    """
    Greedy forward ensemble selection with replacement (Caruana et al., 2004).

    Each round adds whichever model most improves the out-of-fold AUC of the
    running average. Picking a model again raises its weight. Ties go to the
    first name in sorted order, so the result never depends on dict ordering.
    """
    names = sorted(oof)
    chosen: list[str] = []
    total = np.zeros(len(y))
    trajectory = []
    for r in range(rounds):
        scores = {n: roc_auc_score(y, (total + oof[n]) / (len(chosen) + 1)) for n in names}
        pick = max(names, key=lambda n: (round(scores[n], 12), -names.index(n)))
        chosen.append(pick)
        total += oof[pick]
        trajectory.append({"round": r + 1, "added": pick, "oof_auc": float(scores[pick])})
    weights = {n: chosen.count(n) / rounds for n in names if n in chosen}
    return weights, trajectory


def blend(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(w * preds[n] for n, w in weights.items())


def bootstrap_auc(y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = BOOTSTRAPS) -> dict:
    """Paired bootstrap of the AUC difference A - B over the same resampled test calls."""
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(roc_auc_score(y, a) - roc_auc_score(y, b)), "ci_low": float(lo), "ci_high": float(hi),
            "verdict": "better" if lo > 0 else "worse" if hi < 0 else "inconclusive"}


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()
    report = CrispDmReport(
        project_slug="07_automl_stacking", project_title="AutoML Stacking",
        business_question=(
            "Which clients should a bank call about a term deposit, and does an automated "
            "stacked ensemble choose them better than one well-tuned model?"
        ),
    )
    report.phase(
        "business_understanding",
        objective="Rank clients by their chance of subscribing, using only what is known before the call.",
        narrative=(
            "A call centre has more clients than time. A model that ranks clients by their "
            "chance of subscribing lets the best prospects be called first, so ranking "
            "quality (ROC AUC) is the measure, with average precision beside it.\n\n"
            "The second question is about tooling. AutoML systems such as AutoGluon win "
            "benchmarks by stacking many models. This project checks whether that holds on "
            "a deployment-shaped test, against an ensemble built by hand and a single tuned "
            "LightGBM, and whether each method's own validation told the truth about it."
        ),
        code_refs=["ml/train.py:main"],
        gate_notes="Target, ranking metric and a before-the-call feature rule fixed before modelling.",
    ).add("Target", "client subscribes to a term deposit").add("Primary metric", "ROC AUC") \
     .add("Excluded by rule", LEAK)

    print("[1/9] loading bank marketing calls ...")
    df, prov = load()
    prov.save(ARTIFACTS / "provenance.json")
    train, test = split(df)
    X, Xt = train[FEATURES], test[FEATURES]
    y, yt = train[TARGET].to_numpy(), test[TARGET].to_numpy()

    by_period = (df.groupby("period", sort=False)
                   .agg(calls=(TARGET, "size"), rate=(TARGET, "mean"), euribor=("euribor3m", "mean"))
                   .reset_index())
    shift = {
        "periods": [{"period": r.period, "calls": int(r.calls), "subscribed_pct": round(float(r.rate) * 100, 1),
                     "euribor3m": round(float(r.euribor), 3), "is_test": r.period >= f"{TEST_YEAR}"}
                    for r in by_period.itertuples()],
        "train_calls": int(len(train)), "test_calls": int(len(test)),
        "train_rate_pct": float(y.mean() * 100), "test_rate_pct": float(yt.mean() * 100),
        "train_first": str(train["period"].iloc[0]), "train_last": str(train["period"].iloc[-1]),
        "test_first": str(test["period"].iloc[0]), "test_last": str(test["period"].iloc[-1]),
        "pdays_never_pct": float((df["pdays"] == 999).mean() * 100),
        "unknown_default_pct": float((df["default"] == "unknown").mean() * 100),
        "duplicate_rows": int(df.drop(columns=["order", "year", "period"]).duplicated().sum()),
    }
    report.phase(
        "data_understanding",
        objective="Find what the data's time order and documentation imply for evaluation.",
        narrative=(
            f"{len(df):,} calls from {shift['train_first']} to {shift['test_last']}. The file "
            f"is in date order but records only the month, so the year is reconstructed from "
            f"the order.\n\n"
            f"**The success rate moves with the financial crisis.** "
            f"{shift['train_rate_pct']:.1f}% of calls before {TEST_YEAR} led to a subscription, "
            f"against **{shift['test_rate_pct']:.1f}%** in {TEST_YEAR}, while the three-month "
            f"Euribor rate collapsed. A random split would mix these regimes and hide the "
            f"change a deployed model actually faces.\n\n"
            f"**`duration` is a documented leak.** It is the length of the call, known only "
            f"after the client has answered. {shift['pdays_never_pct']:.1f}% of calls have "
            f"`pdays` = 999, meaning never contacted before."
        ),
        code_refs=["ml/data.py:load"],
        gate_notes="Time order, regime change and the leaking column documented.",
    ).add("Calls", len(df)).add("Success rate before 2010", f"{shift['train_rate_pct']:.1f}%") \
     .add(f"Success rate in {TEST_YEAR}", f"{shift['test_rate_pct']:.1f}%")

    print("[2/9] split and checks ...")
    leakage.assert_temporal_split(train["period"], test["period"])
    leakage.require(
        "assert_duration_excluded", LEAK not in FEATURES,
        f"{len(FEATURES)} model inputs; `{LEAK}` (known only after the call) "
        f"{'absent' if LEAK not in FEATURES else 'PRESENT'}.",
        features=len(FEATURES), leak_column=LEAK,
    )
    check = preprocessor(scale=False).fit(X)
    leakage.assert_no_target_leak(pd.DataFrame(check.transform(X), columns=check.get_feature_names_out()), y)
    report.phase(
        "data_preparation",
        objective="Split by time and keep every transformation inside the model pipelines.",
        narrative=(
            f"Training uses the {len(train):,} calls from before {TEST_YEAR}; the test set is "
            f"all {len(test):,} calls made in {TEST_YEAR}. `pdays` = 999 becomes an explicit "
            f"'previously contacted' indicator, because as a number it would read as a very "
            f"long gap. Scaling and one-hot encoding are fitted inside each pipeline on "
            f"training rows only, and every contestant, AutoGluon included, sees the same "
            f"{len(FEATURES)} columns."
        ),
        code_refs=["ml/features.py:preprocessor", "ml/data.py:split"],
        gate_notes="Chronological split, leak exclusion and target-correlation checks executed.",
    ).add("Training calls", len(train)).add("Test calls", len(test))

    run = RunRecord(project_slug="07_automl_stacking", project_title="AutoML Stacking",
                    task="binary classification, ranking clients to call", primary_metric="roc_auc",
                    higher_is_better=True)
    run.set_provenance(prov)
    run.set_split(SplitInfo(strategy=f"chronological: calls before {TEST_YEAR} train, calls in {TEST_YEAR} test",
                            n_train=len(train), n_test=len(test),
                            train_end=shift["train_last"], test_start=shift["test_first"]))
    test_preds: dict[str, np.ndarray] = {}

    def enter(name: str, family: str, proba: np.ndarray, believed: float | None, fit_s: float,
              predict_ms_per_1k: float, notes: str = "", **params) -> None:
        m = {"roc_auc": float(roc_auc_score(yt, proba)), "average_precision": float(average_precision_score(yt, proba))}
        if believed is not None:
            m["believed_auc"] = float(believed)
            m["optimism"] = float(believed - m["roc_auc"])
        run.add_model(ModelScore(name=name, family=family, metrics=m, fit_seconds=round(fit_s, 1),
                                 predict_ms_per_1k=round(predict_ms_per_1k, 2), notes=notes, params=params))
        test_preds[name] = proba
        print(f"      {name:<34} AUC {m['roc_auc']:.4f}  AP {m['average_precision']:.4f}  "
              f"believed {believed if believed is None else round(believed, 4)}  fit {fit_s:.0f}s")

    def timed_predict(models: list[Pipeline]) -> tuple[np.ndarray, float]:
        t = time.perf_counter()
        p = np.mean([m.predict_proba(Xt)[:, 1] for m in models], axis=0)
        return p, (time.perf_counter() - t) / len(Xt) * 1e6

    print("[3/9] baselines ...")
    rate_by_outcome = train.groupby("poutcome")[TARGET].mean()
    enter("Previous-campaign outcome rule", "baseline",
          test["poutcome"].map(rate_by_outcome).fillna(y.mean()).to_numpy(), None, 0.0, 0.0,
          "Scores each client by the pre-2010 success rate for their previous campaign's outcome.")
    enter("Constant score", "baseline", np.full(len(test), y.mean()), 0.5, 0.0, 0.0,
          "The same score for everyone; ranks no better than chance.")

    print("[4/9] level-one models, bagged over five folds ...")
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    oof, bagged, fold_models, fit_time = {}, {}, {}, {}
    for name, pipe in level_one().items():
        oof[name], bagged[name], fold_models[name], fit_time[name] = bag(pipe, X, y, Xt, folds)
        print(f"      {name:<24} OOF AUC {roc_auc_score(y, oof[name]):.4f}  test AUC {roc_auc_score(yt, bagged[name]):.4f}")
    for name in ("Logistic regression", "LightGBM"):
        _, us = timed_predict(fold_models[name])
        enter(name if name != "LightGBM" else "LightGBM, defaults", "linear" if name.startswith("Logistic") else "gbdt",
              bagged[name], roc_auc_score(y, oof[name]), fit_time[name], us,
              "Average of five fold models; believed score is its out-of-fold AUC.")

    print(f"[5/9] tuning LightGBM with Optuna, {TUNING_TRIALS} trials ...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xp = preprocessor(scale=False).fit(X)
    Xtr_p = pd.DataFrame(Xp.transform(X), columns=Xp.get_feature_names_out())

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
        # Preprocessing here is one-hot and passthrough only, so fitting it on all
        # training rows before the folds learns nothing about any fold's target.
        return float(cross_val_score(lgbm(**params), Xtr_p, y, cv=folds, scoring="roc_auc").mean())

    t = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=TUNING_TRIALS)
    best_params = {**study.best_params, "subsample_freq": 1}
    tuned_oof, tuned_test, tuned_models, tuned_fit = bag(
        Pipeline([("pre", preprocessor(scale=False)), ("m", lgbm(**best_params))]), X, y, Xt, folds)
    tuning_seconds = time.perf_counter() - t
    _, us = timed_predict(tuned_models)
    enter("LightGBM, tuned", "gbdt", tuned_test, study.best_value, tuning_seconds, us,
          f"Optuna TPE, {TUNING_TRIALS} trials of five-fold CV on pre-{TEST_YEAR} calls.",
          **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in best_params.items()})
    tuning_trace = [{"trial": tr.number + 1, "cv_auc": float(tr.value),
                     "best_so_far": float(max(x.value for x in study.trials[: tr.number + 1]))}
                    for tr in study.trials]

    print("[6/9] hand-built stack: Caruana selection and a logistic stacker ...")
    t = time.perf_counter()
    weights, trajectory = caruana(oof, y)
    stack_oof = blend(oof, weights)
    select_seconds = time.perf_counter() - t
    stack_fit = sum(fit_time.values()) + select_seconds
    stack_test = blend(bagged, weights)
    member_us = {n: timed_predict(fold_models[n])[1] for n in sorted(oof)}
    stack_us = sum(member_us[n] for n in weights)
    enter("Hand-built stack, Caruana selection", "stacked ensemble", stack_test, roc_auc_score(y, stack_oof),
          stack_fit, stack_us,
          f"Greedy forward selection with replacement over five bagged models, {CARUANA_ROUNDS} rounds.",
          **{f"weight: {k}": round(v, 3) for k, v in weights.items()})

    names = sorted(oof)
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    Z = np.column_stack([logit(oof[n]) for n in names])
    Zt = np.column_stack([logit(bagged[n]) for n in names])
    # The stacker's own believed score is cross-validated on the out-of-fold
    # predictions, so it is not graded on rows it was fitted to.
    stacker_cv = float(cross_val_score(LogisticRegression(max_iter=1000), Z, y,
                                       cv=StratifiedKFold(FOLDS, shuffle=True, random_state=SEED + 1),
                                       scoring="roc_auc").mean())
    stacker = LogisticRegression(max_iter=1000).fit(Z, y)
    enter("Hand-built stack, logistic stacker", "stacked ensemble", stacker.predict_proba(Zt)[:, 1], stacker_cv,
          sum(fit_time.values()), sum(member_us.values()),
          "A logistic regression trained on the five models' out-of-fold predictions.",
          **{f"coef: {n}": round(float(c), 3) for n, c in zip(names, stacker.coef_[0])})

    print("[7/9] AutoGluon, the leak and the estimate check ...")
    ag_path = ARTIFACTS / "autogluon.json"
    autogluon = None
    if ag_path.exists():
        ag = json.loads(ag_path.read_text(encoding="utf-8"))
        order = test["order"].astype(int).tolist()
        if ag["test_order"] != order:
            raise RuntimeError("autogluon.json was produced for different test rows; rerun ml/autogluon_run.py")
        enter("AutoGluon, best_quality", "AutoML", np.asarray(ag["test_proba"]), ag["believed_validation_auc"],
              ag["fit_seconds"], ag["predict_seconds"] / ag["n_test"] * 1e6,
              f"AutoGluon {ag['autogluon_version']}, {ag['presets']} presets, {ag['time_limit_seconds']}s limit; "
              f"best model {ag['best_model']}.",
              models_trained=len(ag["leaderboard"]), best_model=ag["best_model"])
        autogluon = {k: ag[k] for k in ("autogluon_version", "presets", "time_limit_seconds", "fit_seconds",
                                        "best_model", "believed_validation_auc")}
        autogluon["leaderboard"] = sorted(ag["leaderboard"], key=lambda r: -r["score_val"])[:15]
        autogluon["models_trained"] = len(ag["leaderboard"])
    else:
        print("      artifacts/autogluon.json not found; AutoGluon left out of this run.")

    leak_pipe = Pipeline([("pre", preprocessor(scale=False, extra_numeric=(LEAK,))), ("m", lgbm())])
    leak_oof, leak_test, _, _ = bag(leak_pipe, train[FEATURES + [LEAK]], y, test[FEATURES + [LEAK]], folds)
    leak = {"honest_test_auc": float(roc_auc_score(yt, bagged["LightGBM"])),
            "leaky_test_auc": float(roc_auc_score(yt, leak_test)),
            "honest_believed_auc": float(roc_auc_score(y, oof["LightGBM"])),
            "leaky_believed_auc": float(roc_auc_score(y, leak_oof))}

    # Would a validation that respected time have warned us? Train on 2008,
    # validate on 2009, with the same default LightGBM.
    early, late = train[train["year"] < TEST_YEAR - 1], train[train["year"] == TEST_YEAR - 1]
    chrono = Pipeline([("pre", preprocessor(scale=False)), ("m", lgbm())]).fit(early[FEATURES], early[TARGET])
    estimate_check = {
        "random_cv_auc": float(roc_auc_score(y, oof["LightGBM"])),
        "time_ordered_auc": float(roc_auc_score(late[TARGET], chrono.predict_proba(late[FEATURES])[:, 1])),
        "actual_test_auc": float(roc_auc_score(yt, bagged["LightGBM"])),
        "time_ordered_train": f"{int(early['year'].min())} ({len(early):,} calls)",
        "time_ordered_validate": f"{int(late['year'].min())} ({len(late):,} calls)",
    }

    print(f"[8/9] significance ({BOOTSTRAPS} paired bootstraps) and permutation importance ...")
    pairs = [("Hand-built stack, Caruana selection", "LightGBM, tuned"),
             ("Hand-built stack, logistic stacker", "LightGBM, tuned"),
             ("LightGBM, tuned", "LightGBM, defaults"),
             ("Hand-built stack, Caruana selection", "Hand-built stack, logistic stacker")]
    if autogluon:
        pairs = [("AutoGluon, best_quality", "LightGBM, tuned"),
                 ("AutoGluon, best_quality", "Hand-built stack, Caruana selection")] + pairs
    significance = [{"a": a, "b": b, **bootstrap_auc(yt, test_preds[a], test_preds[b])} for a, b in pairs]
    for s in significance:
        print(f"      {s['a']} vs {s['b']}: {s['diff']:+.4f} [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] {s['verdict']}")

    rng = np.random.default_rng(SEED)
    base_auc = roc_auc_score(yt, stack_test)

    def stack_predict(frame: pd.DataFrame) -> np.ndarray:
        return sum(w * np.mean([m.predict_proba(frame)[:, 1] for m in fold_models[n]], axis=0)
                   for n, w in weights.items())

    importance = []
    for col in FEATURES:
        drops = []
        for _ in range(3):
            shuffled = Xt.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(base_auc - roc_auc_score(yt, stack_predict(shuffled)))
        importance.append({"feature": col, "auc_drop": float(np.mean(drops)), "spread": float(np.std(drops)),
                           "kind": "numeric" if col in NUMERIC else "categorical"})
    importance.sort(key=lambda r: -r["auc_drop"])

    print("[9/9] audit, report and artifacts ...")
    own = leakage.audit_source(ROOT, temporal_modules=("train", "data", "features"))
    leakage.save_report(own, ARTIFACTS / "audit.md", "AutoML Stacking")
    crit = [f for f in own if f.severity == "CRITICAL"]
    executed = leakage.ledger()
    run.audit = {"critical": len(crit), "warnings": sum(1 for f in own if f.severity == "WARNING"),
                 "total": len(own), "runtime_assertions": executed,
                 "assertions_passed": sum(1 for a in executed if a["passed"]), "assertions_run": len(executed)}

    best = run.best
    level_one_rows = [{"model": n, "oof_auc": float(roc_auc_score(y, oof[n])),
                       "test_auc": float(roc_auc_score(yt, bagged[n])), "fit_seconds": round(fit_time[n], 1),
                       "weight": weights.get(n, 0.0), "stacker_coef": float(stacker.coef_[0][names.index(n)])}
                      for n in names]
    optimism = sorted(((m.name, m.metrics["optimism"]) for m in run.models if "optimism" in m.metrics
                       and m.family != "baseline"), key=lambda r: r[1])
    sig = {(s["a"], s["b"]): s for s in significance}
    stack_vs_tuned = sig[("Hand-built stack, Caruana selection", "LightGBM, tuned")]
    report.phase(
        "modeling",
        objective="Build the same kind of ensemble AutoML builds, by hand, beside a tuned single model.",
        narrative=(
            f"Five different learners are each bagged over five folds, giving out-of-fold "
            f"predictions for every training call. Caruana selection then adds, {CARUANA_ROUNDS} "
            f"times, whichever model most improves the out-of-fold AUC of the running average; "
            f"it settled on " + ", ".join(f"{n} ({w:.0%})" for n, w in sorted(weights.items(), key=lambda r: -r[1]))
            + ". A logistic stacker trained on the same out-of-fold predictions is the second "
            f"ensemble. LightGBM is tuned with {TUNING_TRIALS} Optuna trials of five-fold "
            f"cross-validation."
            + (f" AutoGluon {autogluon['autogluon_version']} ran with `{autogluon['presets']}` for up "
               f"to {autogluon['time_limit_seconds']} seconds and trained {autogluon['models_trained']} "
               f"models." if autogluon else "")
        ),
        code_refs=["ml/train.py:bag", "ml/train.py:caruana", "ml/autogluon_run.py:main"],
        gate_notes=f"{len(run.models)} contestants scored on the same {TEST_YEAR} calls.",
    ).add("Caruana weights", ", ".join(f"{n} {w:.0%}" for n, w in sorted(weights.items(), key=lambda r: -r[1]))) \
     .add("Tuning trials", TUNING_TRIALS)

    report.phase(
        "evaluation",
        objective="Score every contestant on the deployment year, and check each against its own promise.",
        narrative=(
            f"**{best.name}** ranks {TEST_YEAR} clients best, with ROC AUC "
            f"{best.metrics['roc_auc']:.4f}. The hand-built Caruana stack against tuned "
            f"LightGBM: {stack_vs_tuned['diff']:+.4f} AUC, 95% interval "
            f"[{stack_vs_tuned['ci_low']:+.4f}, {stack_vs_tuned['ci_high']:+.4f}], "
            f"**{stack_vs_tuned['verdict']}**.\n\n"
            f"**Every method overestimated itself.** Validation drawn at random from "
            f"2008-2009 calls expected more than {TEST_YEAR} delivered; the smallest gap was "
            f"{optimism[0][1]:+.3f} ({optimism[0][0]}) and the largest {optimism[-1][1]:+.3f} "
            f"({optimism[-1][0]}). A time-ordered check, training on 2008 and validating on "
            f"2009, gave {estimate_check['time_ordered_auc']:.4f} against a random-CV "
            f"{estimate_check['random_cv_auc']:.4f}; the actual {TEST_YEAR} score was "
            f"{estimate_check['actual_test_auc']:.4f}.\n\n"
            f"**The leak.** Adding `duration` lifts LightGBM's {TEST_YEAR} AUC from "
            f"{leak['honest_test_auc']:.4f} to {leak['leaky_test_auc']:.4f}, a gain that would "
            f"vanish the moment the model had to score a client before calling them."
        ) if all(m.metrics.get("optimism", 0) > 0 for m in run.models
                 if "optimism" in m.metrics and m.family != "baseline") else (
            f"**{best.name}** ranks {TEST_YEAR} clients best, with ROC AUC "
            f"{best.metrics['roc_auc']:.4f}. The hand-built Caruana stack against tuned "
            f"LightGBM: {stack_vs_tuned['diff']:+.4f} AUC, **{stack_vs_tuned['verdict']}**. "
            f"Believed-minus-actual gaps ranged from {optimism[0][1]:+.3f} to "
            f"{optimism[-1][1]:+.3f}. Adding `duration` moves LightGBM's AUC from "
            f"{leak['honest_test_auc']:.4f} to {leak['leaky_test_auc']:.4f}."
        ),
        code_refs=["ml/train.py:bootstrap_auc", "ml/train.py:enter"],
        gate_passed=len(crit) == 0,
        gate_notes="Paired bootstrap on the test year; believed scores compared with actual.",
    ).add("Best ROC AUC", f"{best.metrics['roc_auc']:.4f}") \
     .add("Stack vs tuned LightGBM", stack_vs_tuned["verdict"]) \
     .add("AUC with the duration leak", f"{leak['leaky_test_auc']:.4f}")

    servable = {"LightGBM, tuned", "Hand-built stack, Caruana selection"}
    served = max((m for m in run.models if m.name in servable), key=lambda m: m.metrics["roc_auc"]).name
    report.phase(
        "deployment",
        objective="Serve the stronger of the two servable models, with the stack's parts visible beside it.",
        narrative=(
            f"The API serves **{served}**, the better of the two contestants that run in this "
            f"repository's environment on the {TEST_YEAR} calls. AutoGluon needs its own "
            f"environment, so its results are served from its recorded run rather than live. "
            f"Every prediction also returns the hand-built stack's answer with each member "
            f"model's score and Caruana weight, so an ensemble's output can be traced to its parts."
        ),
        code_refs=["server/inference.py:predict", "server/main.py"],
        gate_notes="Served models persisted with the preprocessing they were trained with.",
    ).add("Served model", served).add("Serving port", 8007)

    joblib.dump({"fold_models": {n: fold_models[n] for n in weights}, "weights": weights,
                 "tuned_models": tuned_models, "served": served,
                 "features": FEATURES, "numeric": NUMERIC, "categorical": CATEGORICAL,
                 "train_rate": float(y.mean())}, ARTIFACTS / "model.joblib")
    choices = {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL}
    defaults = train.iloc[-1][FEATURES].to_dict()
    run.extra = {
        "shift": shift, "level_one": level_one_rows, "caruana_trajectory": trajectory, "weights": weights,
        "tuning_trace": tuning_trace, "tuned_params": best_params, "significance": significance,
        "leak": leak, "estimate_check": estimate_check, "importance": importance, "autogluon": autogluon,
        "form": {"choices": choices,
                 "defaults": {k: (v.item() if hasattr(v, "item") else v) for k, v in defaults.items()},
                 "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC}},
        "served": served,
    }
    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")

    print()
    print(run.leaderboard_markdown(("average_precision", "believed_auc", "optimism")))
    print(f"leak {leak} | estimate {estimate_check} | critical {len(crit)} | {run.duration_seconds}s")


if __name__ == "__main__":
    main()
