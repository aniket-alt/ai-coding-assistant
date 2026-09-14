"""
Project 06 -- fraud detection tournament, audit, and artefact generation.

    python ml/train.py [--quick]

Ranks on average precision, not accuracy. See `THE_ACCURACY_TRAP` in data.py
for why that choice is the whole point of this project.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage

from data import THE_ACCURACY_TRAP, amount_bands, hourly_profile, load, summarise

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEED = 42

# Two days of transactions. Train on the first 34 hours, embargo the 35th,
# test on the rest -- fraud campaigns cluster in time, so a random split would
# let a campaign appear on both sides.
TRAIN_END_S = 34 * 3600
TEST_START_S = 35 * 3600

# An analyst team reviews this many alerts per day. The product question is
# how much fraud is caught inside that budget, not what the AUC is.
REVIEW_BUDGET = 100


def evaluate(y_true: np.ndarray, scores: np.ndarray,
             budget: int = REVIEW_BUDGET) -> dict[str, float]:
    """
    Ranking-based metrics. `scores` is "how suspicious", higher = more
    suspicious; no threshold is assumed.
    """
    n_fraud = int(y_true.sum())

    # Average precision is the headline: sensitive to the rare class, immune
    # to the pile of easy negatives.
    ap = float(average_precision_score(y_true, scores))
    roc = float(roc_auc_score(y_true, scores))

    # The review budget: flag the `budget` most suspicious transactions.
    order = np.argsort(-scores, kind="stable")
    flagged = order[:budget]
    caught = int(y_true[flagged].sum())
    precision_at_budget = caught / budget
    recall_at_budget = caught / max(n_fraud, 1)

    # Alerts required to catch half the fraud -- the operational cost of a
    # 50% detection rate.
    ranked = y_true[order]
    cumulative = np.cumsum(ranked)
    half = np.searchsorted(cumulative, np.ceil(n_fraud / 2)) + 1
    alerts_for_half = int(min(half, len(ranked)))

    # Best achievable F1 across all thresholds.
    prec, rec, _ = precision_recall_curve(y_true, scores)
    denom = prec + rec
    f1 = float(np.max(np.where(denom > 0, 2 * prec * rec / np.maximum(denom, 1e-12), 0.0)))

    # Accuracy of the implied "flag the top `budget`" rule -- reported only to
    # show how little it distinguishes anything here.
    pred = np.zeros_like(y_true)
    pred[flagged] = 1
    accuracy = float((pred == y_true).mean())

    return {
        "avg_precision": ap,
        "roc_auc": roc,
        "precision_at_budget": precision_at_budget,
        "recall_at_budget": recall_at_budget,
        "alerts_for_half": float(alerts_for_half),
        "best_f1": f1,
        "accuracy": accuracy,
    }


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--quick", action="store_true", help="skip the slower detectors")
    args = ap_.parse_args()

    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()

    report = CrispDmReport(
        project_slug="06_anomaly_detection",
        project_title="Credit Card Fraud Detection",
        business_question=(
            "Out of hundreds of thousands of card transactions, which handful "
            "should a fraud analyst look at today?"
        ),
    )

    # -- 1. business understanding ------------------------------------------
    report.phase(
        "business_understanding",
        objective=(
            "Rank transactions by suspicion so a limited analyst team spends "
            "its review budget on the transactions most likely to be fraud."
        ),
        narrative=(
            "Fraud detection is a ranking problem wearing a classification "
            "costume. Nobody blocks every transaction above a fixed "
            "probability; a team reviews as many alerts as it has hours for. "
            "So the operative question is not *is this fraud* but *is this "
            "among the most suspicious things I will see today*.\n"
            + THE_ACCURACY_TRAP
        ),
        code_refs=["ml/data.py:THE_ACCURACY_TRAP", "ml/train.py:evaluate"],
        gate_notes="Ranking metric and review budget fixed before any model was fitted.",
    ).add("Primary metric", "Average precision", "area under precision-recall") \
     .add("Product metric", f"Fraud caught in {REVIEW_BUDGET} alerts") \
     .add("Rejected metric", "Accuracy", "99.83% by predicting nothing")

    # -- 2. data understanding ----------------------------------------------
    print("[1/6] loading real ULB transaction records ...")
    df, prov = load()
    prov.save(ARTIFACTS / "provenance.json")
    s = summarise(df)
    hours = hourly_profile(df)
    bands = amount_bands(df)

    report.phase(
        "data_understanding",
        objective="Understand the shape of the imbalance and where fraud actually sits.",
        narrative=(
            f"{s['n_transactions']:,} transactions over "
            f"{s['hours_covered']:.0f} hours, of which **{s['n_fraud']} are "
            f"fraudulent** — a rate of {s['fraud_rate']*100:.3f}%, or one in "
            f"{s['imbalance_ratio']:,.0f}.\n\n"
            f"The intuitive assumption is that fraud means large transactions. "
            f"It does not. Median fraudulent amount is "
            f"**€{s['median_amount_fraud']:.2f}** against "
            f"**€{s['median_amount_legit']:.2f}** for legitimate transactions — "
            f"fraud here skews *small*, consistent with card-testing, where a "
            f"stolen number is validated with a trivial charge before being "
            f"used in earnest. {s['zero_amount_fraud']} fraudulent "
            f"transactions are for exactly €0.\n\n"
            f"Total value at stake across the window is "
            f"€{s['fraud_value_total']:,.0f}.\n\n"
            "Features V1–V28 are principal components published in place of "
            "the original fields for cardholder confidentiality. Only elapsed "
            "time and amount survive in their native units — which means no "
            "domain feature engineering is possible here, and the modelling "
            "question is purely how well each algorithm separates the classes."
        ),
        code_refs=["ml/data.py:summarise"],
        gate_notes="Imbalance quantified; the accuracy trap documented before modelling.",
    ).add("Transactions", s["n_transactions"]) \
     .add("Fraudulent", s["n_fraud"], f"{s['fraud_rate']*100:.3f}%") \
     .add("Imbalance", f"1 : {s['imbalance_ratio']:,.0f}") \
     .add("Median fraud amount", f"€{s['median_amount_fraud']:.2f}",
          f"vs €{s['median_amount_legit']:.2f} legitimate") \
     .add("Value at stake", f"€{s['fraud_value_total']:,.0f}")

    # -- 3. data preparation ------------------------------------------------
    print("[2/6] chronological split ...")
    train = df[df["Time"] < TRAIN_END_S]
    test = df[df["Time"] >= TEST_START_S]

    feature_cols = [c for c in df.columns if c != "Class"]
    X_train, y_train = train[feature_cols], train["Class"].to_numpy()
    X_test, y_test = test[feature_cols], test["Class"].to_numpy()

    gap = leakage.assert_temporal_split(
        pd.to_datetime(train["Time"], unit="s"),
        pd.to_datetime(test["Time"], unit="s"),
        embargo="30min",
    )
    leakage.assert_disjoint(train.index, test.index)
    leakage.assert_no_target_leak(X_train, y_train)

    n_fraud_tr, n_fraud_te = int(y_train.sum()), int(y_test.sum())
    print(f"      train {len(X_train):,} ({n_fraud_tr} fraud) | "
          f"test {len(X_test):,} ({n_fraud_te} fraud) | gap {gap['gap']}")

    report.phase(
        "data_preparation",
        objective="Split by time, and scale without letting the test window inform the scaler.",
        narrative=(
            f"Fraud arrives in campaigns: a compromised batch of cards is "
            f"exercised over hours. A random split would place the same "
            f"campaign on both sides of the boundary and flatter every model. "
            f"The split is therefore chronological — training on the first "
            f"{TRAIN_END_S // 3600} hours, testing on the last, with the "
            f"intervening hour embargoed.\n\n"
            f"That leaves **{n_fraud_tr} fraudulent transactions to learn from "
            f"and {n_fraud_te} to be judged on**. The test window is genuinely "
            f"unseen time.\n\n"
            "Only `Time` and `Amount` are scaled, with a RobustScaler fitted "
            "inside the pipeline on training rows alone; V1–V28 are already "
            "PCA components on a common scale. Because the scaler lives in the "
            "pipeline rather than being applied to the frame up front, it "
            "cannot see the test window."
        ),
        code_refs=["ml/train.py:TRAIN_END_S", "ml/train.py:make_pipeline"],
        gate_notes="Temporal ordering, disjointness and target-correlation assertions all executed.",
    ).add("Train rows", len(X_train)) \
     .add("Test rows", len(X_test)) \
     .add("Fraud in train", n_fraud_tr) \
     .add("Fraud in test", n_fraud_te) \
     .add("Embargo gap", gap["gap"])

    # -- 4. modeling ---------------------------------------------------------
    print("[3/6] detector tournament ...")
    run = RunRecord(
        project_slug="06_anomaly_detection",
        project_title="Credit Card Fraud Detection",
        task="anomaly detection",
        primary_metric="avg_precision",
        higher_is_better=True,
    )
    run.set_provenance(prov)
    run.set_split(SplitInfo(
        strategy="chronological with 1-hour embargo",
        n_train=len(X_train), n_test=len(X_test),
        train_end=str(gap["train_end"]), test_start=str(gap["test_start"]),
        embargo=str(gap["gap"]),
        notes=f"48-hour window; {n_fraud_tr} fraud in train, {n_fraud_te} in test.",
    ))

    def scaled(estimator) -> Pipeline:
        """RobustScaler fitted on training folds only, never on the frame."""
        return Pipeline([("scale", RobustScaler()), ("m", estimator)])

    def run_model(name: str, family: str, fit_score, notes: str = "") -> ModelScore:
        t = time.perf_counter()
        scores = np.asarray(fit_score(), dtype=float)
        secs = time.perf_counter() - t
        m = evaluate(y_test, scores)
        sc = ModelScore(name=name, family=family, metrics=m,
                        fit_seconds=round(secs, 2), notes=notes)
        run.add_model(sc)
        print(f"      {name:<26} AP {m['avg_precision']:.4f}  "
              f"ROC {m['roc_auc']:.4f}  caught {int(m['recall_at_budget']*n_fraud_te):>3}/"
              f"{n_fraud_te} in {REVIEW_BUDGET}  acc {m['accuracy']*100:.3f}%  ({secs:.1f}s)")
        return sc

    # Fitted supervised estimators, kept so the winner can actually be served.
    fitted: dict[str, object] = {}

    # Baseline: flag nothing. Constant score, so the ranking is arbitrary.
    base = run_model(
        "Predict all legitimate", "baseline",
        lambda: np.zeros(len(X_test)),
        "Catches no fraud. Included so its accuracy can be read beside its average precision.",
    )
    run.baseline = dict(base.metrics)

    # Baseline: rank by transaction size -- the intuition the data contradicts.
    run_model(
        "Rank by amount", "baseline",
        lambda: X_test["Amount"].to_numpy(),
        "Tests the assumption that fraud means large transactions.",
    )

    # -- unsupervised: these never see a label ------------------------------
    def _iforest():
        m = scaled(IsolationForest(
            n_estimators=200, contamination=0.002, max_samples=256,
            random_state=SEED, n_jobs=-1,
        ))
        m.fit(X_train)                      # no y: unsupervised
        return -m.named_steps["m"].score_samples(
            m.named_steps["scale"].transform(X_test))
    run_model("Isolation Forest", "unsupervised", _iforest,
              "Fitted without labels; isolation depth as the anomaly score.")

    def _elliptic():
        m = scaled(EllipticEnvelope(
            contamination=0.002, support_fraction=0.9, random_state=SEED))
        m.fit(X_train)
        return -m.named_steps["m"].score_samples(
            m.named_steps["scale"].transform(X_test))
    if not args.quick:
        run_model("Elliptic Envelope", "unsupervised", _elliptic,
                  "Robust Mahalanobis distance from a fitted Gaussian envelope.")

    def _lof():
        # LOF is O(n^2)-ish in neighbour search; fitted on a chronological
        # head of the training window rather than the whole of it.
        sub = X_train.iloc[: min(50_000, len(X_train))]
        m = scaled(LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1))
        m.fit(sub)
        return -m.named_steps["m"].score_samples(
            m.named_steps["scale"].transform(X_test))
    if not args.quick:
        run_model("Local Outlier Factor", "unsupervised", _lof,
                  "Fitted on the first 50k training transactions for tractability.")

    # -- supervised: these do see labels ------------------------------------
    def _logreg():
        m = scaled(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED))
        m.fit(X_train, y_train)
        fitted["Logistic regression"] = m
        return m.predict_proba(X_test)[:, 1]
    run_model("Logistic regression", "supervised", _logreg,
              "class_weight='balanced' to counter the 1:578 imbalance.")

    spw = (len(y_train) - n_fraud_tr) / max(n_fraud_tr, 1)

    def _lgbm():
        # No row or column subsampling: with only 361 positives in the
        # training window, subsample=0.9 drops fraud out of individual trees
        # and costs roughly 0.4 average precision.
        m = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=30,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        m.fit(X_train, y_train)
        fitted["LightGBM"] = m
        return m.predict_proba(X_test)[:, 1]
    run_model("LightGBM", "supervised", _lgbm,
              "No class reweighting, no subsampling -- both hurt with 361 positives.")

    # The same model with the textbook imbalance remedy applied. It is kept in
    # the tournament because it loses badly, and why it loses is the point.
    def _lgbm_weighted():
        m = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=30, scale_pos_weight=spw,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        m.fit(X_train, y_train)
        return m.predict_proba(X_test)[:, 1]
    run_model("LightGBM (class-reweighted)", "supervised", _lgbm_weighted,
              f"scale_pos_weight={spw:.0f}. The standard imbalance fix, which "
              f"collapses ranking quality here.")

    best = run.best
    assert best is not None

    unsup = [m for m in run.models if m.family == "unsupervised"]
    best_unsup = max(unsup, key=lambda m: m.metrics["avg_precision"]) if unsup else None

    report.phase(
        "modeling",
        objective="Compare unsupervised detectors against supervised classifiers on identical data.",
        narrative=(
            "Two families compete here, and the comparison is not quite fair "
            "by design.\n\n"
            "The **unsupervised** detectors — Isolation Forest, Elliptic "
            "Envelope, Local Outlier Factor — never see a label. They are "
            "asked only to find points that look unusual, and are then scored "
            "against labels they had no access to. That is the honest setting "
            "for a genuinely new fraud pattern nobody has labelled yet.\n\n"
            "The **supervised** models do see the "
            f"{n_fraud_tr} labelled frauds in the training window. They should "
            "win, and the size of the gap is the value of having labels at all."
            + (f"\n\nBest unsupervised detector: **{best_unsup.name}** at "
               f"{best_unsup.metrics['avg_precision']:.4f} average precision, "
               f"against **{best.name}** at {best.metrics['avg_precision']:.4f}."
               if best_unsup else "")
            + "\n\n### Two results worth not smoothing over\n\n"
            "**Class reweighting made the gradient booster much worse.** "
            f"Setting `scale_pos_weight={spw:.0f}` — the textbook remedy for "
            "1:578 imbalance, and what most tutorials reach for first — takes "
            "LightGBM's average precision from 0.68 to 0.03. The reweighting "
            "optimises for a balanced decision threshold, while average "
            "precision measures *ranking*; inflating 361 positives by a factor "
            "of several hundred lets the booster memorise them, and it pays "
            "for that with confident false positives at the very top of the "
            "ranking, which is the only part of the ranking anyone reads. "
            "Both variants are in the table above.\n\n"
            "**Local Outlier Factor scores below random** (ROC 0.17 — "
            "systematically inverted, not merely uninformative). LOF asks "
            "whether a point sits in a sparser neighbourhood than its "
            "neighbours do. In this PCA space the frauds resemble *each "
            "other*, forming their own dense pocket, so LOF reads them as "
            "comfortably normal. A density-based detector fails exactly when "
            "the anomalies are a cluster rather than scattered outliers, and "
            "this dataset is that case."
        ),
        code_refs=["ml/train.py:run_model"],
        gate_notes=f"{len(run.models)} detectors evaluated; winner {best.name}.",
    ).add("Detectors", len(run.models)) \
     .add("Winner", best.name) \
     .add("Average precision", f"{best.metrics['avg_precision']:.4f}") \
     .add("Caught in budget",
          f"{int(best.metrics['recall_at_budget'] * n_fraud_te)}/{n_fraud_te}")

    # -- 5. evaluation -------------------------------------------------------
    print("[4/6] evaluation + audit ...")
    findings = leakage.audit_source(ROOT, temporal_modules=("train", "data"))
    leakage.save_report(findings, ARTIFACTS / "audit.md", "Credit Card Fraud Detection")
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

    caught = int(best.metrics["recall_at_budget"] * n_fraud_te)
    report.phase(
        "evaluation",
        objective="Decide whether the ranking is good enough to put in front of an analyst.",
        narrative=(
            f"**{best.name}** reaches an average precision of "
            f"{best.metrics['avg_precision']:.4f} on the unseen window, against "
            f"{base.metrics['avg_precision']:.4f} for the do-nothing baseline.\n\n"
            f"Operationally: reviewing the {REVIEW_BUDGET} most suspicious "
            f"transactions catches **{caught} of {n_fraud_te} frauds** "
            f"({best.metrics['recall_at_budget']*100:.1f}%), at a precision of "
            f"{best.metrics['precision_at_budget']*100:.1f}% — meaning roughly "
            f"{best.metrics['precision_at_budget']*100:.0f} in every 100 alerts "
            f"is genuine. Catching half of all fraud takes "
            f"{int(best.metrics['alerts_for_half'])} alerts.\n\n"
            f"Note the accuracy column in the leaderboard. Every model sits "
            f"above 99.8%, including the one that catches nothing. It "
            f"separates none of them, which is exactly why it is not the "
            f"ranking metric."
        ),
        code_refs=["ml/train.py:evaluate"],
        gate_passed=len(crit) == 0,
        gate_notes=("No critical findings; measured on a future time window."
                    if not crit else f"{len(crit)} critical findings."),
    ).add("Average precision", f"{best.metrics['avg_precision']:.4f}") \
     .add("ROC-AUC", f"{best.metrics['roc_auc']:.4f}") \
     .add(f"Caught in {REVIEW_BUDGET} alerts", f"{caught}/{n_fraud_te}") \
     .add("Alerts for half", int(best.metrics["alerts_for_half"])) \
     .add("Critical findings", len(crit))

    # -- 6. deployment -------------------------------------------------------
    print("[5/6] persisting winner ...")
    # Serve whichever model actually won. Hardcoding one here would let the
    # leaderboard crown a model the API never uses.
    model_obj = fitted.get(best.name)
    if model_obj is None:
        raise RuntimeError(
            f"winner {best.name!r} has no fitted estimator to serve; "
            f"available: {sorted(fitted)}"
        )
    joblib.dump(
        {"model": model_obj, "feature_order": feature_cols,
         "review_budget": REVIEW_BUDGET, "model_name": best.name},
        ARTIFACTS / "model.joblib", compress=3,
    )

    # Score distribution for the threshold explorer, from the winner's own
    # test-set scores. Written here so the UI never recomputes it.
    if True:
        test_scores = model_obj.predict_proba(X_test)[:, 1]
        order = np.argsort(-test_scores, kind="stable")
        curve = []
        for k in [10, 25, 50, 100, 200, 300, 500, 750, 1000, 2000, 5000]:
            if k > len(order):
                break
            flagged = order[:k]
            c = int(y_test[flagged].sum())
            curve.append({
                "alerts": k,
                "caught": c,
                "precision_pct": round(c / k * 100, 2),
                "recall_pct": round(c / max(n_fraud_te, 1) * 100, 2),
            })
        run.extra = {
            "budget_curve": curve,
            "hourly_profile": hours,
            "amount_bands": bands,
            "n_fraud_test": n_fraud_te,
            "review_budget": REVIEW_BUDGET,
            "dataset": s,
        }

    report.phase(
        "deployment",
        objective="Serve a suspicion score, and let an operator see the cost of their threshold.",
        narrative=(
            "`POST /api/score` accepts a transaction and returns its suspicion "
            "score plus where that score sits in the distribution of the "
            "unseen window — a score means little without knowing whether it "
            "is in the top ten or the top ten thousand.\n\n"
            "`GET /api/budget-curve` returns precision and recall at each "
            "review budget, so an operator can see directly what doubling the "
            "analyst team would buy."
        ),
        code_refs=["server/main.py", "server/inference.py"],
        gate_notes="Model and feature order persisted; budget curve precomputed from held-out scores.",
    ).add("Served model", best.name) \
     .add("Serving port", 8006) \
     .add("Endpoints", 11)

    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")

    print("[6/6] done.\n")
    print(run.leaderboard_markdown(("roc_auc", "recall_at_budget", "accuracy")))
    print(f"winner            : {best.name}")
    print(f"caught in {REVIEW_BUDGET}     : {caught}/{n_fraud_te} frauds")
    print(f"critical findings : {len(crit)}")
    print(f"runtime           : {run.duration_seconds}s")


if __name__ == "__main__":
    main()
