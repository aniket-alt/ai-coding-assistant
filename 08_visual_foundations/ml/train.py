"""
Project 08 -- visual foundations, computed from real data.

    python ml/train.py

Four lessons, each backed by a real analysis and a check against a reference:

1. Naive Bayes on SMS spam, written out and matched against scikit-learn.
2. Evaluation: precision, recall and thresholds on held-out messages, and what
   duplicated messages do to a random split.
3. Gradient descent on power-plant data, matched against least squares, with
   and without feature scaling.
4. Backpropagation in a small network, matched against finite differences.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage

from data import POWER_FEATURES, POWER_TARGET, load_power, load_sms
from lessons import (NaiveBayes, TinyMLP, gradient_check, gradient_descent, largest_stable_rate, mse, ols,
                     tokenize)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEED = 42
LEAK_SEEDS = 20
SHORTCODE = r"\d{5,}"


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {"average_precision": float(average_precision_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred))}


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()
    report = CrispDmReport(
        project_slug="08_visual_foundations", project_title="DS Visual Foundations",
        business_question=(
            "Can the foundations of machine learning -- probability, evaluation, gradient descent "
            "and backpropagation -- be taught on real data, with every mechanism checked?"
        ),
    )
    report.phase(
        "business_understanding",
        objective="Teach four foundations interactively, each on a real dataset and verified against a reference.",
        narrative=(
            "Foundations are usually taught on toy numbers, so the lessons never meet the "
            "problems real data brings: duplicated records, features on wildly different "
            "scales, and thresholds that trade one error for another. Each lesson here runs "
            "on a real dataset, and each algorithm is written out and then checked against a "
            "trusted implementation, so what is taught is shown to be correct."
        ),
        code_refs=["ml/lessons.py"],
        gate_notes="Four lessons, two datasets and three reference checks fixed before building.",
    ).add("Lessons", 4).add("Datasets", "SMS Spam Collection, Combined Cycle Power Plant")

    print("[1/6] SMS spam: duplicates and split ...")
    sms, sms_prov = load_sms()
    sms_prov.save(ARTIFACTS / "provenance.json")
    power, power_prov = load_power()
    raw_dupes = int(sms["text"].duplicated().sum())
    dup_spam_share = float(sms.loc[sms["text"].duplicated(keep=False), "spam"].mean())
    unique = sms.drop_duplicates("text").reset_index(drop=True)
    train, test = train_test_split(unique, test_size=0.2, stratify=unique["spam"], random_state=SEED)
    overlap = len(set(train["text"]) & set(test["text"]))
    leakage.require("assert_no_message_in_both_splits", overlap == 0,
                    f"{len(train):,} training and {len(test):,} test messages after removing {raw_dupes:,} "
                    f"duplicates; {overlap} message texts appear in both.",
                    train=len(train), test=len(test), duplicates_removed=raw_dupes, overlap=overlap)
    ytr, yte = train["spam"].to_numpy(), test["spam"].to_numpy()
    report.phase(
        "data_understanding",
        objective="Find what in each dataset would mislead a naive lesson.",
        narrative=(
            f"**SMS.** {len(sms):,} messages, {sms['spam'].mean() * 100:.1f}% spam. {raw_dupes:,} are exact "
            f"copies of an earlier message, and copies are disproportionately spam "
            f"({dup_spam_share * 100:.0f}% of duplicated messages, against "
            f"{sms['spam'].mean() * 100:.1f}% overall): the same promotion sent to many people.\n\n"
            f"**Power plant.** {len(power):,} hourly readings. Ambient pressure averages "
            f"{power['AP'].mean():,.0f} mbar while temperature averages {power['AT'].mean():.1f} °C, so "
            f"unscaled features differ by two orders of magnitude."
        ),
        code_refs=["ml/data.py:load_sms", "ml/data.py:load_power"],
        gate_notes="Duplicates counted; feature scales recorded.",
    ).add("Duplicate messages", raw_dupes).add("Spam share", f"{sms['spam'].mean() * 100:.1f}%")

    print("[2/6] classifiers ...")
    run = RunRecord(project_slug="08_visual_foundations", project_title="DS Visual Foundations",
                    task="SMS spam classification (lesson 1 and 2)", primary_metric="average_precision",
                    higher_is_better=True)
    run.set_provenance(sms_prov)
    run.set_split(SplitInfo(strategy="deduplicated messages, stratified random 80/20", n_train=len(train), n_test=len(test)))
    preds: dict[str, np.ndarray] = {}

    def enter(name, family, p, fit_s=0.0, notes=""):
        run.add_model(ModelScore(name=name, family=family, metrics=scores(yte, p), fit_seconds=round(fit_s, 2), notes=notes))
        preds[name] = p
        m = run.models[-1].metrics
        print(f"      {name:<34} AP {m['average_precision']:.4f}  F1 {m['f1']:.3f}")

    enter("Everything is ham", "baseline", np.zeros(len(test)), notes="Predicts no spam at all.")
    enter("Contains a 5+ digit number", "baseline", test["text"].str.contains(SHORTCODE).astype(float).to_numpy(),
          notes="A one-line hand-written rule: shortcodes and phone numbers. Not tuned on the test messages.")
    t = time.perf_counter()
    nb = NaiveBayes(alpha=1.0).fit(train["text"].tolist(), ytr)
    nb_p = np.array([nb.spam_probability(x) for x in test["text"]])
    enter("Naive Bayes, written out", "probabilistic", nb_p, time.perf_counter() - t,
          "Multinomial, add-one smoothing, from ml/lessons.py.")

    vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    reference = MultinomialNB(alpha=1.0).fit(vec.fit_transform(train["text"]), ytr)
    ref_p = reference.predict_proba(vec.transform(test["text"]))[:, 1]
    gap = float(np.abs(ref_p - nb_p).max())
    leakage.require("assert_naive_bayes_matches_sklearn", gap < 1e-9,
                    f"Spam probabilities for {len(test):,} test messages from the written-out Naive Bayes and "
                    f"scikit-learn's MultinomialNB; largest difference {gap:.1e} (limit 1e-9).",
                    messages=len(test), max_difference=gap)

    t = time.perf_counter()
    tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
    lr = LogisticRegression(max_iter=2000, C=10).fit(tfidf.fit_transform(train["text"]), ytr)
    enter("Logistic regression, TF-IDF", "linear", lr.predict_proba(tfidf.transform(test["text"]))[:, 1], time.perf_counter() - t)

    print(f"[3/6] duplicate leak over {LEAK_SEEDS} splits, thresholds ...")
    leak_rows = []
    for s in range(LEAK_SEEDS):
        for mode, frame in (("random split, duplicates kept", sms), ("deduplicated first", unique)):
            tr, te = train_test_split(frame, test_size=0.2, stratify=frame["spam"], random_state=s)
            model = NaiveBayes().fit(tr["text"].tolist(), tr["spam"].to_numpy())
            p = np.array([model.spam_probability(x) for x in te["text"]])
            seen = te["text"].isin(set(tr["text"])).mean()
            leak_rows.append({"seed": s, "mode": mode, "seen_in_train_pct": float(seen * 100),
                              **scores(te["spam"].to_numpy(), p)})
    leak = {}
    for mode in ("random split, duplicates kept", "deduplicated first"):
        rows = [r for r in leak_rows if r["mode"] == mode]
        leak[mode] = {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows]))}
                      for k in ("average_precision", "f1", "recall", "seen_in_train_pct")}

    precision, recall, thresholds = precision_recall_curve(yte, nb_p)
    step = max(1, len(thresholds) // 200)
    pr_curve = [{"threshold": float(thresholds[i]), "precision": float(precision[i]), "recall": float(recall[i])}
                for i in range(0, len(thresholds), step)]
    report.phase(
        "data_preparation",
        objective="Split so that no message is tested on after being trained on.",
        narrative=(
            f"Messages are deduplicated before an 80/20 stratified split, leaving {len(train):,} to "
            f"train and {len(test):,} to test, with no text in both.\n\n"
            f"Measured over {LEAK_SEEDS} splits: when duplicates are kept, "
            f"{leak['random split, duplicates kept']['seen_in_train_pct']['mean']:.1f}% of test messages "
            f"already appear in training, and Naive Bayes recall is "
            f"{leak['random split, duplicates kept']['recall']['mean'] * 100:.1f}% against "
            f"{leak['deduplicated first']['recall']['mean'] * 100:.1f}% when messages are deduplicated first."
        ),
        code_refs=["ml/train.py:main"],
        gate_notes="Executed assertion: no message text in both splits.",
    ).add("Test messages seen in training, duplicates kept", f"{leak['random split, duplicates kept']['seen_in_train_pct']['mean']:.1f}%")

    print("[4/6] gradient descent ...")
    X = power[list(POWER_FEATURES)].to_numpy(float)
    y = power[POWER_TARGET].to_numpy(float)
    Xtr, Xte, ytr_p, yte_p = train_test_split(X, y, test_size=0.2, random_state=SEED)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
    w_exact = ols(Xtr_s, ytr_p)
    exact_mse = mse(w_exact, Xtr_s, ytr_p)
    limit_scaled, limit_raw = largest_stable_rate(Xtr_s), largest_stable_rate(Xtr)
    cond = lambda A: float(np.linalg.cond(np.hstack([np.ones((len(A), 1)), A])))
    converged = gradient_descent(Xtr_s, ytr_p, lr=0.1, steps=3000, record_every=3000)
    w_gd = np.asarray(converged["weights"])
    coef_gap = float(np.abs(w_gd - w_exact).max() / np.abs(w_exact).max())
    leakage.require("assert_gradient_descent_reaches_least_squares", coef_gap < 1e-6,
                    f"Gradient descent (learning rate 0.1, 3,000 steps, scaled features) against the exact "
                    f"least-squares solution: largest coefficient difference {coef_gap:.1e} relative (limit 1e-6).",
                    relative_gap=coef_gap, steps=3000, lr=0.1)
    runs = []
    for scaled, rates, Xa in ((True, [0.01, 0.1, 0.5, round(limit_scaled * 0.99, 4), round(limit_scaled * 1.01, 4)], Xtr_s),
                              (False, [round(limit_raw * 0.99, 10), round(limit_raw * 1.01, 10)], Xtr)):
        for rate in rates:
            g = gradient_descent(Xa, ytr_p, lr=rate, steps=500, record_every=5)
            runs.append({"scaled": scaled, "lr": rate, "diverged": g["diverged"], "final_mse": g["final_mse"],
                         "trajectory": g["trajectory"]})
    gd = {
        "features": POWER_FEATURES, "target": "Net electrical output (MW)",
        "exact_mse": exact_mse, "exact_test_rmse": math.sqrt(mse(w_exact, Xte_s, yte_p)),
        "mean_baseline_test_rmse": float(np.sqrt(((yte_p - ytr_p.mean()) ** 2).mean())),
        "exact_weights": w_exact.tolist(), "stable_rate_scaled": limit_scaled, "stable_rate_raw": limit_raw,
        "condition_scaled": cond(Xtr_s), "condition_raw": cond(Xtr), "relative_gap": coef_gap,
        "runs": runs, "scaling": {"mean": mu.tolist(), "sd": sd.tolist()},
    }
    raw_best = gradient_descent(Xtr, ytr_p, lr=limit_raw * 0.99, steps=500, record_every=500)
    gd["raw_after_500"] = raw_best["final_mse"]
    scaled_500 = gradient_descent(Xtr_s, ytr_p, lr=0.1, steps=500, record_every=500)
    gd["scaled_after_500"] = scaled_500["final_mse"]

    print("[5/6] backpropagation ...")
    mlp = TinyMLP.init(4, 8, SEED)
    y_mean, y_sd = ytr_p.mean(), ytr_p.std()
    ytr_z = (ytr_p - y_mean) / y_sd
    check_rows = np.random.default_rng(SEED).choice(len(Xtr_s), 64, replace=False)
    check_before = gradient_check(mlp, Xtr_s[check_rows], ytr_z[check_rows])
    losses = []
    for epoch in range(1, 3001):
        loss = mlp.step(Xtr_s, ytr_z, lr=0.05)
        if epoch % 50 == 0:
            losses.append({"epoch": epoch, "mse_standardised": loss})
    check_after = gradient_check(mlp, Xtr_s[check_rows], ytr_z[check_rows])
    worst = max(check_before["max_relative_error"], check_after["max_relative_error"])
    leakage.require("assert_backprop_matches_finite_differences", worst < 1e-5,
                    f"All {check_before['parameters_checked']} parameter gradients compared with central finite "
                    f"differences, before and after training; largest relative error {worst:.1e} (limit 1e-5).",
                    parameters=check_before["parameters_checked"], max_relative_error=worst)
    mlp_rmse = float(np.sqrt(((mlp.forward(Xte_s)["out"] * y_sd + y_mean - yte_p) ** 2).mean()))
    backprop = {"hidden": 8, "epochs": 3000, "lr": 0.05, "losses": losses, "test_rmse": mlp_rmse,
                "linear_test_rmse": gd["exact_test_rmse"], "gradient_check": {"before": check_before, "after": check_after},
                "example": Xte_s[0].tolist(), "example_target": float((yte_p[0] - y_mean) / y_sd)}
    joblib_payload = {"nb": nb, "mlp": mlp, "scaling": (mu, sd), "y": (y_mean, y_sd), "Xtr_s": Xtr_s, "ytr": ytr_p,
                      "Xtr": Xtr}

    print("[6/6] audit, report and artifacts ...")
    own = leakage.audit_source(ROOT, temporal_modules=())
    leakage.save_report(own, ARTIFACTS / "audit.md", "DS Visual Foundations")
    crit = [f for f in own if f.severity == "CRITICAL"]
    executed = leakage.ledger()
    run.audit = {"critical": len(crit), "warnings": sum(1 for f in own if f.severity == "WARNING"),
                 "total": len(own), "runtime_assertions": executed,
                 "assertions_passed": sum(1 for a in executed if a["passed"]), "assertions_run": len(executed)}

    best = run.best
    nbm = next(m for m in run.models if m.name.startswith("Naive"))
    report.phase(
        "modeling",
        objective="Write out each algorithm and check it against a trusted implementation.",
        narrative=(
            f"**Naive Bayes** is written out with add-one smoothing; its spam probabilities match "
            f"scikit-learn's to within {gap:.0e}. **Gradient descent** on scaled power-plant features "
            f"reaches the least-squares coefficients to a relative difference of {coef_gap:.0e}. "
            f"**Backpropagation** in a 4-8-1 tanh network matches finite differences to "
            f"{worst:.0e} over all {check_before['parameters_checked']} parameters."
        ),
        code_refs=["ml/lessons.py:NaiveBayes", "ml/lessons.py:gradient_descent", "ml/lessons.py:TinyMLP.backward"],
        gate_notes="Three executed reference checks.",
    ).add("Naive Bayes vs scikit-learn", f"{gap:.0e}").add("Backprop vs finite differences", f"{worst:.0e}")
    report.phase(
        "evaluation",
        objective="Report what each lesson's analysis actually found.",
        narrative=(
            f"**Spam.** {best.name} has the highest average precision, {best.metrics['average_precision']:.3f}; "
            f"Naive Bayes reaches {nbm.metrics['average_precision']:.3f} with recall "
            f"{nbm.metrics['recall'] * 100:.1f}% at a 0.5 threshold.\n\n"
            f"**Scaling.** Gradient descent is stable only below a learning rate of "
            f"{limit_scaled:.3f} on scaled features and {limit_raw:.1e} on raw ones. After 500 steps, "
            f"scaled features at learning rate 0.1 reach a training MSE of {gd['scaled_after_500']:.2f}; raw "
            f"features, at just under their own limit, reach {raw_best['final_mse']:,.1f}. The exact minimum "
            f"is {exact_mse:.2f}.\n\n"
            f"**Networks.** The trained network's test RMSE is {mlp_rmse:.2f} MW against "
            f"{gd['exact_test_rmse']:.2f} MW for linear least squares and "
            f"{gd['mean_baseline_test_rmse']:.2f} MW for predicting the mean."
        ),
        code_refs=["ml/train.py:scores"],
        gate_passed=len(crit) == 0 and all(a["passed"] for a in executed),
        gate_notes="Baselines beside every result.",
    ).add("Best average precision", f"{best.metrics['average_precision']:.3f}") \
     .add("Stable learning rate, scaled / raw", f"{limit_scaled:.3f} / {limit_raw:.1e}")
    report.phase(
        "deployment",
        objective="Serve the lessons live from the checked implementations.",
        narrative=(
            "The API loads the trained Naive Bayes and network and imports ml/lessons.py, so "
            "classifying a typed message, moving a threshold, rerunning gradient descent at a new "
            "learning rate, or stepping through backpropagation all run the checked code."
        ),
        code_refs=["server/inference.py", "server/main.py"],
        gate_notes="Interactive views call the same functions the checks covered.",
    ).add("Serving port", 8008)

    import joblib
    joblib.dump(joblib_payload, ARTIFACTS / "model.joblib")
    spam_rate = sms["spam"].mean()
    kept, dedup = leak["random split, duplicates kept"], leak["deduplicated first"]
    quiz = [
        {"id": "base-rate", "phase": "Probability", "question": "What share of the SMS messages are spam?",
         "options": [f"{spam_rate * 100:.1f}%", "50.0%", f"{(1 - spam_rate) * 100:.1f}%", "2.0%"], "answer": 0,
         "explanation": f"{spam_rate * 100:.1f}% of {len(sms):,} messages. That is the prior Naive Bayes starts from."},
        {"id": "unseen-word", "phase": "Probability", "question": "How does this Naive Bayes treat a word it never saw in training?",
         "options": ["It makes the message spam", "It is ignored", "It makes the message ham", "It raises an error"], "answer": 1,
         "explanation": "Unseen words carry no evidence either way, so they add nothing to the log-odds."},
        {"id": "duplicates", "phase": "Evaluation",
         "question": "With duplicates kept, what share of test messages already appear in training?",
         "options": [f"{kept['seen_in_train_pct']['mean']:.1f}%", "0.0%", "50.0%", f"{spam_rate * 100:.1f}%"], "answer": 0,
         "explanation": f"{kept['seen_in_train_pct']['mean']:.1f}% on average over {LEAK_SEEDS} splits; after deduplication, none."},
        {"id": "scaling", "phase": "Gradient descent", "question": "Why does gradient descent struggle on the raw power-plant features?",
         "options": ["There are too few rows", "Features on very different scales force a tiny learning rate",
                     "The target is not normal", "Least squares has no solution"], "answer": 1,
         "explanation": f"The largest stable learning rate is {limit_raw:.1e} raw against {limit_scaled:.3f} scaled."},
        {"id": "backprop", "phase": "Backpropagation", "question": "How is the hand-written backpropagation checked?",
         "options": ["Against a textbook", "By nudging each weight and measuring the change in loss",
                     "By training longer", "It is not checked"], "answer": 1,
         "explanation": f"Central finite differences on all {check_before['parameters_checked']} parameters agree to {worst:.0e}."},
    ]
    run.extra = {
        "sms": {"messages": len(sms), "spam_pct": float(spam_rate * 100), "duplicates": raw_dupes,
                "duplicate_spam_share_pct": dup_spam_share * 100, "train": len(train), "test": len(test),
                "top_tokens": nb.top_tokens(15), "vocabulary": len(nb.vocab), "sklearn_gap": gap},
        "leak": leak, "leak_seeds": LEAK_SEEDS, "pr_curve": pr_curve,
        "test_scores": [{"p": float(p), "spam": int(t)} for p, t in zip(nb_p, yte)],
        "gd": gd, "backprop": backprop, "quiz": quiz,
        "power": {"rows": len(power), "duplicates": int(power.duplicated().sum()),
                  "means": {k: float(power[k].mean()) for k in POWER_FEATURES}},
    }
    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")
    print(run.leaderboard_markdown(("f1", "recall")))
    print(f"leak {kept['seen_in_train_pct']['mean']:.1f}% seen; recall {kept['recall']['mean']:.3f} vs {dedup['recall']['mean']:.3f} | "
          f"lr limits {limit_scaled:.4f} / {limit_raw:.2e} | mlp {mlp_rmse:.2f} vs linear {gd['exact_test_rmse']:.2f} | "
          f"checks {gap:.1e} {coef_gap:.1e} {worst:.1e} | critical {len(crit)} | {run.duration_seconds}s")


if __name__ == "__main__":
    main()
