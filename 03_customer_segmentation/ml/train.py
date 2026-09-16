"""
Project 03 -- segmentation tournament, audit, and artefact generation.

    python ml/train.py

A clustering algorithm always returns clusters, including on noise, and
silhouette scores reward tidy geometry rather than useful groups. So every
segmentation here is judged on two things that can genuinely fail: whether the
segments still separate customers' behaviour a year later, and whether they
survive being rebuilt on a different sample.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "datasets" / "loaders"))

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage  # noqa: E402
from online_retail import clean, load_raw  # noqa: E402

from features import (  # noqa: E402
    FEATURES, SNAPSHOT, customer_features, future_outcomes, model_matrix, split_periods,
)

ARTIFACTS = ROOT / "artifacts"
SEED = 42
STABILITY_PAIRS = 10
SUBSAMPLE = 0.8


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def omega_squared(groups: np.ndarray, y: np.ndarray) -> float:
    """
    Share of variance in y explained by group membership, corrected for the
    number of groups.

    Plain eta-squared rises with every extra segment, so a 20-way split always
    looks better than a 4-way split. Omega-squared subtracts what that many
    groups would explain by chance, and can go negative for a useless split.
    """
    labels = np.unique(groups)
    k, n = len(labels), len(y)
    if k < 2 or n <= k:
        return 0.0
    grand = y.mean()
    ss_between = sum(len(y[groups == g]) * (y[groups == g].mean() - grand) ** 2 for g in labels)
    ss_total = float(((y - grand) ** 2).sum())
    ms_within = (ss_total - ss_between) / (n - k)
    return float((ss_between - (k - 1) * ms_within) / (ss_total + ms_within))


def stability(fit_labels, X: np.ndarray, seed: int = SEED) -> float:
    """
    Mean adjusted Rand index between segmentations built on pairs of
    overlapping 80% subsamples, compared on the customers both contain.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    scores = []
    for _ in range(STABILITY_PAIRS):
        a = np.sort(rng.choice(n, int(n * SUBSAMPLE), replace=False))
        b = np.sort(rng.choice(n, int(n * SUBSAMPLE), replace=False))
        common = np.intersect1d(a, b)
        la = pd.Series(fit_labels(X[a], int(rng.integers(1e9))), index=a)
        lb = pd.Series(fit_labels(X[b], int(rng.integers(1e9))), index=b)
        scores.append(adjusted_rand_score(la.loc[common], lb.loc[common]))
    return float(np.mean(scores))


def profile(labels: np.ndarray, feats: pd.DataFrame, outcome: pd.DataFrame) -> list[dict]:
    """Per-segment medians and second-year behaviour, ordered by future value."""
    frame = feats.assign(segment=labels, future_revenue=outcome["future_revenue"].to_numpy(),
                         returned=outcome["returned"].to_numpy())
    overall_monetary = feats["monetary"].median()
    overall_recency = feats["recency_days"].median()
    rows = []
    for seg, g in frame.groupby("segment"):
        value = "high value" if g["monetary"].median() > 2 * overall_monetary else \
            "low value" if g["monetary"].median() < 0.5 * overall_monetary else "mid value"
        recency = "recent" if g["recency_days"].median() < overall_recency else "lapsing"
        rows.append({
            "segment_id": int(seg),
            "customers": int(len(g)),
            "share_pct": round(len(g) / len(frame) * 100, 1),
            "median_recency_days": float(g["recency_days"].median()),
            "median_orders": float(g["frequency"].median()),
            "median_spend": round(float(g["monetary"].median()), 2),
            "median_basket": round(float(g["avg_basket"].median()), 2),
            "return_rate_pct": round(float(g["return_rate"].mean() * 100), 2),
            "retained_pct": round(float(g["returned"].mean() * 100), 1),
            "median_future_revenue": round(float(g["future_revenue"].median()), 2),
            "mean_future_revenue": round(float(g["future_revenue"].mean()), 2),
            "tag": "noise" if seg == -1 else f"{value} · {recency}",
        })
    rows.sort(key=lambda r: r["mean_future_revenue"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["label"] = f"Segment {i}" if r["segment_id"] != -1 else "Unassigned"
    return rows


def rfm_scores(feats: pd.DataFrame) -> np.ndarray:
    """
    The classic marketing rule: quintile scores for recency, frequency and
    monetary value, summed, then cut into four tiers. No learning involved.
    """
    r = pd.qcut(-feats["recency_days"].rank(method="first"), 5, labels=False) + 1
    f = pd.qcut(feats["frequency"].rank(method="first"), 5, labels=False) + 1
    m = pd.qcut(feats["monetary"].rank(method="first"), 5, labels=False) + 1
    total = (r + f + m).to_numpy()
    return np.digitize(total, [7, 10, 13])


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()

    report = CrispDmReport(
        project_slug="03_customer_segmentation",
        project_title="Customer Segmentation",
        business_question=(
            "Which groups of customers behave differently enough to be treated "
            "differently, and do those groups still mean anything a year later?"
        ),
    )

    report.phase(
        "business_understanding",
        objective="Find segments a retailer could act on, and prove they are more than an artefact of the algorithm.",
        narrative=(
            "Segmentation is unusually easy to fake. Every clustering algorithm "
            "returns clusters, including on random noise, and the usual quality "
            "scores measure how tidy the clusters look rather than whether they "
            "are useful.\n\n"
            "A segment is useful if it predicts how customers will behave, and "
            "it is trustworthy if rebuilding it on a different sample gives the "
            "same answer. So segments here are built from customers' **first "
            "year** of purchases and judged on their **second year**, and every "
            "segmentation is rebuilt on repeated subsamples to measure its "
            "stability.\n\n"
            "Two baselines keep the comparison honest: segments assigned at "
            "random, which should explain nothing, and the classic RFM scoring "
            "rule marketers have used for decades, which any learned method has "
            "to beat to be worth its complexity."
        ),
        code_refs=["ml/train.py:omega_squared", "ml/train.py:stability"],
        gate_notes="Outcome, stability test and baselines fixed before clustering.",
    ).add("Primary metric", "ω² of future spend", "variance explained, corrected for k") \
     .add("Robustness", "Subsample stability", "adjusted Rand index") \
     .add("Baselines", "Random · RFM rule")

    # -- data -----------------------------------------------------------------
    print("[1/6] loading Online Retail II ...")
    raw, prov = load_raw()
    cleaned = clean(raw)
    sales, cancels = cleaned["sales"], cleaned["cancellations"]
    periods = split_periods(sales, cancels)
    prov = prov.stamped(extra={
        "product_lines_after_cleaning": int(len(sales)),
        "customers_in_first_year": int(periods["sales_p1"]["CustomerID"].nunique()),
    })
    prov.save(ARTIFACTS / "provenance.json")

    steps_md = "\n".join(
        f"| {s['rule']} | {s['removed']:,} | {s['pct']:.2f}% | {s['remaining']:,} |"
        for s in cleaned["steps"]
    )
    report.phase(
        "data_understanding",
        objective="Reduce two years of raw transaction lines to genuine product sales by known customers.",
        narrative=(
            f"The export holds **{cleaned['n_raw']:,} lines** across two yearly "
            f"sheets.\n\n"
            "| Rule | Removed | Share | Remaining |\n|---|---|---|---|\n" + steps_md +
            "\n\nThe two sheets **overlap by nine days**, 1 to 9 December 2010, so "
            "a straightforward concatenation counts those orders twice. The "
            "duplicate rule removes them along with genuine repeated lines.\n\n"
            "Cancellations are removed from sales but not discarded: how often a "
            "customer returns goods is behaviour, and it becomes a feature."
        ),
        code_refs=["datasets/loaders/online_retail.py:clean"],
        gate_notes=f"{len(sales):,} product lines retained, each removal rule recorded.",
    ).add("Raw lines", cleaned["n_raw"]) \
     .add("Product sales lines", len(sales)) \
     .add("First-year customers", int(periods["sales_p1"]["CustomerID"].nunique()))

    # -- preparation ------------------------------------------------------------
    print("[2/6] features from year one, outcomes from year two ...")
    p1, p2 = periods["sales_p1"], periods["sales_p2"]
    leakage.assert_temporal_split(p1["InvoiceDate"], p2["InvoiceDate"])
    leakage.assert_disjoint(p1.index, p2.index)

    feats = customer_features(p1, periods["cancel_p1"])
    outcome = future_outcomes(p2, feats.index)
    y_future = np.log1p(outcome["future_revenue"].to_numpy())
    leakage.assert_no_target_leak(feats, y_future)

    X_raw = model_matrix(feats)
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    retained = float(outcome["returned"].mean() * 100)
    report.phase(
        "data_preparation",
        objective="Describe each customer by first-year behaviour only.",
        narrative=(
            f"Features are computed at a snapshot on {SNAPSHOT:%d %B %Y} from "
            f"first-year purchases alone: recency, order count, total spend, "
            f"average basket, distinct products, tenure and return rate. Counts "
            f"and amounts are log-transformed so a handful of wholesale buyers "
            f"do not each form a cluster of their own.\n\n"
            f"The outcome — second-year revenue — is attached afterwards and "
            f"used only for scoring. **{retained:.1f}%** of first-year customers "
            f"bought again in the second year.\n\n"
            "Segmentations are scored on $\\omega^2$ of log future revenue:\n\n"
            "$$\\omega^2 = \\frac{SS_B - (k-1)\\,MS_W}{SS_T + MS_W}$$\n\n"
            "Plain $\\eta^2$ grows with every extra segment, so a twenty-way "
            "split would always win. $\\omega^2$ subtracts what $k$ groups "
            "explain by chance, which is why it can be negative."
        ),
        code_refs=["ml/features.py:customer_features", "ml/train.py:omega_squared"],
        gate_notes="Features and outcome drawn from disjoint, ordered periods; assertions executed.",
    ).add("Customers", len(feats)) \
     .add("Features", len(FEATURES)) \
     .add("Bought again in year two", f"{retained:.1f}%")

    # -- tournament ---------------------------------------------------------------
    print("[3/6] segmentation tournament ...")
    run = RunRecord(
        project_slug="03_customer_segmentation",
        project_title="Customer Segmentation",
        task="customer segmentation (clustering)",
        primary_metric="omega2_future",
        higher_is_better=True,
    )
    run.set_provenance(prov)
    run.set_split(SplitInfo(
        strategy="features from year one, outcomes from year two",
        n_train=len(feats), n_test=len(feats),
        train_end=str(p1["InvoiceDate"].max()), test_start=str(p2["InvoiceDate"].min()),
        notes="Every first-year customer is scored on their second-year revenue.",
    ))

    segments: dict[str, list[dict]] = {}
    fitted_labels: dict[str, np.ndarray] = {}

    def evaluate(name: str, family: str, labels: np.ndarray, fit_labels,
                 params: dict, notes: str = "") -> None:
        t = time.perf_counter()
        uniq = np.unique(labels)
        sil = float(silhouette_score(X, labels, sample_size=min(3000, len(X)),
                                     random_state=SEED)) if len(uniq) > 1 else float("nan")
        prof = profile(labels, feats, outcome)
        sized = [p for p in prof if p["share_pct"] >= 1.0]
        metrics = {
            "omega2_future": omega_squared(labels, y_future),
            "retention_spread": (max(p["retained_pct"] for p in sized)
                                 - min(p["retained_pct"] for p in sized)) if len(sized) > 1 else 0.0,
            "stability_ari": stability(fit_labels, X),
            "silhouette": sil,
            "n_segments": float(len(uniq)),
            "smallest_pct": float(min(p["share_pct"] for p in prof)),
        }
        secs = time.perf_counter() - t
        run.add_model(ModelScore(name=name, family=family, metrics=metrics,
                                 params=params, fit_seconds=round(secs, 2), notes=notes))
        segments[name] = prof
        fitted_labels[name] = labels
        print(f"      {name:<26} ω² {metrics['omega2_future']:+.4f}  "
              f"stability {metrics['stability_ari']:.3f}  silhouette {sil:.3f}  "
              f"k={len(uniq)}  ({secs:.1f}s)")

    rng = np.random.default_rng(SEED)
    evaluate("Random segments", "baseline", rng.integers(0, 4, len(X)),
             lambda Xs, s: np.random.default_rng(s).integers(0, 4, len(Xs)),
             {"k": 4}, "Assigns customers to four segments at random.")

    # The rule scores by rank, and ranks are unchanged by the log transform and
    # scaling, so undoing only the scaling is enough to rescore a subsample.
    evaluate("RFM scoring rule", "baseline", rfm_scores(feats),
             lambda Xs, s: rfm_scores(pd.DataFrame(scaler.inverse_transform(Xs), columns=FEATURES)),
             {"tiers": 4}, "Quintile scores for recency, frequency and spend, summed into four tiers.")

    for k in (3, 4, 5, 6, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        evaluate(f"k-means (k={k})", "partitioning", km.labels_,
                 lambda Xs, s, k=k: KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(Xs),
                 {"k": k, "n_init": 10})

    for k in (4, 6):
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED).fit(X)
        evaluate(f"Gaussian mixture (k={k})", "probabilistic", gm.predict(X),
                 lambda Xs, s, k=k: GaussianMixture(n_components=k, covariance_type="full",
                                                    random_state=s).fit(Xs).predict(Xs),
                 {"k": k, "covariance": "full"})

    ag = AgglomerativeClustering(n_clusters=5, linkage="ward").fit(X)
    evaluate("Agglomerative (ward, k=5)", "hierarchical", ag.labels_,
             lambda Xs, s: AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(Xs),
             {"k": 5, "linkage": "ward"})

    hd = HDBSCAN(min_cluster_size=50).fit(X)
    evaluate("HDBSCAN", "density", hd.labels_,
             lambda Xs, s: HDBSCAN(min_cluster_size=50).fit_predict(Xs),
             {"min_cluster_size": 50}, "Density-based; customers in sparse regions are left unassigned.")

    best = run.best
    assert best is not None
    rfm = next(m for m in run.models if m.name == "RFM scoring rule")
    top_sil = max((m for m in run.models if m.metrics["silhouette"] == m.metrics["silhouette"]),
                  key=lambda m: m.metrics["silhouette"])

    report.phase(
        "modeling",
        objective="Compare partitioning, probabilistic, hierarchical and density-based methods against two baselines.",
        narrative=(
            f"Eleven segmentations were built on identical first-year features. "
            f"The highest silhouette belongs to **{top_sil.name}** "
            f"({top_sil.metrics['silhouette']:.3f}); the highest future-spend "
            f"separation belongs to **{best.name}** "
            f"(ω² {best.metrics['omega2_future']:.4f}). "
            + ("They are the same method." if top_sil.name == best.name else
               "They are different methods, which is the reason silhouette is "
               "not the ranking metric.")
            + f"\n\nThe RFM rule reaches ω² {rfm.metrics['omega2_future']:.4f} "
            f"with no learning at all."
        ),
        code_refs=["ml/train.py:evaluate"],
        gate_notes=f"{len(run.models)} segmentations scored on outcome, stability and geometry.",
    ).add("Segmentations", len(run.models)) \
     .add("Best on future spend", best.name) \
     .add("Best silhouette", top_sil.name) \
     .add("RFM rule ω²", f"{rfm.metrics['omega2_future']:.4f}")

    # -- evaluation ---------------------------------------------------------------
    print("[4/6] audit ...")
    findings = leakage.audit_source(ROOT, temporal_modules=("train", "features"))
    leakage.save_report(findings, ARTIFACTS / "audit.md", "Customer Segmentation")
    crit = [f for f in findings if f.severity == "CRITICAL"]
    executed = leakage.ledger()
    run.audit = {
        "critical": len(crit),
        "warnings": sum(1 for f in findings if f.severity == "WARNING"),
        "total": len(findings),
        "runtime_assertions": executed,
        "assertions_passed": sum(1 for a in executed if a["passed"]),
        "assertions_run": len(executed),
    }

    lead = rfm.metrics["omega2_future"]
    margin = best.metrics["omega2_future"] - lead
    report.phase(
        "evaluation",
        objective="Decide whether any learned segmentation earns its place over the marketing rule.",
        narrative=(
            f"**{best.name}** separates second-year spend best, at ω² "
            f"{best.metrics['omega2_future']:.4f}"
            + (f", {margin:+.4f} ahead of the RFM rule." if best.name != rfm.name
               else ". The RFM rule itself won: no learned method beat a decades-old heuristic.")
            + f" Its segments rebuild with a stability of "
            f"{best.metrics['stability_ari']:.3f} (adjusted Rand index, where 1 "
            f"is identical and 0 is chance), and retention differs by "
            f"{best.metrics['retention_spread']:.1f} percentage points between "
            f"its most and least loyal segments.\n\n"
            "Random segments score close to zero, as they should: that row is "
            "the check that the metric is not rewarding the act of splitting "
            "itself.\n\n"
            "Stability deserves attention in the leaderboard. A method can "
            "separate future spend well and still produce different segments "
            "every time it is rebuilt, and a marketing team cannot act on a "
            "segment that changes on every rerun."
        ),
        code_refs=["ml/train.py:omega_squared", "ml/train.py:stability"],
        gate_passed=len(crit) == 0,
        gate_notes="No critical findings; every segment scored on behaviour it never saw."
        if not crit else f"{len(crit)} critical findings.",
    ).add("Best ω²", f"{best.metrics['omega2_future']:.4f}") \
     .add("Stability (ARI)", f"{best.metrics['stability_ari']:.3f}") \
     .add("Retention spread", f"{best.metrics['retention_spread']:.1f} pts") \
     .add("Critical findings", len(crit))

    # -- deployment -------------------------------------------------------------
    print("[5/6] persisting ...")
    labels = fitted_labels[best.name]
    centroids = {int(s): X[labels == s].mean(axis=0).tolist() for s in np.unique(labels) if s != -1}
    joblib.dump({
        "scaler": scaler, "centroids": centroids, "features": FEATURES,
        "model_name": best.name, "profiles": segments[best.name],
    }, ARTIFACTS / "model.joblib", compress=3)

    report.phase(
        "deployment",
        objective="Assign a new customer to a segment and show what that segment did next.",
        narrative=(
            "`POST /api/assign` takes a customer's recency, order count, spend "
            "and the other features, and returns the nearest segment of the "
            "winning method by distance to its centre in the scaled feature "
            "space, with that segment's second-year behaviour. Nearest-centre "
            "assignment works for every method in the tournament, including "
            "those with no native way to label a new point."
        ),
        code_refs=["server/inference.py"],
        gate_notes="Scaler, segment centres and profiles persisted together.",
    ).add("Served segmentation", best.name).add("Serving port", 8003)

    run.extra = {
        "segments": segments,
        "cleaning_steps": cleaned["steps"],
        "retained_pct": retained,
        "feature_ranges": {
            c: {"p5": float(feats[c].quantile(0.05)), "median": float(feats[c].median()),
                "p95": float(feats[c].quantile(0.95))}
            for c in FEATURES
        },
    }
    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")

    print("[6/6] done.\n")
    print(run.leaderboard_markdown(("stability_ari", "silhouette", "retention_spread")))
    print(f"critical findings: {len(crit)}  runtime: {run.duration_seconds}s")


if __name__ == "__main__":
    main()
