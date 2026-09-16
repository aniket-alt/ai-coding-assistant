"""
Project 04 -- market basket tournament, rule survival, and artefacts.

    python ml/train.py

Association rules mined and scored on the same baskets always look good. Here
rules are mined from purchases before September 2011 and tested on the
following Christmas season: every recommender must recover an item hidden
from real later baskets, and every rule is recomputed on those baskets to see
whether it still holds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "datasets" / "loaders"))

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage  # noqa: E402
from online_retail import clean, load_raw  # noqa: E402

from baskets import (  # noqa: E402
    Bestsellers, CoOccurrence, RandomItems, RuleSet, hold_out, item_names,
    item_support, rule_survival, score_recommender, to_baskets,
)

ARTIFACTS = ROOT / "artifacts"
SEED = 42
K = 5
TRAIN_END = pd.Timestamp("2011-09-01")
TEST_START = pd.Timestamp("2011-09-08")   # one-week embargo


def describe(items, names: dict[str, str]) -> str:
    return " + ".join(names.get(i, i) for i in sorted(items))


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()

    report = CrispDmReport(
        project_slug="04_market_basket",
        project_title="Market Basket Mining",
        business_question=(
            "Which products go together strongly enough to recommend, and do "
            "those associations still hold next season?"
        ),
    )
    report.phase(
        "business_understanding",
        objective="Recommend a next item to add to a basket, using associations that hold up over time.",
        narrative=(
            "Association rules are usually reported with support, confidence "
            "and lift computed on the same transactions they were mined from. "
            "Measured that way, every rule above its thresholds looks valid by "
            "construction.\n\n"
            "The questions a retailer actually has are different: *if I "
            "recommend from these rules, will customers want what I suggest?* "
            "and *will the rules still be true next season?* So rules are mined "
            "from purchases before September 2011 and judged on the following "
            "Christmas season, in two ways:\n\n"
            "- **Recommendation.** One item is hidden from every later basket, "
            "and each method recommends five items from the rest. A hit means "
            "the hidden item was among them.\n"
            "- **Survival.** Every rule is recomputed on the later baskets to "
            "see whether its association still exists.\n\n"
            "Rules compete against two baselines, a random guess and the "
            "current bestsellers, and against plain item co-occurrence, which "
            "learns conditional probabilities but needs no mining or thresholds."
        ),
        code_refs=["ml/baskets.py:score_recommender", "ml/baskets.py:rule_survival"],
        gate_notes="Held-out test, survival definition and baselines fixed before mining.",
    ).add("Primary metric", "Hit rate @ 5", "hidden item recovered") \
     .add("Robustness", "Rule survival", "lift still above 1 next season") \
     .add("Baselines", "Random · Bestsellers") \
     .add("Threshold-free rival", "Item co-occurrence")

    print("[1/6] loading Online Retail II ...")
    raw, prov = load_raw()
    cleaned = clean(raw, require_customer=False)
    sales = cleaned["sales"]
    names = item_names(sales)

    train_lines = sales[sales["InvoiceDate"] < TRAIN_END]
    test_lines = sales[sales["InvoiceDate"] >= TEST_START]
    train_b = to_baskets(train_lines)
    test_b = to_baskets(test_lines)
    support = item_support(train_b)
    prov = prov.stamped(extra={
        "training_baskets": int(len(train_b)),
        "test_baskets": int(len(test_b)),
        "products_in_training": int(len(support)),
    })
    prov.save(ARTIFACTS / "provenance.json")

    sizes = train_b.apply(len)
    steps_md = "\n".join(
        f"| {s['rule']} | {s['removed']:,} | {s['pct']:.2f}% | {s['remaining']:,} |"
        for s in cleaned["steps"]
    )
    report.phase(
        "data_understanding",
        objective="Turn transaction lines into baskets, keeping orders from customers without an ID.",
        narrative=(
            f"The shared Online Retail II loader reduces {cleaned['n_raw']:,} "
            f"lines to genuine product sales. Unlike segmentation, market-basket "
            f"analysis keeps orders with no customer ID: a basket is still a "
            f"basket.\n\n"
            "| Rule | Removed | Share | Remaining |\n|---|---|---|---|\n" + steps_md +
            f"\n\nBefore September 2011 there are **{len(train_b):,} baskets** "
            f"over {len(support):,} products, with a median of "
            f"{int(sizes.median())} distinct products per basket. Purchases are "
            f"very long-tailed: only {int((support >= 0.01).sum())} products "
            f"appear in at least 1% of baskets."
        ),
        code_refs=["datasets/loaders/online_retail.py:clean", "ml/baskets.py:to_baskets"],
        gate_notes=f"{len(train_b):,} training and {len(test_b):,} test baskets.",
    ).add("Training baskets", len(train_b)) \
     .add("Test baskets", len(test_b)) \
     .add("Products", len(support)) \
     .add("Median basket size", int(sizes.median()))

    print("[2/6] split and hold-out ...")
    leakage.assert_temporal_split(train_lines["InvoiceDate"], test_lines["InvoiceDate"], embargo="6D")
    leakage.assert_disjoint(train_lines.index, test_lines.index)
    shared = set(train_b.index) & set(test_b.index)
    leakage.require(
        "assert_no_shared_invoices", not shared,
        f"{len(train_b):,} training and {len(test_b):,} test invoices compared; {len(shared)} in both.",
        train_invoices=len(train_b), test_invoices=len(test_b), shared=len(shared),
    )
    holdout = hold_out(test_b, SEED)
    leaked = sum(1 for visible, hidden in holdout if hidden in visible)
    leakage.require(
        "assert_hidden_item_not_visible", leaked == 0,
        f"{len(holdout):,} test baskets checked; the hidden item was visible to the model in {leaked}.",
        baskets=len(holdout), violations=leaked,
    )
    unseen = sum(1 for _, hidden in holdout if hidden not in support.index)
    unseen_pct = unseen / len(holdout) * 100

    report.phase(
        "data_preparation",
        objective="Build an evaluation no method can pass by seeing the answer.",
        narrative=(
            f"Training covers purchases before {TRAIN_END:%d %B %Y}; testing "
            f"starts a week later, on {TEST_START:%d %B %Y}. From each of the "
            f"**{len(holdout):,} test baskets** with at least two products, one "
            f"product is hidden at random and the rest are shown to the "
            f"recommender.\n\n"
            f"**{unseen_pct:.1f}%** of hidden products never appeared in "
            f"training at all — new Christmas stock, mostly. No method can "
            f"recommend a product it has never seen, so these count as misses "
            f"for everyone. They are kept in rather than filtered out, because "
            f"filtering them would flatter every method equally and hide a real "
            f"limit of mining past purchases.\n\n"
            "Two project-specific checks run alongside the standard ones: no "
            "invoice appears on both sides, and no hidden product is visible to "
            "the model."
        ),
        code_refs=["ml/baskets.py:hold_out", "ml/train.py:TEST_START"],
        gate_notes="Temporal, disjointness, invoice and hidden-item assertions all executed.",
    ).add("Test baskets scored", len(holdout)) \
     .add("Hidden products never seen in training", f"{unseen_pct:.1f}%") \
     .add("Embargo", "7 days")

    print("[3/6] recommender tournament ...")
    run = RunRecord(
        project_slug="04_market_basket",
        project_title="Market Basket Mining",
        task="association rule mining and recommendation",
        primary_metric="hit_rate_at_5",
        higher_is_better=True,
    )
    run.set_provenance(prov)
    run.set_split(SplitInfo(
        strategy="chronological with 7-day embargo, one item hidden per test basket",
        n_train=len(train_b), n_test=len(holdout),
        train_end=str(train_lines["InvoiceDate"].max()), test_start=str(test_lines["InvoiceDate"].min()),
        embargo="7 days",
        notes=f"{unseen_pct:.1f}% of hidden products never appear in training and count as misses.",
    ))

    catalogue = len(support)
    fitted: dict[str, object] = {}
    rule_tables: dict[str, pd.DataFrame] = {}
    survival: dict[str, dict] = {}

    def enter(name: str, family: str, rec, params: dict, notes: str = "") -> None:
        t = time.perf_counter()
        m = score_recommender(rec, holdout, K, catalogue)
        if getattr(rec, "uses_rules", False):
            table, summary = rule_survival(rec.rules, test_b)
            rule_tables[name] = table
            survival[name] = summary
            m.update({"n_rules": float(len(rec.rules)),
                      "rule_survival_pct": summary["survival_pct"],
                      "confidence_drift": summary["confidence_drift"]})
        secs = time.perf_counter() - t
        run.add_model(ModelScore(name=name, family=family, metrics=m, params=params,
                                 fit_seconds=round(secs, 2), notes=notes))
        fitted[name] = rec
        extra = (f"  rules {int(m['n_rules']):>5}  survive {m['rule_survival_pct']:5.1f}%"
                 if "n_rules" in m else "")
        print(f"      {name:<34} hit@5 {m['hit_rate_at_5']*100:5.2f}%  "
              f"coverage {m['coverage_pct']:5.1f}%{extra}  ({secs:.1f}s)")

    enter("Random products", "baseline", RandomItems(list(support.index), SEED), {"k": K})
    popular = Bestsellers(support)
    enter("Bestsellers", "baseline", popular, {"k": K},
          "The five most frequently bought products not already in the basket.")
    enter("Item co-occurrence", "item-to-item", CoOccurrence(train_b, min_count=20),
          {"min_count": 20}, "Highest P(item | basket item); no mining and no thresholds.")

    for supp, conf in ((0.02, 0.3), (0.01, 0.2), (0.005, 0.1)):
        rs = RuleSet(min_support=supp, min_confidence=conf).fit(train_b, support)
        enter(f"FP-Growth rules (s≥{supp:.1%}, c≥{conf:.0%})", "association rules", rs,
              {"min_support": supp, "min_confidence": conf, "max_len": 3})

    base_rules = fitted["FP-Growth rules (s≥0.5%, c≥10%)"]
    hybrid = RuleSet(min_support=0.005, min_confidence=0.1, fallback=popular)
    hybrid.rules, hybrid.by_antecedent, hybrid.antecedent_items = (
        base_rules.rules, base_rules.by_antecedent, base_rules.antecedent_items)
    enter("Rules + bestseller fallback", "association rules", hybrid,
          {"min_support": 0.005, "min_confidence": 0.1, "fallback": "bestsellers"},
          "Rule recommendations first; any empty slots filled with bestsellers.")

    best = run.best
    assert best is not None
    rules_only = [m for m in run.models if m.family == "association rules" and "fallback" not in m.params]
    most_rules = max(rules_only, key=lambda m: m.metrics["n_rules"])

    report.phase(
        "modeling",
        objective="Compare mined rules at three thresholds with baselines that need no mining.",
        narrative=(
            f"FP-Growth was run at three support and confidence thresholds, from "
            f"strict to permissive, producing between "
            f"{int(min(m.metrics['n_rules'] for m in rules_only)):,} and "
            f"{int(most_rules.metrics['n_rules']):,} single-consequent rules.\n\n"
            "Stricter thresholds give fewer, stronger rules, but a rule can only "
            "recommend when its antecedent is in the basket, so strict rule sets "
            "stay silent for many baskets. **Coverage** in the leaderboard is the "
            "share of baskets where a method produced any recommendation of its "
            "own. The hybrid fills the silence with bestsellers, which separates "
            "the value of the rules from the value of always saying something."
        ),
        code_refs=["ml/baskets.py:RuleSet", "ml/baskets.py:CoOccurrence"],
        gate_notes=f"{len(run.models)} recommenders scored on {len(holdout):,} held-out baskets.",
    ).add("Recommenders", len(run.models)) \
     .add("Winner", best.name) \
     .add("Hit rate @ 5", f"{best.metrics['hit_rate_at_5']*100:.2f}%") \
     .add("Most rules", f"{int(most_rules.metrics['n_rules']):,}")

    print("[4/6] audit ...")
    findings = leakage.audit_source(ROOT, temporal_modules=("train", "baskets"))
    leakage.save_report(findings, ARTIFACTS / "audit.md", "Market Basket Mining")
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

    bestseller = next(m for m in run.models if m.name == "Bestsellers")
    surv = survival[most_rules.name]
    report.phase(
        "evaluation",
        objective="Establish whether mined rules add anything a retailer could not get from a bestseller list.",
        narrative=(
            f"**{best.name}** recovers the hidden product in "
            f"{best.metrics['hit_rate_at_5']*100:.2f}% of later baskets, against "
            f"{bestseller.metrics['hit_rate_at_5']*100:.2f}% for simply "
            f"recommending the bestsellers.\n\n"
            f"Of the {int(most_rules.metrics['n_rules']):,} rules in the most "
            f"permissive set, {surv['evaluable']:,} had antecedents common enough "
            f"in the test season to judge. **{surv['survival_pct']:.1f}%** of "
            f"those still showed a positive association (lift above 1), and "
            f"their confidence moved by {surv['confidence_drift']:.3f} on average "
            f"between seasons. The median rule kept "
            f"{surv['median_confidence_ratio']*100:.0f}% of its training "
            f"confidence.\n\n"
            "That survival rate should not be read as a discovery. Lift above 1 "
            "is a low bar, and the confidence figures are the stronger evidence. "
            "More revealing still is what the strongest rules say: the top rule "
            f"by lift is *{describe(rule_tables[most_rules.name].sort_values('lift', ascending=False).iloc[0]['antecedents'], names)}* "
            f"→ *{names.get(rule_tables[most_rules.name].sort_values('lift', ascending=False).iloc[0]['consequent'], '')}*. "
            "Many of the strongest associations link variants of one product "
            "line. Those are durable precisely because they are obvious, and a "
            "merchandiser would not need a mining algorithm to find them.\n\n"
            "Read the absolute hit rates with the unseen-product share in mind: "
            "a hidden product that never existed before September counts as a "
            "miss for every method, so no method can approach 100%."
        ),
        code_refs=["ml/baskets.py:rule_survival"],
        gate_passed=len(crit) == 0,
        gate_notes="No critical findings; every recommendation judged on a later season."
        if not crit else f"{len(crit)} critical findings.",
    ).add("Best hit rate @ 5", f"{best.metrics['hit_rate_at_5']*100:.2f}%") \
     .add("Bestsellers hit rate", f"{bestseller.metrics['hit_rate_at_5']*100:.2f}%") \
     .add("Rules surviving next season", f"{surv['survival_pct']:.1f}%") \
     .add("Mean confidence drift", f"{surv['confidence_drift']:.3f}")

    print("[5/6] persisting ...")
    table = rule_tables[most_rules.name]
    judged = table[table["test_antecedent_count"] >= 10]
    top = judged.sort_values("lift", ascending=False).head(60)
    rule_rows = [{
        "antecedent": describe(r.antecedents, names),
        "antecedent_codes": sorted(r.antecedents),
        "consequent": names.get(r.consequent, r.consequent),
        "consequent_code": r.consequent,
        "support_pct": round(r.support * 100, 3),
        "confidence": round(r.confidence, 4),
        "lift": round(r.lift, 2),
        "test_confidence": round(float(r.test_confidence), 4),
        "test_lift": round(float(r.test_lift), 2),
        "survived": bool(r.test_lift > 1),
    } for r in top.itertuples()]

    served = fitted[best.name]
    joblib.dump({"recommender": served, "fallback": popular, "names": names, "model_name": best.name},
                ARTIFACTS / "model.joblib", compress=3)

    report.phase(
        "deployment",
        objective="Recommend for a basket, and show which rules justify each suggestion.",
        narrative=(
            "`POST /api/recommend` takes the products in a basket and returns "
            "five suggestions from the winning method. `GET /api/rules` returns "
            "the strongest rules with both their training statistics and their "
            "measured behaviour in the following season, so a merchandiser can "
            "see which associations are durable and which were seasonal."
        ),
        code_refs=["server/inference.py"],
        gate_notes="Winning recommender, bestseller fallback and product names persisted.",
    ).add("Served recommender", best.name).add("Serving port", 8004)

    run.extra = {
        "rules": rule_rows,
        "survival": survival,
        "unseen_hidden_pct": unseen_pct,
        "popular_items": [{"code": c, "name": names.get(c, c), "support_pct": round(float(s) * 100, 2)}
                          for c, s in support.head(300).items()],
        "cleaning_steps": cleaned["steps"],
    }
    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")

    print("[6/6] done.\n")
    print(run.leaderboard_markdown(("mrr_at_5", "coverage_pct", "rule_survival_pct")))
    print(f"unseen hidden products: {unseen_pct:.1f}%  critical: {len(crit)}  runtime: {run.duration_seconds}s")


if __name__ == "__main__":
    main()
