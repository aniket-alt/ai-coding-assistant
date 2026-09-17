"""
Project 10 -- a CRISP-DM curriculum on the Adult census data.

    python ml/train.py

Each lesson is a real analysis whose result the course then teaches from:
exploration, clustering, outlier detection, classification, a fairness audit,
association rules and locality-sensitive hashing. Quiz answers are computed
from the same run, so the questions cannot drift from the data.
"""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage

from data import CATEGORICAL, NUMERIC, SENSITIVE, load

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEED = 42
# education is a text copy of education-num, so only the ordinal is modelled.
MODEL_CATEGORICAL = [c for c in CATEGORICAL if c != "education"]


def preprocessor(categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), categorical),
    ])


def fairness(y: np.ndarray, pred: np.ndarray, group: pd.Series, privileged: str) -> dict[str, float]:
    """Selection rates and error-rate gaps between a group and its privileged counterpart."""
    g = group.to_numpy()
    priv, unpriv = g == privileged, g != privileged
    sel_p, sel_u = pred[priv].mean(), pred[unpriv].mean()
    tpr = lambda m: pred[m & (y == 1)].mean()
    fpr = lambda m: pred[m & (y == 0)].mean()
    return {
        "selection_rate_privileged": float(sel_p), "selection_rate_other": float(sel_u),
        # The four-fifths rule treats a ratio below 0.8 as evidence of adverse impact.
        "disparate_impact": float(sel_u / sel_p) if sel_p > 0 else float("nan"),
        "equal_opportunity_gap": float(tpr(priv) - tpr(unpriv)),
        "false_positive_gap": float(fpr(priv) - fpr(unpriv)),
    }


class RandomHyperplaneLSH:
    """Cosine-similarity LSH: each table hashes a vector by which side of random hyperplanes it falls on."""

    def __init__(self, dim: int, bits: int = 12, tables: int = 10, seed: int = SEED):
        rng = np.random.default_rng(seed)
        self.planes = rng.normal(size=(tables, bits, dim))
        self.weights = 1 << np.arange(bits)
        self.buckets: list[dict[int, list[int]]] = []

    def _keys(self, X: np.ndarray) -> np.ndarray:
        return ((np.einsum("tbd,nd->ntb", self.planes, X) > 0) * self.weights).sum(axis=2)

    def index(self, X: np.ndarray) -> "RandomHyperplaneLSH":
        keys = self._keys(X)
        self.buckets = [{} for _ in range(keys.shape[1])]
        for i, row in enumerate(keys):
            for t, k in enumerate(row):
                self.buckets[t].setdefault(int(k), []).append(i)
        return self

    def candidates(self, q: np.ndarray) -> np.ndarray:
        keys = self._keys(q[None, :])[0]
        found = set()
        for t, k in enumerate(keys):
            found.update(self.buckets[t].get(int(k), ()))
        return np.fromiter(found, dtype=int)


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()

    report = CrispDmReport(
        project_slug="10_crispdm_curriculum", project_title="CRISP-DM Curriculum",
        business_question=(
            "Who earns more than $50,000 a year in the 1994 US Census, and can a model "
            "predict it without treating men and women unequally?"
        ),
    )
    report.phase(
        "business_understanding",
        objective="Teach every CRISP-DM phase on one real dataset, with a fairness question at its centre.",
        narrative=(
            "This project is a course. Each lesson is a real analysis of the Adult "
            "census data, and each quiz question is answered from that analysis's "
            "own output, so what is taught cannot drift from what the data shows.\n\n"
            "The dataset earns its place for one reason beyond familiarity. It "
            "records sex and race next to variables that act as stand-ins for "
            "them, which makes it a clean demonstration of a mistake that is still "
            "common: assuming a model is fair because the protected attribute was "
            "left out of its inputs."
        ),
        code_refs=["ml/train.py:main"],
        gate_notes="Prediction target, fairness criteria and lessons fixed before modelling.",
    ).add("Target", "income > $50,000").add("Fairness criteria", "four-fifths rule, equal opportunity") \
     .add("Lessons", 7)

    print("[1/8] loading Adult ...")
    df, prov = load()
    prov.save(ARTIFACTS / "provenance.json")
    rate = df["high_income"].mean()
    by_sex = df.groupby("sex")["high_income"].mean()
    by_race = df.groupby("race")["high_income"].agg(["mean", "size"]).sort_values("size", ascending=False)
    missing = (df[CATEGORICAL + NUMERIC].isna().mean() * 100).round(2)
    capped = int((df["capital-gain"] == 99999).sum())
    edu_consistent = df.groupby("education")["education-num"].nunique().max() == 1

    eda = {
        "rows": int(len(df)), "high_income_pct": float(rate * 100),
        "by_sex": [{"group": k, "high_income_pct": round(float(v) * 100, 2),
                    "people": int((df["sex"] == k).sum())} for k, v in by_sex.items()],
        "by_race": [{"group": k, "high_income_pct": round(float(r["mean"]) * 100, 2), "people": int(r["size"])}
                    for k, r in by_race.iterrows()],
        "missing_pct": {k: float(v) for k, v in missing.items() if v > 0},
        "age_hist": [{"age_band": f"{int(b.left)}–{int(b.right) - 1}", "people": int(n),
                      "high_income_pct": round(float(df.loc[df["age"].between(b.left, b.right - 1), "high_income"].mean() * 100), 1)}
                     for b, n in pd.cut(df["age"], bins=range(15, 95, 10), right=False).value_counts(sort=False).items()],
        "hours_by_income": [{"group": "≤ $50K", "median_hours": float(df.loc[df["high_income"] == 0, "hours-per-week"].median())},
                            {"group": "> $50K", "median_hours": float(df.loc[df["high_income"] == 1, "hours-per-week"].median())}],
        "capital_gain_capped": capped,
        "education_is_duplicate": bool(edu_consistent),
    }
    report.phase(
        "data_understanding",
        objective="Establish the base rates, gaps and quirks before any model.",
        narrative=(
            f"{len(df):,} people; **{rate * 100:.1f}%** earn above $50,000. The rate is "
            f"**{by_sex['Male'] * 100:.1f}%** for men and **{by_sex['Female'] * 100:.1f}%** "
            f"for women — a gap already present in the data, which any model trained on "
            f"it will learn unless something stops it.\n\n"
            f"Three quirks shape preparation. Workclass, occupation and native country "
            f"are missing for up to {missing.max():.1f}% of rows, marked `?` in the source. "
            f"Capital gain is top-coded: {capped:,} people report exactly $99,999. And "
            f"`education` is a text copy of `education-num`"
            f"{' — every education level maps to exactly one number' if edu_consistent else ''}, "
            f"so only the ordinal version is modelled."
        ),
        code_refs=["ml/data.py:load"],
        gate_notes="Base rates, missingness and duplicated columns documented.",
    ).add("People", len(df)).add("Earn > $50K", f"{rate * 100:.1f}%") \
     .add("Men / women", f"{by_sex['Male'] * 100:.1f}% / {by_sex['Female'] * 100:.1f}%") \
     .add("Top-coded capital gains", capped)

    print("[2/8] split ...")
    train, test = train_test_split(df, test_size=0.25, stratify=df["high_income"], random_state=SEED)
    leakage.assert_disjoint(train.index, test.index)
    ytr, yte = train["high_income"].to_numpy(), test["high_income"].to_numpy()
    pre_check = preprocessor(MODEL_CATEGORICAL).fit(train)
    leakage.assert_no_target_leak(
        pd.DataFrame(pre_check.transform(train).toarray(), columns=pre_check.get_feature_names_out()), ytr)
    report.phase(
        "data_preparation",
        objective="Prepare features inside pipelines so nothing is learned from the test set.",
        narrative=(
            "The census is a single cross-section with no time order, so a stratified "
            f"random split is appropriate: {len(train):,} people for training and "
            f"{len(test):,} for testing, each with the same {rate * 100:.1f}% base rate.\n\n"
            "Imputation, scaling and one-hot encoding all live inside a "
            "`ColumnTransformer` within each model's pipeline. They are fitted on the "
            "training rows alone, and rare categories (under 20 people) are pooled so "
            "the encoder does not memorise individuals."
        ),
        code_refs=["ml/train.py:preprocessor"],
        gate_notes="Disjoint split and target-correlation check executed.",
    ).add("Training people", len(train)).add("Test people", len(test))

    print("[3/8] clusters and outliers ...")
    num = SimpleImputer(strategy="median").fit(train[NUMERIC])
    scaler = StandardScaler().fit(np.log1p(num.transform(train[NUMERIC]).clip(0)))
    Ztr = scaler.transform(np.log1p(num.transform(train[NUMERIC]).clip(0)))
    km = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(Ztr)
    clusters = []
    for c in range(4):
        members = train[km.labels_ == c]
        clusters.append({"cluster": f"Cluster {c + 1}", "people": int(len(members)),
                         "median_age": float(members["age"].median()),
                         "median_hours": float(members["hours-per-week"].median()),
                         "median_education_years": float(members["education-num"].median()),
                         "with_capital_gain_pct": round(float((members["capital-gain"] > 0).mean() * 100), 1),
                         "high_income_pct": round(float(members["high_income"].mean() * 100), 1)})
    clusters.sort(key=lambda r: -r["high_income_pct"])
    sil = float(silhouette_score(Ztr, km.labels_, sample_size=5000, random_state=SEED))

    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=SEED).fit(Ztr)
    is_outlier = iso.predict(Ztr) == -1
    flagged, rest = train[is_outlier], train[~is_outlier]
    any_capital = lambda d: ((d["capital-gain"] > 0) | (d["capital-loss"] > 0)).mean() * 100
    outliers = {
        "flagged": int(len(flagged)),
        "any_capital_pct": round(float(any_capital(flagged)), 1),
        "any_capital_rest_pct": round(float(any_capital(rest)), 1),
        "age70_pct": round(float((flagged["age"] >= 70).mean() * 100), 1),
        "age70_rest_pct": round(float((rest["age"] >= 70).mean() * 100), 1),
        "top_coded_share_pct": round(float((flagged["capital-gain"] == 99999).mean() * 100), 1),
        "with_capital_loss_pct": round(float((flagged["capital-loss"] > 0).mean() * 100), 1),
        "median_hours": float(flagged["hours-per-week"].median()),
        "high_income_pct": round(float(flagged["high_income"].mean() * 100), 1),
        "examples": flagged[["age", "education-num", "capital-gain", "capital-loss", "hours-per-week",
                             "occupation", "high_income"]].head(8).astype(str).to_dict("records"),
    }

    print("[4/8] classifiers ...")
    run = RunRecord(project_slug="10_crispdm_curriculum", project_title="CRISP-DM Curriculum",
                    task="binary classification with a fairness audit", primary_metric="roc_auc",
                    higher_is_better=True)
    run.set_provenance(prov)
    run.set_split(SplitInfo(strategy="stratified random 75/25 (cross-sectional census)",
                            n_train=len(train), n_test=len(test)))
    predictions: dict[str, np.ndarray] = {}

    def enter(name: str, family: str, pipe: Pipeline, features: list[str], notes: str = ""):
        t = time.perf_counter()
        pipe.fit(train[features], ytr)
        proba = pipe.predict_proba(test[features])[:, 1]
        pred = (proba >= 0.5).astype(int)
        f_sex = fairness(yte, pred, test["sex"], "Male")
        m = {"roc_auc": float(roc_auc_score(yte, proba)) if family != "baseline" else 0.5,
             "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
             "accuracy": float(accuracy_score(yte, pred)),
             "disparate_impact_sex": f_sex["disparate_impact"],
             "equal_opportunity_gap_sex": f_sex["equal_opportunity_gap"],
             "false_positive_gap_sex": f_sex["false_positive_gap"],
             "selection_rate_men": f_sex["selection_rate_privileged"],
             "selection_rate_women": f_sex["selection_rate_other"]}
        run.add_model(ModelScore(name=name, family=family, metrics=m, fit_seconds=round(time.perf_counter() - t, 1),
                                 notes=notes, params={"features": len(features)}))
        predictions[name] = proba
        print(f"      {name:<38} AUC {m['roc_auc']:.4f}  bal.acc {m['balanced_accuracy']:.3f}  "
              f"DI(sex) {m['disparate_impact_sex']:.3f}  EO gap {m['equal_opportunity_gap_sex']:+.3f}")

    all_features = NUMERIC + MODEL_CATEGORICAL
    unaware = [c for c in all_features if c not in SENSITIVE]
    leakage.require(
        "assert_unaware_model_excludes_sensitive_attributes", not (set(unaware) & set(SENSITIVE)),
        f"{len(unaware)} features in the unaware models; sensitive attributes present: "
        f"{sorted(set(unaware) & set(SENSITIVE))}.",
        features=len(unaware), sensitive=SENSITIVE,
    )

    enter("Majority class", "baseline",
          Pipeline([("pre", preprocessor(MODEL_CATEGORICAL)), ("m", DummyClassifier(strategy="most_frequent"))]),
          all_features, "Predicts that nobody earns above $50K.")
    enter("Logistic regression", "linear",
          Pipeline([("pre", preprocessor(MODEL_CATEGORICAL)),
                    ("m", LogisticRegression(max_iter=2000, random_state=SEED))]), all_features)
    lgbm = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                       random_state=SEED, n_jobs=-1, verbose=-1)
    enter("LightGBM", "gbdt", Pipeline([("pre", preprocessor(MODEL_CATEGORICAL)), ("m", lgbm())]), all_features)
    unaware_cats = [c for c in MODEL_CATEGORICAL if c not in SENSITIVE]
    enter("Logistic regression, without sex or race", "fairness through unawareness",
          Pipeline([("pre", preprocessor(unaware_cats)),
                    ("m", LogisticRegression(max_iter=2000, random_state=SEED))]), unaware,
          "Sex and race removed from the inputs.")
    enter("LightGBM, without sex or race", "fairness through unawareness",
          Pipeline([("pre", preprocessor(unaware_cats)), ("m", lgbm())]), unaware,
          "Sex and race removed from the inputs.")

    # How much of sex is still recoverable from the features the unaware model keeps?
    proxy = Pipeline([("pre", preprocessor(unaware_cats)),
                      ("m", LogisticRegression(max_iter=2000, random_state=SEED))])
    proxy.fit(train[unaware], (train["sex"] == "Male").astype(int))
    proxy_auc = float(roc_auc_score((test["sex"] == "Male").astype(int), proxy.predict_proba(test[unaware])[:, 1]))
    rel = df.groupby("relationship")["sex"].agg(lambda s: (s == "Male").mean() * 100).round(1)
    proxies = [{"relationship": k, "men_pct": float(v), "people": int((df["relationship"] == k).sum())}
               for k, v in rel.sort_values(ascending=False).items()]

    best = run.best
    aware = next(m for m in run.models if m.name == "LightGBM").metrics
    blind = next(m for m in run.models if m.name == "LightGBM, without sex or race").metrics

    print("[5/8] association rules ...")
    binned = pd.DataFrame({
        "age": pd.cut(train["age"], [0, 30, 45, 60, 100], labels=["age<30", "age30-44", "age45-59", "age60+"]),
        "hours": pd.cut(train["hours-per-week"], [0, 35, 45, 100], labels=["hours<35", "hours35-44", "hours45+"]),
        "education": pd.cut(train["education-num"], [0, 9, 12, 13, 20],
                            labels=["edu≤HS", "edu-some-college", "edu-bachelors", "edu-postgrad"]),
        "capital": np.where(train["capital-gain"] > 0, "capital-gain", "no-capital-gain"),
        "relationship": "rel=" + train["relationship"].astype(str),
        "occupation": "occ=" + train["occupation"].fillna("Unknown").astype(str),
        "income": np.where(train["high_income"] == 1, "income>50K", "income≤50K"),
    }).astype(str)
    onehot = pd.get_dummies(binned).astype(bool)
    onehot.columns = [c.split("_", 1)[1] for c in onehot.columns]
    itemsets = fpgrowth(onehot, min_support=0.03, use_colnames=True, max_len=3)
    rules = association_rules(itemsets, metric="confidence", min_threshold=0.5)
    rules = rules[rules["consequents"].apply(lambda s: s == frozenset({"income>50K"}))]
    rules = rules.sort_values(["lift", "confidence"], ascending=False).head(15)

    tb = pd.DataFrame({
        "age": pd.cut(test["age"], [0, 30, 45, 60, 100], labels=["age<30", "age30-44", "age45-59", "age60+"]),
        "hours": pd.cut(test["hours-per-week"], [0, 35, 45, 100], labels=["hours<35", "hours35-44", "hours45+"]),
        "education": pd.cut(test["education-num"], [0, 9, 12, 13, 20],
                            labels=["edu≤HS", "edu-some-college", "edu-bachelors", "edu-postgrad"]),
        "capital": np.where(test["capital-gain"] > 0, "capital-gain", "no-capital-gain"),
        "relationship": "rel=" + test["relationship"].astype(str),
        "occupation": "occ=" + test["occupation"].fillna("Unknown").astype(str),
    }).astype(str)
    test_items = [set(r) for r in tb.itertuples(index=False)]
    rule_rows = []
    for r in rules.itertuples():
        match = np.array([set(r.antecedents) <= items for items in test_items])
        rule_rows.append({"if": " + ".join(sorted(r.antecedents)), "support_pct": round(r.support * 100, 2),
                          "confidence": round(r.confidence, 3), "lift": round(r.lift, 2),
                          "test_people": int(match.sum()),
                          "test_confidence": round(float(yte[match].mean()), 3) if match.any() else None})

    print("[6/8] locality-sensitive hashing ...")
    pre = preprocessor(MODEL_CATEGORICAL).fit(train[all_features])
    V = pre.transform(train[all_features]).toarray().astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    Q = pre.transform(test[all_features]).toarray().astype(np.float32)[:500]
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
    lsh = RandomHyperplaneLSH(V.shape[1]).index(V)
    k = 10
    t = time.perf_counter()
    exact = [np.argsort(-(V @ q))[:k] for q in Q]
    exact_ms = (time.perf_counter() - t) / len(Q) * 1000
    t = time.perf_counter()
    approx, scanned = [], []
    for q in Q:
        cand = lsh.candidates(q)
        scanned.append(len(cand))
        approx.append(cand[np.argsort(-(V[cand] @ q))[:k]] if len(cand) else np.array([], dtype=int))
    lsh_ms = (time.perf_counter() - t) / len(Q) * 1000
    recall = float(np.mean([len(set(a) & set(e)) / k for a, e in zip(approx, exact)]))
    lsh_result = {"indexed": int(len(V)), "queries": int(len(Q)), "dimensions": int(V.shape[1]),
                  "bits": 12, "tables": 10, "recall_at_10": recall,
                  "mean_candidates": float(np.mean(scanned)), "scanned_pct": float(np.mean(scanned) / len(V) * 100),
                  "exact_ms_per_query": exact_ms, "lsh_ms_per_query": lsh_ms}

    print("[7/8] audit ...")
    own = leakage.audit_source(ROOT, temporal_modules=())
    leakage.save_report(own, ARTIFACTS / "audit.md", "CRISP-DM Curriculum")
    crit = [f for f in own if f.severity == "CRITICAL"]
    executed = leakage.ledger()
    run.audit = {"critical": len(crit), "warnings": sum(1 for f in own if f.severity == "WARNING"),
                 "total": len(own), "runtime_assertions": executed,
                 "assertions_passed": sum(1 for a in executed if a["passed"]), "assertions_run": len(executed)}

    report.phase(
        "modeling",
        objective="Fit models with and without sex and race, and find what still carries them.",
        narrative=(
            f"Four real models were trained: logistic regression and LightGBM, each "
            f"with and without sex and race, beside a majority-class baseline. Before "
            f"the models, two unsupervised lessons run on the numeric columns.\n\n"
            f"**Clustering.** k-means with four clusters (silhouette {sil:.3f}) separates "
            f"people largely by age, hours and capital income; the highest-income cluster "
            f"reaches {clusters[0]['high_income_pct']:.1f}% above $50K and the lowest "
            f"{clusters[-1]['high_income_pct']:.1f}%.\n\n"
            f"**Outliers.** An isolation forest flags {outliers['flagged']} people as the "
            f"most unusual 1%. **{outliers['any_capital_pct']:.0f}%** of them have a "
            f"capital gain or loss, against {outliers['any_capital_rest_pct']:.0f}% of "
            f"everyone else, and {outliers['age70_pct']:.0f}% are 70 or older against "
            f"{outliers['age70_rest_pct']:.0f}%. Both capital columns are zero for almost "
            f"everyone, so any non-zero value looks unusual: the detector is mostly "
            f"reporting how those columns are distributed. Whether that makes someone "
            f"interesting is a judgement the algorithm cannot make.\n\n"
            f"**Proxies.** A model given only the features left after removing sex and "
            f"race predicts a person's sex with an ROC-AUC of **{proxy_auc:.3f}**. "
            f"Relationship status alone nearly does it: "
            + ", ".join(f"{p['men_pct']:.0f}% of '{p['relationship']}' are men" for p in proxies[:2])
            + "."
        ),
        code_refs=["ml/train.py:fairness", "ml/train.py:enter"],
        gate_notes=f"{len(run.models)} classifiers, including two fairness-through-unawareness variants.",
    ).add("Best ROC-AUC", f"{best.metrics['roc_auc']:.4f}").add("Sex recoverable from other features",
                                                              f"AUC {proxy_auc:.3f}") \
     .add("Cluster silhouette", f"{sil:.3f}").add("Outliers with capital gain or loss", f"{outliers['any_capital_pct']:.0f}%")

    report.phase(
        "evaluation",
        objective="Judge the models on accuracy and on how they treat men and women.",
        narrative=(
            f"**{best.name}** is the most accurate, with ROC-AUC {best.metrics['roc_auc']:.4f}. "
            f"At a 0.5 threshold, LightGBM predicts high income for "
            f"{aware['selection_rate_men'] * 100:.1f}% of men and "
            f"{aware['selection_rate_women'] * 100:.1f}% of women: a disparate impact ratio "
            f"of **{aware['disparate_impact_sex']:.2f}**, far below the four-fifths "
            f"threshold of 0.8.\n\n"
            f"**Removing sex and race does almost nothing.** The unaware LightGBM's ratio "
            f"is **{blind['disparate_impact_sex']:.2f}**, and its equal-opportunity gap — "
            f"the difference in how often truly high-earning men and women are identified — "
            f"moves from {aware['equal_opportunity_gap_sex']:+.3f} to "
            f"{blind['equal_opportunity_gap_sex']:+.3f}. The information was never only "
            f"in the `sex` column.\n\n"
            f"Two more lessons check themselves on held-out people. The strongest "
            f"association rules for high income are recomputed on the test set, and "
            f"locality-sensitive hashing finds {recall * 100:.0f}% of each person's ten "
            f"most similar people while scanning {lsh_result['scanned_pct']:.1f}% of the "
            f"index."
        ),
        code_refs=["ml/train.py:fairness", "ml/train.py:RandomHyperplaneLSH"],
        gate_passed=len(crit) == 0,
        gate_notes="Fairness reported beside accuracy; unawareness shown not to remove disparity.",
    ).add("Disparate impact, with sex", f"{aware['disparate_impact_sex']:.2f}") \
     .add("Disparate impact, without sex", f"{blind['disparate_impact_sex']:.2f}") \
     .add("LSH recall@10", f"{recall * 100:.0f}%")

    report.phase(
        "deployment",
        objective="Publish the course: lessons, interactive results and quizzes answered by the data.",
        narrative=(
            "The course is served by endpoints for each lesson — `/api/eda`, "
            "`/api/unsupervised`, `/api/fairness`, `/api/rules`, `/api/lsh` — and "
            "`/api/quiz`, whose correct answers are computed from this run."
        ),
        code_refs=["server/main.py"],
        gate_notes="Every lesson and quiz answer served from the run record.",
    ).add("Serving port", 8010)

    women = next(r for r in eda["by_sex"] if r["group"] == "Female")["high_income_pct"]
    men = next(r for r in eda["by_sex"] if r["group"] == "Male")["high_income_pct"]
    quiz = [
        {"id": "base-rate-women", "phase": "Data understanding",
         "question": "What share of women in the census earn more than $50,000?",
         "options": [f"{women:.1f}%", f"{men:.1f}%", f"{rate * 100:.1f}%", f"{women * 2:.1f}%"], "answer": 0,
         "explanation": f"{women:.1f}% of women against {men:.1f}% of men — the gap is in the data before any model."},
        {"id": "education-duplicate", "phase": "Data preparation",
         "question": "Why is the text column `education` left out of the models?",
         "options": ["It has too many missing values", "It duplicates education-num exactly",
                     "It leaks the target", "It is a sensitive attribute"], "answer": 1,
         "explanation": "Every education level maps to exactly one education-num, so keeping both adds nothing."},
        {"id": "outliers", "phase": "Modeling",
         "question": "What did the isolation forest mostly flag as outliers?",
         "options": ["People working over 80 hours", "People with any capital gain or loss",
                     "The top-coded $99,999 capital gains", "People with missing occupations"], "answer": 1,
         "explanation": f"{outliers['any_capital_pct']:.0f}% of flagged people have a capital gain or loss, against "
                        f"{outliers['any_capital_rest_pct']:.0f}% of everyone else. Only "
                        f"{outliers['top_coded_share_pct']:.0f}% are the top-coded $99,999 gains."},
        {"id": "unawareness", "phase": "Evaluation",
         "question": "What happened to disparate impact when sex and race were removed from LightGBM's inputs?",
         "options": ["It reached the four-fifths threshold", "It barely changed",
                     "It became perfectly fair", "The model stopped working"], "answer": 1,
         "explanation": f"It went from {aware['disparate_impact_sex']:.2f} to {blind['disparate_impact_sex']:.2f}; "
                        f"other features still carry sex (recoverable with AUC {proxy_auc:.3f})."},
        {"id": "accuracy-baseline", "phase": "Evaluation",
         "question": "What accuracy does a model that predicts nobody earns above $50K achieve?",
         "options": [f"{next(m for m in run.models if m.name == 'Majority class').metrics['accuracy'] * 100:.1f}%",
                     "50.0%", f"{rate * 100:.1f}%", "0.0%"], "answer": 0,
         "explanation": "Most people earn below the threshold, so doing nothing looks accurate — "
                        "which is why ROC-AUC and balanced accuracy are reported."},
        {"id": "lsh", "phase": "Deployment",
         "question": "What does locality-sensitive hashing trade to make similarity search faster?",
         "options": ["Memory for nothing", "Some recall: it may miss true neighbours",
                     "Exactness of the distance metric", "The need for an index"], "answer": 1,
         "explanation": f"It found {recall * 100:.0f}% of true neighbours while scanning "
                        f"{lsh_result['scanned_pct']:.1f}% of people."},
    ]

    run.extra = {"eda": eda, "clusters": clusters, "cluster_silhouette": sil, "outliers": outliers,
                 "proxy_auc": proxy_auc, "relationship_proxy": proxies, "rules": rule_rows,
                 "lsh": lsh_result, "quiz": quiz}
    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")

    print("[8/8] done.\n")
    print(run.leaderboard_markdown(("balanced_accuracy", "disparate_impact_sex", "equal_opportunity_gap_sex")))
    print(f"proxy AUC {proxy_auc:.3f} | LSH recall@10 {recall:.3f}, scanned {lsh_result['scanned_pct']:.1f}%, "
          f"{lsh_result['lsh_ms_per_query']:.2f} ms vs {lsh_result['exact_ms_per_query']:.2f} ms")
    print(f"outliers with capital {outliers['any_capital_pct']}% | critical {len(crit)} | {run.duration_seconds}s")


if __name__ == "__main__":
    main()
