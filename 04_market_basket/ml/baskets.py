"""
Project 04 -- baskets, association rules and recommenders.

Every method in the tournament implements one call:

    recommend(visible_items, k) -> list of item codes

so a random guess, a bestseller list, item co-occurrence and FP-Growth rules
are all judged by the same held-out-item test rather than by statistics only
some of them produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from scipy import sparse


def to_baskets(sales: pd.DataFrame) -> pd.Series:
    """Invoice -> frozenset of distinct stock codes."""
    return sales.groupby("Invoice")["StockCode"].agg(frozenset)


def item_names(sales: pd.DataFrame) -> dict[str, str]:
    """Most common description for each stock code; descriptions vary slightly between lines."""
    return (
        sales.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(lambda s: str(s.value_counts().index[0]).strip())
        .to_dict()
    )


def item_support(baskets: pd.Series) -> pd.Series:
    """Share of baskets containing each item."""
    counts: dict[str, int] = {}
    for b in baskets:
        for item in b:
            counts[item] = counts.get(item, 0) + 1
    # Ties are broken by product code. Iterating a frozenset follows string
    # hashing, which Python randomises per process, so without an explicit
    # tie-break the ranking -- and every result built on it -- changed between
    # runs with the same seed.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return pd.Series(dict(ranked)) / len(baskets)


def hold_out(baskets: pd.Series, seed: int) -> list[tuple[frozenset, str]]:
    """Hide one item from every basket with at least two, keeping the rest visible."""
    rng = np.random.default_rng(seed)
    out = []
    for b in baskets:
        if len(b) < 2:
            continue
        items = sorted(b)
        hidden = items[int(rng.integers(len(items)))]
        out.append((frozenset(items) - {hidden}, hidden))
    return out


# --------------------------------------------------------------------------
# Recommenders
# --------------------------------------------------------------------------

class Recommender:
    name = "recommender"
    uses_rules = False

    def recommend(self, visible: frozenset, k: int) -> list[str]:
        raise NotImplementedError

    def own_recommendations(self, visible: frozenset, k: int) -> list[str]:
        """What the method produces without any fallback; used for coverage."""
        return self.recommend(visible, k)


class RandomItems(Recommender):
    def __init__(self, catalogue: list[str], seed: int) -> None:
        self.catalogue = np.array(sorted(catalogue))
        self.rng = np.random.default_rng(seed)

    def recommend(self, visible, k):
        picks = self.rng.choice(self.catalogue, size=min(len(self.catalogue), k + len(visible)),
                                replace=False)
        return [p for p in picks if p not in visible][:k]


class Bestsellers(Recommender):
    def __init__(self, support: pd.Series) -> None:
        self.ranked = list(support.index)

    def recommend(self, visible, k):
        out = []
        for item in self.ranked:
            if item not in visible:
                out.append(item)
                if len(out) == k:
                    break
        return out


class CoOccurrence(Recommender):
    """
    Item-to-item: for each candidate, the highest P(candidate | visible item)
    over the items already in the basket. No thresholds, no mining.
    """

    def __init__(self, baskets: pd.Series, min_count: int = 20) -> None:
        support = item_support(baskets) * len(baskets)
        self.vocab = [i for i, c in support.items() if c >= min_count]
        index = {item: j for j, item in enumerate(self.vocab)}
        rows, cols = [], []
        for r, b in enumerate(baskets):
            for item in b:
                j = index.get(item)
                if j is not None:
                    rows.append(r)
                    cols.append(j)
        B = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                              shape=(len(baskets), len(self.vocab)))
        C = (B.T @ B).toarray()
        diag = np.diag(C).copy()
        self.conf = (C / np.maximum(diag[:, None], 1)).astype(np.float32)
        np.fill_diagonal(self.conf, 0.0)
        self.index = index

    def recommend(self, visible, k):
        rows = [self.index[i] for i in visible if i in self.index]
        if not rows:
            return []
        scores = self.conf[rows].max(axis=0)
        scores[rows] = -1.0
        # Stable ordering on score, then vocabulary position, so ties resolve identically.
        top = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [self.vocab[j] for j in top if scores[j] > 0]


@dataclass
class RuleSet(Recommender):
    """FP-Growth frequent itemsets turned into single-consequent rules."""
    min_support: float
    min_confidence: float
    fallback: Bestsellers | None = None
    rules: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_antecedent: dict = field(default_factory=dict)
    antecedent_items: set = field(default_factory=set)
    uses_rules: bool = True

    def fit(self, baskets: pd.Series, support: pd.Series) -> "RuleSet":
        # Apriori property: an item below min_support cannot be in any frequent
        # itemset, so pruning those columns first changes nothing but memory.
        keep = list(support[support >= self.min_support].index)
        onehot = _onehot_sparse(baskets, keep, set(keep))

        itemsets = fpgrowth(onehot, min_support=self.min_support, use_colnames=True, max_len=3)
        if itemsets.empty:
            return self
        rules = association_rules(itemsets, metric="confidence", min_threshold=self.min_confidence)
        rules = rules[rules["consequents"].apply(len) == 1].copy()
        rules["consequent"] = rules["consequents"].apply(lambda s: next(iter(s)))
        rules["antecedent_key"] = rules["antecedents"].apply(lambda a: " ".join(sorted(a)))
        rules = rules.sort_values(["confidence", "lift", "antecedent_key", "consequent"],
                                  ascending=[False, False, True, True])
        self.rules = rules[["antecedents", "consequent", "support", "confidence", "lift"]] \
            .reset_index(drop=True)
        for r in self.rules.itertuples():
            self.by_antecedent.setdefault(r.antecedents, []).append((r.consequent, r.confidence, r.lift))
            self.antecedent_items |= set(r.antecedents)
        return self

    def own_recommendations(self, visible, k):
        present = sorted(i for i in visible if i in self.antecedent_items)
        best: dict[str, tuple[float, float]] = {}
        for size in (1, 2):
            for combo in combinations(present, size):
                for item, conf, lift in self.by_antecedent.get(frozenset(combo), ()):
                    if item in visible:
                        continue
                    if item not in best or (conf, lift) > best[item]:
                        best[item] = (conf, lift)
        ranked = sorted(best, key=lambda i: (-best[i][0], -best[i][1], i))
        return ranked[:k]

    def recommend(self, visible, k):
        out = self.own_recommendations(visible, k)
        if self.fallback is not None and len(out) < k:
            for item in self.fallback.recommend(visible | frozenset(out), k - len(out)):
                out.append(item)
        return out


def _onehot_sparse(baskets: pd.Series, keep: list[str], keep_set: set) -> pd.DataFrame:
    index = {item: j for j, item in enumerate(keep)}
    rows, cols = [], []
    for r, b in enumerate(baskets):
        for item in b & keep_set:
            rows.append(r)
            cols.append(index[item])
    M = sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                          shape=(len(baskets), len(keep)))
    return pd.DataFrame.sparse.from_spmatrix(M, columns=keep).astype(pd.SparseDtype(bool, False))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def score_recommender(rec: Recommender, holdout: list[tuple[frozenset, str]], k: int,
                      catalogue_size: int) -> dict[str, float]:
    hits = 0
    reciprocal = 0.0
    covered = 0
    recommended: set[str] = set()
    for visible, hidden in holdout:
        own = rec.own_recommendations(visible, k)
        covered += bool(own)
        recs = rec.recommend(visible, k)
        recommended.update(recs)
        if hidden in recs:
            hits += 1
            reciprocal += 1.0 / (recs.index(hidden) + 1)
    n = len(holdout)
    return {
        "hit_rate_at_5": hits / n,
        "mrr_at_5": reciprocal / n,
        "coverage_pct": covered / n * 100,
        "catalogue_coverage_pct": len(recommended) / catalogue_size * 100,
    }


def rule_survival(rules: pd.DataFrame, test_baskets: pd.Series,
                  min_antecedent_count: int = 10) -> tuple[pd.DataFrame, dict]:
    """
    Recompute every training rule on the later baskets.

    A rule "survives" if its antecedent still appears often enough to judge and
    its lift is still above 1, i.e. the association has not vanished or reversed.
    """
    if rules.empty:
        return rules, {"evaluable": 0, "survival_pct": float("nan"), "confidence_drift": float("nan")}
    vocab = sorted({i for a in rules["antecedents"] for i in a} | set(rules["consequent"]))
    col = {item: j for j, item in enumerate(vocab)}
    vocab_set = set(vocab)
    M = np.zeros((len(test_baskets), len(vocab)), dtype=bool)
    for r, b in enumerate(test_baskets):
        for item in b & vocab_set:
            M[r, col[item]] = True

    n = len(test_baskets)
    out = rules.copy()
    t_conf, t_lift, t_ante = [], [], []
    for r in out.itertuples():
        ante = M[:, [col[i] for i in r.antecedents]].all(axis=1)
        cons = M[:, col[r.consequent]]
        a = int(ante.sum())
        both = int((ante & cons).sum())
        p_cons = cons.mean()
        conf = both / a if a else np.nan
        t_ante.append(a)
        t_conf.append(conf)
        t_lift.append(conf / p_cons if a and p_cons > 0 else np.nan)
    out["test_antecedent_count"] = t_ante
    out["test_confidence"] = t_conf
    out["test_lift"] = t_lift
    judged = out[out["test_antecedent_count"] >= min_antecedent_count]
    summary = {
        "evaluable": int(len(judged)),
        "survival_pct": float((judged["test_lift"] > 1).mean() * 100) if len(judged) else float("nan"),
        "confidence_drift": float((judged["test_confidence"] - judged["confidence"]).abs().mean())
        if len(judged) else float("nan"),
        "median_confidence_ratio": float((judged["test_confidence"] / judged["confidence"]).median())
        if len(judged) else float("nan"),
    }
    return out, summary
