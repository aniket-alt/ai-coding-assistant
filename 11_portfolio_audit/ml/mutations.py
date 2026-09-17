"""
Project 11 -- a benchmark for the portfolio's own leakage auditor.

One clean training script is mutated into variants that each plant a known
kind of leakage, alongside clean controls that plant none. Every variant is
then shown to the two detectors this portfolio relies on:

* the static AST audit (`dsfs.leakage.audit_source`), which reads the code;
* the runtime assertions (`assert_temporal_split`, `assert_disjoint`,
  `assert_no_target_leak`), which run inside it.

Because each variant is also executed with the assertions switched off, the
benchmark records how much each leak inflates the reported score -- so a missed
leak can be ranked by how much damage it would have done.

The data is real (the ULB credit card fraud set from project 06). The leaks are
planted deliberately, and are labelled that way everywhere they are reported.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# The clean script. Slots in {braces} are replaced by each mutation.
# --------------------------------------------------------------------------

TEMPLATE = '''"""Benchmark variant: {title}"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [c for c in COLUMNS if c not in ("Class",)]


def run(frame, leakage):
    frame = frame.sort_values("Time").reset_index(drop=True)
{prepare}
{split}
    leakage.assert_temporal_split(pd.to_timedelta(train["Time"], unit="s"),
                                  pd.to_timedelta(test["Time"], unit="s"))
    leakage.assert_disjoint(train.index, test.index)
    X_train, y_train = train[FEATURES], train["Class"]
    X_test, y_test = test[FEATURES], test["Class"]
    leakage.assert_no_target_leak(X_train, y_train)
{scale}
{fit}
    return average_precision_score(y_test, scores)
'''

CLEAN = {
    "prepare": "",
    "split": (
        "    cut = frame[\"Time\"].quantile(0.7)\n"
        "    train = frame[frame[\"Time\"] < cut]\n"
        "    test = frame[frame[\"Time\"] >= cut + 3600]"
    ),
    "scale": (
        "    scaler = StandardScaler()\n"
        "    scaler.fit(X_train)"
    ),
    "fit": (
        "    model = LogisticRegression(max_iter=1000, random_state=0)\n"
        "    model.fit(scaler.transform(X_train), y_train)\n"
        "    scores = model.predict_proba(scaler.transform(X_test))[:, 1]"
    ),
}


@dataclass(frozen=True)
class Variant:
    id: str
    title: str
    kind: str            # "leak" or "control"
    category: str        # which family of mistake
    explanation: str
    slots: dict

    def source(self, columns: list[str]) -> str:
        filled = {**CLEAN, **self.slots}
        code = TEMPLATE.format(title=self.title, **filled)
        return code.replace("COLUMNS", repr(columns))


VARIANTS: list[Variant] = [
    # -- clean controls: a good detector flags none of these ---------------
    Variant("control_template", "Clean template", "control", "control",
            "Chronological split with a one-hour gap; scaler fitted on training rows.", {}),
    Variant("control_pipeline", "Clean, scaler inside a pipeline", "control", "control",
            "The same model with scaling wrapped in a Pipeline.",
            {"scale": "    pipe = Pipeline([(\"scale\", StandardScaler()), "
                      "(\"m\", LogisticRegression(max_iter=1000, random_state=0))])",
             "fit": "    pipe.fit(X_train, y_train)\n"
                    "    scores = pipe.predict_proba(X_test)[:, 1]"}),
    Variant("control_ordered_split", "Clean, unshuffled train_test_split", "control", "control",
            "train_test_split with shuffle=False on time-sorted rows is a legitimate chronological split.",
            {"split": "    train, test = train_test_split(frame, test_size=0.3, shuffle=False)"}),

    # -- preprocessing leaks --------------------------------------------------
    Variant("scaler_on_all_rows", "Scaler fitted on every row", "leak", "preprocessing",
            "The scaler sees test rows before the split, so test statistics shape training.",
            {"scale": "    scaler = StandardScaler()\n"
                      "    scaler.fit(frame[FEATURES])"}),
    Variant("scaler_on_all_rows_renamed", "Scaler fitted on every row, innocuous name", "leak",
            "preprocessing",
            "The same leak with the full frame stored under an unremarkable variable name.",
            {"scale": "    everything = frame[FEATURES]\n"
                      "    scaler = StandardScaler()\n"
                      "    scaler.fit(everything)"}),
    Variant("scaler_on_test", "Scaler fitted on the test set", "leak", "preprocessing",
            "Scaling parameters are learned from held-out rows.",
            {"scale": "    scaler = StandardScaler()\n"
                      "    scaler.fit(X_test)"}),

    # -- split leaks ------------------------------------------------------------
    Variant("shuffled_split", "Shuffled split on time-ordered data", "leak", "split",
            "A random split scatters later transactions into training.",
            {"split": "    train, test = train_test_split(frame, test_size=0.3, random_state=0)"}),
    Variant("overlapping_windows", "Train and test windows overlap", "leak", "split",
            "The test window starts five hours before training ends.",
            {"split": "    cut = frame[\"Time\"].quantile(0.7)\n"
                      "    train = frame[frame[\"Time\"] < cut]\n"
                      "    test = frame[frame[\"Time\"] >= cut - 5 * 3600]"}),
    Variant("test_rows_copied_into_train", "Test rows copied into training", "leak", "split",
            "A third of the test rows, including their labels, are appended to training.",
            {"split": CLEAN["split"] + "\n"
                      "    train = pd.concat([train, test.sample(frac=0.33, random_state=0)])"}),

    # -- target leaks -------------------------------------------------------------
    Variant("label_as_feature", "The label used as a feature", "leak", "target",
            "The fraud flag itself is left among the inputs.",
            {"prepare": "    frame[\"flag\"] = frame[\"Class\"]",
             "split": CLEAN["split"]}),
    Variant("noisy_label_proxy", "A noisy copy of the label", "leak", "target",
            "A feature built from the label plus noise: correlated, but not perfectly.",
            {"prepare": "    rng = np.random.default_rng(0)\n"
                        "    frame[\"proxy\"] = frame[\"Class\"] * 3 + rng.normal(0, 1, len(frame))"}),
    Variant("future_fraud_rate", "Rolling fraud rate centred on each row", "leak", "target",
            "A centred window averages labels from transactions that happen later.",
            {"prepare": "    frame[\"nearby_fraud_rate\"] = frame[\"Class\"].rolling(501, center=True, "
                        "min_periods=1).mean()"}),
    Variant("in_sample_target_encoding", "Target encoding computed on all rows", "leak", "target",
            "The fraud rate per amount band is computed before the split, test labels included.",
            {"prepare": "    band = pd.qcut(frame[\"Amount\"].rank(method=\"first\"), 50, labels=False)\n"
                        "    frame[\"band_fraud_rate\"] = frame.groupby(band)[\"Class\"].transform(\"mean\")"}),

    # -- evaluation leaks ---------------------------------------------------------
    Variant("tuned_on_test", "Hyperparameter chosen on the test set", "leak", "evaluation",
            "Several models are scored on the test set and only the best score is reported.",
            {"fit": "    best = -1.0\n"
                    "    for c in (0.001, 0.01, 0.1, 1.0, 10.0):\n"
                    "        model = LogisticRegression(C=c, max_iter=1000, random_state=0)\n"
                    "        model.fit(scaler.transform(X_train), y_train)\n"
                    "        s = model.predict_proba(scaler.transform(X_test))[:, 1]\n"
                    "        if average_precision_score(y_test, s) > best:\n"
                    "            best, scores = average_precision_score(y_test, s), s"}),
]


def columns_for(variant: Variant, base_columns: list[str]) -> list[str]:
    """Feature columns a variant's script will see, including any it adds."""
    added = {"label_as_feature": ["flag"], "noisy_label_proxy": ["proxy"],
             "future_fraud_rate": ["nearby_fraud_rate"],
             "in_sample_target_encoding": ["band_fraud_rate"]}
    return base_columns + added.get(variant.id, [])
