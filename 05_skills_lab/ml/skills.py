"""
Project 05 -- the skills catalogue, as executable comparisons.

Every entry answers one question of the form "does this choice matter, here?"
by running the careless version and the careful version of the same analysis on
a real dataset, over many seeds, and reporting the paired difference with a
bootstrap interval. An entry whose interval covers zero says so; the catalogue
is a record of what was measured, not of what is usually claimed.

Each skill returns the same record shape, so one view in the browser can show
any of them and the leaderboard can rank them by how much the careless version
flatters the result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import columns

BOOTSTRAPS = 2000


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------

def preprocessor(numeric: list[str], categorical: list[str], scale: bool = True) -> ColumnTransformer:
    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False))]),
         categorical),
    ])


def logistic(seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=seed)


def model_for(df: pd.DataFrame, seed: int, scale: bool = True) -> Pipeline:
    numeric, categorical = columns(df)
    return Pipeline([("pre", preprocessor(numeric, categorical, scale)), ("m", logistic(seed))])


def cv_auc(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, seed: int, folds: int = 5) -> float:
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())


def split(df: pd.DataFrame, seed: int, test_size: float = 0.25):
    train, test = train_test_split(df, test_size=test_size, stratify=df["target"], random_state=seed)
    return train, test


def features(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if c not in ("target", "family")]]


class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Replace each category by the mean target of its *other* rows.

    Fitted inside a pipeline, the encoding of a training row never includes
    that row's own target, which is what stops the encoder from memorising it.
    """

    def __init__(self, folds: int = 5, seed: int = 0, out_of_fold: bool = True):
        self.folds = folds
        self.seed = seed
        self.out_of_fold = out_of_fold

    def fit(self, X: pd.DataFrame, y=None):
        y = np.asarray(y)
        self.prior_ = float(y.mean())
        self.maps_ = {c: pd.Series(y).groupby(X[c].astype(str).to_numpy()).mean().to_dict() for c in X.columns}
        self.fit_frame_ = X.reset_index(drop=True)
        self.fit_y_ = y
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        same = X.equals(self.fit_frame_)
        out = np.empty((len(X), X.shape[1]))
        for j, c in enumerate(X.columns):
            keys = X[c].astype(str).to_numpy()
            if same and self.out_of_fold:
                # Encode each training row from the other folds only.
                values = np.full(len(X), self.prior_)
                cv = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
                for inner, held in cv.split(X, self.fit_y_):
                    means = pd.Series(self.fit_y_[inner]).groupby(keys[inner]).mean().to_dict()
                    values[held] = [means.get(k, self.prior_) for k in keys[held]]
                out[:, j] = values
            else:
                out[:, j] = [self.maps_[c].get(k, self.prior_) for k in keys]
        return out

    def get_feature_names_out(self, names=None):
        return np.asarray(list(self.fit_frame_.columns))


def oversample(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Duplicate rows of the minority class until both classes are equally common."""
    counts = frame["target"].value_counts()
    minority = counts.idxmin()
    extra = counts.max() - counts.min()
    if extra <= 0:
        return frame
    pool = frame[frame["target"] == minority]
    drawn = pool.sample(n=extra, replace=True, random_state=seed)
    return pd.concat([frame, drawn], ignore_index=True)


# ---------------------------------------------------------------------------
# the record every skill returns
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    id: str
    name: str
    category: str
    question: str
    dataset: str
    metric: str
    metric_meaning: str
    careless_label: str
    careful_label: str
    explanation: str
    pitfalls: list[str]
    run: Callable[[pd.DataFrame, int], tuple[float, float]] = field(repr=False, default=None)
    higher_is_better: bool = True

    def measure(self, frame: pd.DataFrame, seeds: list[int]) -> dict[str, Any]:
        started = time.perf_counter()
        careless, careful = [], []
        for seed in seeds:
            a, b = self.run(frame, seed)
            careless.append(a)
            careful.append(b)
        careless, careful = np.array(careless), np.array(careful)
        # Paired by seed: both versions saw the same split, so the difference
        # removes the split-to-split variation that dominates small datasets.
        diff = careless - careful
        rng = np.random.default_rng(0)
        means = diff[rng.integers(0, len(diff), size=(BOOTSTRAPS, len(diff)))].mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        flatters = (diff.mean() > 0) if self.higher_is_better else (diff.mean() < 0)
        decisive = lo > 0 or hi < 0
        magnitude = abs(float(diff.mean()))
        return {
            "id": self.id, "name": self.name, "category": self.category, "question": self.question,
            "dataset": self.dataset, "metric": self.metric, "metric_meaning": self.metric_meaning,
            "higher_is_better": self.higher_is_better, "seeds": len(seeds),
            "careless": {"label": self.careless_label, "mean": float(careless.mean()), "sd": float(careless.std(ddof=1))},
            "careful": {"label": self.careful_label, "mean": float(careful.mean()), "sd": float(careful.std(ddof=1))},
            "gap": {"mean": float(diff.mean()), "sd": float(diff.std(ddof=1)), "ci_low": float(lo), "ci_high": float(hi),
                    "decisive": bool(decisive), "flatters": bool(flatters and decisive),
                    "negligible": bool(magnitude < 5e-4)},
            "verdict": ("the careless version flatters the result" if decisive and flatters
                        else "the careless version really is worse" if decisive
                        else "no difference this experiment can resolve"),
            "explanation": self.explanation, "pitfalls": self.pitfalls,
            "seconds": round(time.perf_counter() - started, 2),
        }


# ---------------------------------------------------------------------------
# the skills
# ---------------------------------------------------------------------------

def _impute_before_split(df: pd.DataFrame, seed: int) -> tuple[float, float]:
    numeric, categorical = columns(df)
    filled = df.copy()
    for c in numeric:
        filled[c] = filled[c].fillna(filled[c].median())            # every row, including the folds to come
    for c in categorical:
        filled[c] = filled[c].fillna(filled[c].mode().iloc[0])
    careless = cv_auc(Pipeline([("pre", preprocessor(numeric, categorical)), ("m", logistic(seed))]),
                      features(filled), filled["target"].to_numpy(), seed)
    careful = cv_auc(model_for(df, seed), features(df), df["target"].to_numpy(), seed)
    return careless, careful


def _scale_before_split(df: pd.DataFrame, seed: int) -> tuple[float, float]:
    numeric, categorical = columns(df)
    scaled = df.copy()
    scaled[numeric] = StandardScaler().fit_transform(scaled[numeric].fillna(scaled[numeric].median()))
    careless = cv_auc(Pipeline([("pre", preprocessor(numeric, categorical, scale=False)), ("m", logistic(seed))]),
                      features(scaled), scaled["target"].to_numpy(), seed)
    careful = cv_auc(model_for(df, seed), features(df), df["target"].to_numpy(), seed)
    return careless, careful


def _select_on_all_data(df: pd.DataFrame, seed: int, k: int = 8) -> tuple[float, float]:
    numeric, categorical = columns(df)
    pre = preprocessor(numeric, categorical)
    X, y = features(df), df["target"].to_numpy()
    chosen = SelectKBest(f_classif, k=k).fit(pre.fit_transform(X, y), y)   # scores computed from every row
    careless_matrix = chosen.transform(pre.fit_transform(X, y))
    careless = cv_auc(Pipeline([("m", logistic(seed))]), pd.DataFrame(careless_matrix), y, seed)
    careful = cv_auc(Pipeline([("pre", pre), ("select", SelectKBest(f_classif, k=k)), ("m", logistic(seed))]), X, y, seed)
    return careless, careful


def _target_encode(df: pd.DataFrame, seed: int) -> tuple[float, float]:
    _, categorical = columns(df)
    X, y = df[categorical], df["target"].to_numpy()
    # The mistake as it is usually written: encode the whole table first, then
    # cross-validate what comes out of it.
    encoded = OutOfFoldTargetEncoder(seed=seed, out_of_fold=False).fit(X, y).transform(X.iloc[: len(X)].copy())
    careless = cv_auc(Pipeline([("m", logistic(seed))]), pd.DataFrame(encoded, columns=categorical), y, seed)
    careful = cv_auc(Pipeline([("enc", OutOfFoldTargetEncoder(seed=seed, out_of_fold=True)), ("m", logistic(seed))]), X, y, seed)
    return careless, careful


def _neighbours(df: pd.DataFrame) -> Pipeline:
    """
    Nearest neighbours, for the duplication demonstrations.

    How much a duplicated row costs depends on whether the model can memorise
    it. A duplicate sits at distance zero from its original, so this model
    finds it; a logistic regression has nowhere to store it, and the same
    mistake barely moves its score.
    """
    numeric, categorical = columns(df)
    return Pipeline([("pre", preprocessor(numeric, categorical)), ("m", KNeighborsClassifier(n_neighbors=5))])


def _oversample_with(df: pd.DataFrame, seed: int, build: Callable[[pd.DataFrame, int], Pipeline]) -> tuple[float, float]:
    balanced = oversample(df, seed)                                  # duplicates land in every fold
    careless = cv_auc(build(df, seed), features(balanced), balanced["target"].to_numpy(), seed)
    folds, scores = StratifiedKFold(5, shuffle=True, random_state=seed), []
    X, y = features(df), df["target"].to_numpy()
    for train_rows, held_rows in folds.split(X, y):
        inner = oversample(df.iloc[train_rows], seed)
        pipe = build(df, seed).fit(features(inner), inner["target"].to_numpy())
        scores.append(roc_auc_score(y[held_rows], pipe.predict_proba(X.iloc[held_rows])[:, 1]))
    return careless, float(np.mean(scores))


def _oversample_before_split(df: pd.DataFrame, seed: int) -> tuple[float, float]:
    return _oversample_with(df, seed, lambda frame, s: _neighbours(frame))


def _oversample_linear(df: pd.DataFrame, seed: int) -> tuple[float, float]:
    return _oversample_with(df, seed, model_for)


def _threshold_on_test(df: pd.DataFrame, seed: int) -> tuple[float, float]:
    train, held = split(df, seed)
    inner, validation = split(train, seed + 1)
    pipe = model_for(df, seed).fit(features(inner), inner["target"].to_numpy())
    grid = np.linspace(0.05, 0.95, 91)
    p_val = pipe.predict_proba(features(validation))[:, 1]
    p_held = pipe.predict_proba(features(held))[:, 1]
    y_val, y_held = validation["target"].to_numpy(), held["target"].to_numpy()
    best_on_held = max(f1_score(y_held, (p_held >= t).astype(int), zero_division=0) for t in grid)
    chosen = max(grid, key=lambda t: f1_score(y_val, (p_val >= t).astype(int), zero_division=0))
    return best_on_held, f1_score(y_held, (p_held >= chosen).astype(int), zero_division=0)


def _best_of_many(df: pd.DataFrame, seed: int, n: int = 12) -> tuple[float, float]:
    train, held = split(df, seed)
    inner, validation = split(train, seed + 1)
    numeric, categorical = columns(df)
    candidates = []
    for i, C in enumerate(np.logspace(-3, 2, n)):
        pipe = Pipeline([("pre", preprocessor(numeric, categorical)),
                         ("m", LogisticRegression(max_iter=2000, C=float(C), random_state=seed + i))])
        pipe.fit(features(inner), inner["target"].to_numpy())
        candidates.append((roc_auc_score(validation["target"], pipe.predict_proba(features(validation))[:, 1]),
                           roc_auc_score(held["target"], pipe.predict_proba(features(held))[:, 1])))
    best_on_held = max(h for _, h in candidates)
    picked_on_validation = max(candidates, key=lambda r: r[0])[1]
    return best_on_held, picked_on_validation


def _accuracy_on_imbalanced(df: pd.DataFrame, seed: int) -> tuple[float, float]:
    train, held = split(df, seed)
    pipe = model_for(df, seed).fit(features(train), train["target"].to_numpy())
    predicted = pipe.predict(features(held))
    y = held["target"].to_numpy()
    return float((predicted == y).mean()), float(balanced_accuracy_score(y, predicted))


def _calibration(df: pd.DataFrame, seed: int) -> tuple[float, float]:
    from sklearn.calibration import CalibratedClassifierCV

    train, held = split(df, seed)
    numeric, categorical = columns(df)
    raw = Pipeline([("pre", preprocessor(numeric, categorical)),
                    ("m", HistGradientBoostingClassifier(max_iter=80, random_state=seed))])
    raw.fit(features(train), train["target"].to_numpy())
    # Three folds, not five: the calibrator fits one copy of the model per fold,
    # and this entry would otherwise cost more than the rest of the catalogue.
    calibrated = CalibratedClassifierCV(clone(raw), method="isotonic", cv=3)
    calibrated.fit(features(train), train["target"].to_numpy())
    y = held["target"].to_numpy()
    return (float(brier_score_loss(y, raw.predict_proba(features(held))[:, 1])),
            float(brier_score_loss(y, calibrated.predict_proba(features(held))[:, 1])))


def _group_split(df: pd.DataFrame, seed: int) -> tuple[float, float]:
    X, y, groups = features(df), df["target"].to_numpy(), df["family"].to_numpy()
    train, held = split(df, seed)
    careless_pipe = model_for(df, seed).fit(features(train), train["target"].to_numpy())
    careless = roc_auc_score(held["target"], careless_pipe.predict_proba(features(held))[:, 1])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_rows, held_rows = next(splitter.split(X, y, groups))
    grouped_pipe = model_for(df, seed).fit(X.iloc[train_rows], y[train_rows])
    careful = roc_auc_score(y[held_rows], grouped_pipe.predict_proba(X.iloc[held_rows])[:, 1])
    return float(careless), float(careful)


def _single_split_spread(df: pd.DataFrame, seed: int) -> tuple[float, float]:
    """One split's estimate against five-fold cross-validation, both on the same data."""
    train, held = split(df, seed)
    pipe = model_for(df, seed).fit(features(train), train["target"].to_numpy())
    one = roc_auc_score(held["target"], pipe.predict_proba(features(held))[:, 1])
    folds = cv_auc(model_for(df, seed), features(df), df["target"].to_numpy(), seed)
    return float(one), float(folds)


CATALOGUE: list[Skill] = [
    Skill("impute-before-split", "Imputation fitted before the split", "Preparation",
          "Does filling missing values on the whole dataset before splitting flatter the result?",
          "titanic", "cross-validated ROC AUC", "What a practitioner would report after cross-validation.",
          "medians and modes from every row", "imputers fitted inside each fold",
          "A median computed over all rows carries a little information from the rows that will become the "
          "validation fold. A fifth of ages are missing here, so there is something to carry -- but a median is a "
          "robust statistic, and leaving out a fifth of the rows rarely moves it, which bounds how much can leak.",
          ["Calling fillna on the full frame before any split.",
           "Imputing with a statistic computed outside the pipeline, then cross-validating the pipeline."],
          _impute_before_split),
    Skill("scale-before-split", "Scaling fitted before the split", "Preparation",
          "Does standardising on the whole dataset before splitting flatter the result?",
          "diabetes", "cross-validated ROC AUC", "What a practitioner would report after cross-validation.",
          "mean and standard deviation from every row", "scaler fitted inside each fold",
          "The scaler sees the validation rows' mean and spread. For a linear model scored by a ranking metric, "
          "standardising is an affine change of the inputs that the fit undoes, so the two versions agree to "
          "floating-point precision. The leak is real and its effect here is nil.",
          ["Scaling once at the top of a notebook and splitting afterwards."],
          _scale_before_split),
    Skill("select-on-all-data", "Feature selection before the split", "Preparation",
          "Does choosing features using every row flatter the result?",
          "credit_g", "cross-validated ROC AUC", "What a practitioner would report after cross-validation.",
          "eight features chosen using every row", "selection repeated inside each fold",
          "Selection is a model decision. Ranking features by how well they separate the target on all rows lets the "
          "validation folds influence which features exist at all.",
          ["Running SelectKBest once, then cross-validating the reduced matrix.",
           "Dropping low-correlation features after looking at the whole dataset."],
          _select_on_all_data),
    Skill("target-encoding", "Target encoding without out-of-fold means", "Preparation",
          "How much does a target encoder flatter the result if a row's own label is in its encoding?",
          "credit_g", "cross-validated ROC AUC", "What a practitioner would report after cross-validation.",
          "category means from the full training set", "each row encoded from the other folds",
          "A category mean that includes the row being encoded carries that row's label. Rare categories are the worst "
          "case: a category appearing once is encoded as its own answer.",
          ["Encoding before splitting, which is the usual way this is written.",
           "Forgetting that a category seen once becomes a perfect predictor of its own row."],
          _target_encode),
    Skill("oversample-before-split", "Oversampling before the split", "Preparation",
          "What happens when duplicated minority rows end up in both training and validation?",
          "blood", "cross-validated ROC AUC", "What a practitioner would report after cross-validation.",
          "classes balanced before cross-validation (5-nearest-neighbours)", "balanced inside each training fold",
          "Random oversampling copies minority rows. Copies made before splitting land on both sides, so the model is "
          "scored partly on rows it memorised.",
          ["Balancing the dataset as a preparation step, before any split.",
           "The same applies to SMOTE, whose synthetic rows interpolate between neighbours that may be split apart."],
          _oversample_before_split),
    Skill("oversample-linear", "The same oversampling, with a linear model", "Preparation",
          "Does the cost of duplicated rows depend on the model they are given to?",
          "blood", "cross-validated ROC AUC", "What a practitioner would report after cross-validation.",
          "classes balanced before cross-validation (logistic regression)", "balanced inside each training fold",
          "The mistake is identical to the entry above; only the model changes. A logistic regression has nowhere to "
          "store an individual row, so a duplicate on both sides of the split cannot be recognised. Read the two "
          "entries together: how much a leak costs is a property of the pipeline, not of the mistake alone.",
          ["Assuming a leak is harmless because it was harmless with a linear model.",
           "Assuming a leak is fatal because it was fatal with a nearest-neighbour or tree model."],
          _oversample_linear),
    Skill("threshold-on-test", "Choosing the decision threshold on the test set", "Evaluation design",
          "How much does picking the best threshold on the test set overstate the score?",
          "diabetes", "F1 on the held-out set", "The F1 a practitioner would publish.",
          "threshold that maximises F1 on the held-out set", "threshold chosen on a validation split",
          "The threshold is a parameter like any other. Choosing it against the test set tunes the model to that "
          "particular sample.",
          ["Sweeping thresholds against the test labels and reporting the maximum.",
           "Reporting the best point on a precision-recall curve drawn from the test set."],
          _threshold_on_test),
    Skill("best-of-many", "Reporting the best of several models on the test set", "Evaluation design",
          "How much does trying twelve models and reporting the best test score overstate it?",
          "credit_g", "ROC AUC on the held-out set", "The AUC a practitioner would publish.",
          "best of twelve on the held-out set", "chosen on validation, then scored once",
          "Each model's test score carries noise. Taking the maximum of twelve selects partly for the model and partly "
          "for the luckiest noise.",
          ["Comparing models on the test set and reporting the winner's score.",
           "Repeating an experiment until it gives a good number."],
          _best_of_many),
    Skill("group-split", "Splitting rows that belong to the same group", "Evaluation design",
          "Do family members on both sides of a split flatter the result?",
          "titanic", "ROC AUC on the held-out set", "The AUC a practitioner would publish.",
          "rows split at random", "families kept on one side",
          "Passengers travelling together share a ticket, a cabin and often a fate. A random split puts relatives on "
          "both sides, so the model can recognise a family rather than learn who survived.",
          ["Splitting rows when the unit that repeats is a person, household or session.",
           "Assuming rows are independent because each is one line in a file."],
          _group_split),
    Skill("single-split", "Trusting one train/test split", "Evaluation design",
          "How far does a single split's score wander from a cross-validated one?",
          "diabetes", "ROC AUC", "One split's score against the mean of five folds, on the same data.",
          "one 75/25 split", "five-fold cross-validation",
          "Neither is careless; the point is the spread. On a few hundred rows one split's score moves by several "
          "points from seed to seed, which is wider than most differences people report between models.",
          ["Comparing two models on a single split of a small dataset.",
           "Reporting a score without saying how it would move under another split."],
          _single_split_spread),
    Skill("accuracy-on-imbalanced", "Reading accuracy on an imbalanced target", "Measurement",
          "What does accuracy hide when one class is rare?",
          "blood", "accuracy against balanced accuracy", "Two readings of the same predictions.",
          "accuracy", "balanced accuracy",
          "Fewer than a quarter of donors gave again, so predicting 'no' for everyone is right most of the time. "
          "Balanced accuracy averages the two classes' recall and gives that strategy 0.5.",
          ["Reporting accuracy on a target where one class is rare.",
           "Choosing a model by accuracy when the rare class is the one that matters."],
          _accuracy_on_imbalanced),
    Skill("calibration", "Calibrating predicted probabilities", "Measurement",
          "Does isotonic calibration improve the probabilities, or only the story?",
          "credit_g", "Brier score (lower is better)", "Mean squared error of the predicted probabilities.",
          "gradient boosting, uncalibrated", "the same model, isotonic calibration",
          "Ranking and probability are different goals. A model can order applicants well and still give probabilities "
          "that are systematically too confident. Calibration is fitted on training data only.",
          ["Reading a predicted probability as a frequency without checking calibration.",
           "Calibrating on the test set, which is the same mistake as tuning a threshold there."],
          _calibration, higher_is_better=False),
]

BY_ID = {s.id: s for s in CATALOGUE}
