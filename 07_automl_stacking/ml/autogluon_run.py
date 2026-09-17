"""
Project 07 -- the AutoGluon contestant.

    ../.venv-automl/Scripts/python ml/autogluon_run.py

AutoGluon pins older pandas and pyarrow than the rest of the repository, so it
runs in its own environment and hands its results over as a file:
artifacts/autogluon.json holds its test-set probabilities, its internal
leaderboard and the validation score it believed. train.py scores those
probabilities exactly as it scores every other contestant.

It sees the same rows and the same columns as the other contestants: calls
before 2010, without `duration`.
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path

import autogluon.tabular
from autogluon.tabular import TabularPredictor

from data import FEATURES, TARGET, load, split

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "autogluon_model"
TIME_LIMIT = 900
PRESETS = "best_quality"
SEED = 42


def main() -> None:
    df, _ = load()
    train, test = split(df)
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    t = time.perf_counter()
    predictor = TabularPredictor(label=TARGET, eval_metric="roc_auc", path=str(MODEL_DIR),
                                 verbosity=2).fit(
        train[FEATURES + [TARGET]], presets=PRESETS, time_limit=TIME_LIMIT,
        ag_args_fit={"random_seed": SEED},
    )
    fit_seconds = time.perf_counter() - t

    t = time.perf_counter()
    proba = predictor.predict_proba(test[FEATURES], as_multiclass=False)
    predict_seconds = time.perf_counter() - t

    board = predictor.leaderboard(silent=True)
    best = predictor.model_best
    out = {
        "autogluon_version": autogluon.tabular.__version__,
        "python": platform.python_version(),
        "presets": PRESETS,
        "time_limit_seconds": TIME_LIMIT,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_test": int(len(test)),
        "best_model": best,
        "believed_validation_auc": float(board.loc[board["model"] == best, "score_val"].iloc[0]),
        "test_order": test["order"].astype(int).tolist(),
        "test_proba": [float(p) for p in proba],
        "leaderboard": [
            {"model": r["model"], "score_val": float(r["score_val"]), "stack_level": int(r["stack_level"]),
             "fit_time": float(r["fit_time"]), "pred_time_val": float(r["pred_time_val"])}
            for _, r in board.iterrows()
        ],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "autogluon.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"AutoGluon {best}: believed AUC {out['believed_validation_auc']:.4f}, "
          f"fit {fit_seconds:.0f}s, {len(board)} models")


if __name__ == "__main__":
    main()
