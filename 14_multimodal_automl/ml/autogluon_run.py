"""
Project 14 -- the AutoGluon contestant.

    ../.venv-automl/Scripts/python ml/autogluon_run.py

Reads the fused features train.py exported (CLIP image embedding, 128-dimension
name SVD and catalogue fields), trains AutoGluon on the training and validation
products, and writes its test-set probabilities to artifacts/autogluon.json for
train.py to score like every other contestant. It runs in .venv-automl because
AutoGluon pins older pandas and pyarrow than the rest of the repository.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import autogluon.tabular
import pandas as pd
from autogluon.tabular import TabularPredictor

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "autogluon_model"
TIME_LIMIT = 900
PRESETS = "best_quality"
SEED = 42


def main() -> None:
    feats = pd.read_parquet(ARTIFACTS / "work" / "fused_features.parquet")
    classes = sorted(feats["label"].unique())
    train = feats[feats["split"] != "test"].drop(columns=["split", "id"])
    test = feats[feats["split"] == "test"]
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    t = time.perf_counter()
    predictor = TabularPredictor(label="label", eval_metric="f1_macro", path=str(MODEL_DIR), verbosity=2).fit(
        train, presets=PRESETS, time_limit=TIME_LIMIT, ag_args_fit={"random_seed": SEED})
    fit_seconds = time.perf_counter() - t
    proba = predictor.predict_proba(test.drop(columns=["split", "id", "label"]))[classes]
    board = predictor.leaderboard(silent=True)
    out = {
        "autogluon_version": autogluon.tabular.__version__, "presets": PRESETS, "time_limit_seconds": TIME_LIMIT,
        "fit_seconds": fit_seconds, "best_model": predictor.model_best, "models_trained": int(len(board)),
        "classes": classes, "test_ids": test["id"].astype(int).tolist(), "test_proba": proba.to_numpy().round(6).tolist(),
    }
    (ARTIFACTS / "autogluon.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"AutoGluon {predictor.model_best}: {len(board)} models, fit {fit_seconds:.0f}s")


if __name__ == "__main__":
    main()
