---
name: reproduce-14-multimodal-automl
description: Reproduce and verify project 14, Multimodal AutoML, from a fresh clone. Use when asked to rebuild, retrain or check this project.
---

# Reproduce Multimodal AutoML

## Steps

From the repository root, install the kernel once: `pip install -e platform` and `npm install`. Then, inside the project directory:

1. `python ml/train.py`
2. `../.venv-automl/Scripts/python ml/autogluon_run.py`
3. `python ml/train.py`
4. `python -m uvicorn main:app --app-dir server --port 8014`
5. `npm run dev -w 14_multimodal_automl/client`

## Verify

- Training must finish with all runtime assertions passing (the last run recorded 3 of 3).
- The winner and its macro-f1 should match `artifacts/metrics.json`. Small differences are expected if the upstream dataset changed; compare the SHA-256 in `artifacts/provenance.json` first.
- `python ../scripts/capture_screenshots.py 14_multimodal_automl` must report every view rendered.
- `python ../scripts/generate_docs.py --check` must report no drift.

## Do not

- Hand-edit figures in README tables, `paper.md`, `abstract.md` or `article.md`; regenerate them.
- Replace the dataset with generated data. If that ever becomes necessary, set `synthetic=True` on the Provenance so the UI discloses it.
