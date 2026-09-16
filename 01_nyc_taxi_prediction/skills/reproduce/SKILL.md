---
name: reproduce-01-nyc-taxi-prediction
description: Reproduce and verify project 01, NYC Taxi Trip Duration, from a fresh clone. Use when asked to rebuild, retrain or check this project.
---

# Reproduce NYC Taxi Trip Duration

## Steps

From the repository root, install the kernel once: `pip install -e platform` and `npm install`. Then, inside the project directory:

1. `python ml/data.py`
2. `python ml/build_zone_map.py`
3. `python ml/train.py`
4. `python -m uvicorn main:app --app-dir server --port 8001`
5. `npm run dev -w 01_nyc_taxi_prediction/client`

## Verify

- Training must finish with all runtime assertions passing (the last run recorded 3 of 3).
- The winner and its rmse (min) should match `artifacts/metrics.json`. Small differences are expected if the upstream dataset changed; compare the SHA-256 in `artifacts/provenance.json` first.
- `python ../scripts/capture_screenshots.py 01_nyc_taxi_prediction` must report every view rendered.
- `python ../scripts/generate_docs.py --check` must report no drift.

## Do not

- Hand-edit figures in README tables, `paper.md`, `abstract.md` or `article.md`; regenerate them.
- Replace the dataset with generated data. If that ever becomes necessary, set `synthetic=True` on the Provenance so the UI discloses it.
