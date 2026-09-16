---
name: reproduce-06-anomaly-detection
description: Reproduce and verify project 06, Credit Card Fraud Detection, from a fresh clone. Use when asked to rebuild, retrain or check this project.
---

# Reproduce Credit Card Fraud Detection

## Steps

From the repository root, install the kernel once: `pip install -e platform` and `npm install`. Then, inside the project directory:

1. `python ml/data.py`
2. `python ml/train.py`
3. `python -m uvicorn main:app --app-dir server --port 8006`
4. `npm run dev -w 06_anomaly_detection/client`

## Verify

- Training must finish with all runtime assertions passing (the last run recorded 3 of 3).
- The winner and its avg precision should match `artifacts/metrics.json`. Small differences are expected if the upstream dataset changed; compare the SHA-256 in `artifacts/provenance.json` first.
- `python ../scripts/capture_screenshots.py 06_anomaly_detection` must report every view rendered.
- `python ../scripts/generate_docs.py --check` must report no drift.

## Do not

- Hand-edit figures in README tables, `paper.md`, `abstract.md` or `article.md`; regenerate them.
- Replace the dataset with generated data. If that ever becomes necessary, set `synthetic=True` on the Provenance so the UI discloses it.
