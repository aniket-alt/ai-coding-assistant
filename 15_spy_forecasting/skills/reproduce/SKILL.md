---
name: reproduce-15-spy-forecasting
description: Reproduce and verify project 15, SPY Probabilistic Forecasting, from a fresh clone. Use when asked to rebuild, retrain or check this project.
---

# Reproduce SPY Probabilistic Forecasting

## Steps

From the repository root, install the kernel once: `pip install -e platform` and `npm install`. Then, inside the project directory:

1. `python ml/data.py`
2. `python ml/train.py --horizon 1`
3. `python -m uvicorn main:app --app-dir server --port 8015`
4. `npm run dev -w 15_spy_forecasting/client`

## Verify

- Training must finish with all runtime assertions passing (the last run recorded 2 of 2).
- The winner and its pinball loss should match `artifacts/metrics.json`. Small differences are expected if the upstream dataset changed; compare the SHA-256 in `artifacts/provenance.json` first.
- `python ../scripts/capture_screenshots.py 15_spy_forecasting` must report every view rendered.
- `python ../scripts/generate_docs.py --check` must report no drift.

## Do not

- Hand-edit figures in README tables, `paper.md`, `abstract.md` or `article.md`; regenerate them.
- Replace the dataset with generated data. If that ever becomes necessary, set `synthetic=True` on the Provenance so the UI discloses it.
