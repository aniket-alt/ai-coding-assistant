---
name: reproduce-05-skills-lab
description: Reproduce and verify project 05, Data Science Skills Lab, from a fresh clone. Use when asked to rebuild, retrain or check this project.
---

# Reproduce Data Science Skills Lab

## Steps

From the repository root, install the kernel once: `pip install -e platform` and `npm install`. Then, inside the project directory:

1. `python ml/train.py`
2. `python -m uvicorn main:app --app-dir server --port 8005`
3. `npm run dev -w 05_skills_lab/client`

## Verify

- Training must finish with all runtime assertions passing (the last run recorded 3 of 3).
- The winner and its how much it flatters should match `artifacts/metrics.json`. Small differences are expected if the upstream dataset changed; compare the SHA-256 in `artifacts/provenance.json` first.
- `python ../scripts/capture_screenshots.py 05_skills_lab` must report every view rendered.
- `python ../scripts/generate_docs.py --check` must report no drift.

## Do not

- Hand-edit figures in README tables, `paper.md`, `abstract.md` or `article.md`; regenerate them.
- Replace the dataset with generated data. If that ever becomes necessary, set `synthetic=True` on the Provenance so the UI discloses it.
