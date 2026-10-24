# Data Science Full Stack, Built With an AI Coding Assistant

Sixteen full-stack data science applications — each one a real dataset, a real
model tournament, a CRISP-DM lifecycle, a leakage audit, and a React front end.
Built with [Claude Code](https://claude.com/claude-code) as the pair programmer,
with every prompt recorded in [`PROMPTS.md`](./PROMPTS.md).

> **▶ Video walkthrough:** _not yet recorded_ — one chaptered tour of the code and
> UX of all sixteen projects. Every row of the index below links to its chapter;
> see [Video walkthrough](#video-walkthrough).

---

## The one rule this portfolio runs on

**Every published number traces back to real data and a run that actually happened.**

That sounds obvious. It is also the single hardest thing to hold to across
sixteen projects, and it is where most portfolios quietly break: a README claims
an impressive score, and the code behind it trained on data the script invented
moments earlier.

So this repository enforces it structurally rather than by good intentions:

| Guarantee | How it is enforced |
|---|---|
| Data is real and citable | Every dataset carries a **provenance receipt** — source URL, licence, SHA-256, row count — generated at download time and rendered in the app's own UI |
| Documentation cannot drift | Each project's paper, abstract, article, design doc, audit report and README result tables are **generated from the `metrics.json` its training run wrote**, and CI fails if any of them drifts. Figures quoted in README prose are written by hand and checked against the same file |
| Leakage is caught, not asserted | **Runtime assertions that halt training** and record what they measured, plus a static AST audit that **fails CI** on any critical finding |
| Synthetic data must confess | If a project ever generates data, the provenance badge says so in red, in the UI, on the page |

If a project can't honour these, that's reported rather than hidden.

---

## Portfolio index

Ports are assigned so every project can run simultaneously.

| # | Project | Domain | Dataset | API | Web | Status | Video |
|---|---|---|---|:---:|:---:|:---:|:---:|
| 00 | [**Task Workspace**](./00_task_workspace) | Live sync, no lost edits, quick-entry grammar, all verified | Generated demo tasks (synthetic) | 8000 | 5100 | ✅ | [▶][v00] |
| 01 | [**NYC Taxi Trip Prediction**](./01_nyc_taxi_prediction) | Booking-time spatial regression | NYC TLC (2.96M trips) | 8001 | 5101 | ✅ | [▶][v01] |
| 02 | [**Nano LLM Transformer**](./02_nano_llm) | 8.7M-parameter transformer from scratch; ablations against seed noise | TinyStories V2 (2.72M stories) | 8002 | 5102 | ✅ | [▶][v02] |
| 03 | [**Customer Segmentation**](./03_customer_segmentation) | Clustering judged on next-year behaviour | Online Retail II (1,067,371 lines) | 8003 | 5103 | ✅ | [▶][v03] |
| 04 | [**Market Basket Mining**](./04_market_basket) | FP-Growth rules judged on a later season | Online Retail II (shared with 03) | 8004 | 5104 | ✅ | [▶][v04] |
| 05 | [**Data Science Skills Lab**](./05_skills_lab) | Techniques measured careless vs careful, with intervals | Four OpenML sets (Titanic, Credit, Diabetes, Blood) | 8005 | 5105 | ✅ | [▶][v05] |
| 06 | [**Anomaly Detection**](./06_anomaly_detection) | Fraud ranking under 1:578 imbalance | ULB Credit Card Fraud (284,807) | 8006 | 5106 | ✅ | [▶][v06] |
| 07 | [**AutoML Stacking**](./07_automl_stacking) | AutoGluon vs hand-built stack vs tuned model, on a later year | UCI Bank Marketing (41,188 calls) | 8007 | 5107 | ✅ | [▶][v07] |
| 08 | [**DS Visual Foundations**](./08_visual_foundations) | Four lessons, each algorithm checked against a reference | UCI SMS Spam + Power Plant | 8008 | 5108 | ✅ | [▶][v08] |
| 09 | [**FlowForge DAG Engine**](./09_flowforge_dag) | Typed engine proven by property, differential and mutation tests | Seeded random graphs (synthetic) | 8009 | 5109 | ✅ | [▶][v09] |
| 10 | [**CRISP-DM Curriculum**](./10_crispdm_curriculum) | Seven lessons, fairness through unawareness tested, LSH | Adult census (48,842 people) | 8010 | 5110 | ✅ | [▶][v10] |
| 11 | [**Portfolio Audit**](./11_portfolio_audit) | Benchmark of the leakage auditor itself | Fraud data with planted leaks | 8011 | 5111 | ✅ | [▶][v11] |
| 12 | [**Energy Demand Forecasting**](./12_energy_forecasting) | Day-ahead quantiles, measured leak | UCI household power (2.08M minutes) | 8012 | 5112 | ✅ | [▶][v12] |
| 13 | [**MLOps & Governance**](./13_mlops_governance) | Retraining policies, drift, SHAP, load test | NYC TLC (23.4M trips, Jan–Jul 2024) | 8013 | 5113 | ✅ | [▶][v13] |
| 14 | [**Multimodal AutoML**](./14_multimodal_automl) | Image, text and fields fused; the name leak measured | Fashion Product Images (44,072 products) | 8014 | 5114 | ✅ | [▶][v14] |
| 15 | [**SPY Forecasting**](./15_spy_forecasting) | Probabilistic returns, walk-forward, purged | SPY/VIX/TNX daily, 2015–2026 | 8015 | 5115 | ✅ | [▶][v15] |

✅ complete · 🚧 in progress · ⬜ not started

**Bold** projects are built to full depth; the rest share their architecture.
This table reflects what is actually in the repository, not what is planned.

---

## Video walkthrough

The assignment asks for a YouTube walkthrough of the code and UX of every
project, linked from this README. It is **one video with a chapter per
project**, so each row below is a timestamped link into it rather than a
separate upload.

> **Status:** not yet recorded. Until it is, every ▶ below points back to this
> section, so the index has no broken links.

| # | Project | Chapter | Timestamp |
|---|---|:---:|---|
| 00 | Task Workspace | [▶][v00] | _pending_ |
| 01 | NYC Taxi Trip Prediction | [▶][v01] | _pending_ |
| 02 | Nano LLM Transformer | [▶][v02] | _pending_ |
| 03 | Customer Segmentation | [▶][v03] | _pending_ |
| 04 | Market Basket Mining | [▶][v04] | _pending_ |
| 05 | Data Science Skills Lab | [▶][v05] | _pending_ |
| 06 | Anomaly Detection | [▶][v06] | _pending_ |
| 07 | AutoML Stacking | [▶][v07] | _pending_ |
| 08 | DS Visual Foundations | [▶][v08] | _pending_ |
| 09 | FlowForge DAG Engine | [▶][v09] | _pending_ |
| 10 | CRISP-DM Curriculum | [▶][v10] | _pending_ |
| 11 | Portfolio Audit | [▶][v11] | _pending_ |
| 12 | Energy Demand Forecasting | [▶][v12] | _pending_ |
| 13 | MLOps & Governance | [▶][v13] | _pending_ |
| 14 | Multimodal AutoML | [▶][v14] | _pending_ |
| 15 | SPY Forecasting | [▶][v15] | _pending_ |

<!--
  To publish the walkthrough, replace the URL on the next line and give each
  chapter its start time; the targets below are the only place video links live.
-->

[v00]: #video-walkthrough
[v01]: #video-walkthrough
[v02]: #video-walkthrough
[v03]: #video-walkthrough
[v04]: #video-walkthrough
[v05]: #video-walkthrough
[v06]: #video-walkthrough
[v07]: #video-walkthrough
[v08]: #video-walkthrough
[v09]: #video-walkthrough
[v10]: #video-walkthrough
[v11]: #video-walkthrough
[v12]: #video-walkthrough
[v13]: #video-walkthrough
[v14]: #video-walkthrough
[v15]: #video-walkthrough

None of the six shared components computes an aggregate of its own. Every figure
they render arrives from `metrics.json` or an API response, so a number on screen
can always be traced to the run that produced it.

---

## The platform

Sixteen applications is only tractable because the shared parts are written
once. `platform/` is roughly a third of the engineering effort and the reason
each project is small.

```
platform/
├── dsfs/                    Python kernel
│   ├── provenance.py        dataset receipts (URL, licence, SHA-256, shape)
│   ├── datasets.py          content-addressed cache; download once, offline after
│   ├── crispdm.py           CRISP-DM as a data structure the training code fills
│   ├── metrics.py           RunRecord — the single source of truth for numbers
│   ├── leakage.py           AST audit + runtime assertions with a ledger
│   ├── docs.py              papers, design docs and tables generated from artifacts
│   ├── app.py               FastAPI factory: 8 standard endpoints per project
│   └── (tests/)             pytest suite, run in CI
└── web/                     React + TypeScript kernel (@dsfs/ui)
    ├── styles/tokens.css    design system + colourblind-validated palette
    ├── types.ts             TS mirrors of the Python dataclasses
    ├── hooks/useApi.ts      typed fetch layer
    └── components/          six API-driven components shared by every project:
                             MetricsPanel · ModelLeaderboard · CrispDmWalkthrough
                             AuditPanel · PredictionForm · Chart
                             plus AppShell · ProvenanceBadge · StatTile · MarkdownMath
```

A project therefore reduces to: a data loader, a training script, ~40 lines of
FastAPI, and its own distinctive views.

### On the colour palette

The categorical palette was **validated, not chosen by eye** — checked in both
light and dark against lightness band, chroma floor, adjacent-pair
colour-vision-deficiency separation, and contrast:

```
LIGHT  worst adjacent CVD ΔE 9.1 (protan)   PASS
DARK   worst adjacent CVD ΔE 8.4 (protan)   PASS
```

Three light-mode slots fall below 3:1 contrast, which obliges *relief* — so every
chart in the portfolio ships direct labels or a companion table view, and
identity is never carried by colour alone.

---

## Quick start

```bash
git clone https://github.com/aniket-alt/ai-coding-assistant.git
cd ai-coding-assistant

# Python kernel (editable, so projects import `dsfs` directly)
pip install -e platform

# Web kernel + all project clients
npm install
```

Then run any project:

```bash
cd 01_nyc_taxi_prediction
python ml/data.py          # fetch real data (cached; safe to re-run)
python ml/train.py         # train, audit, write artifacts/
python -m uvicorn server.main:app --port 8001
npm run dev -w 01_nyc_taxi_prediction/client
```

Datasets are **not** committed — they are re-downloaded on demand, and each
project's `provenance.json` records the SHA-256 so anyone can verify they
received identical bytes.

---

## Documentation

| Document | Contents |
|---|---|
| [`PROMPTS.md`](./PROMPTS.md) | Every prompt given to the AI assistant, in order, with commentary on what worked |
| [`IMPLEMENTATION_PLANS.md`](./IMPLEMENTATION_PLANS.md) | Per-project technical specifications |
| [`WALKTHROUGH.md`](./WALKTHROUGH.md) | Code-level tour across all projects |
| [`SKILLS.md`](./SKILLS.md) | Data science skills catalogue and where each is demonstrated |
| [`AUDIT_REPORT.md`](./AUDIT_REPORT.md) | Portfolio-wide leakage and reproducibility audit |

Each built project carries its own `README.md`, `DESIGN_DOC.md`, `paper.md`,
`abstract.md`, `article.md`, `audit_report.md`, `skills/reproduce/SKILL.md` and
a screenshot tour in light and dark themes.

Regenerate or verify every derived document with:

```bash
python scripts/generate_docs.py           # write
python scripts/generate_docs.py --check   # CI: fail if anything drifted
python scripts/run_audit.py               # CI: fail on critical leakage findings
```

---

## Provenance of this work

Assignment: replicate a 16-project data science portfolio using an AI coding
assistant, improvising beyond the original where possible. The original prompt
catalogue by Vijay Eranti is at
[`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples).

This is a reimplementation, not a fork. The deliberate departure is the one
stated at the top: real data and generated documentation, so that every claim on
these pages is checkable.

**Author:** Aniket Anil Naik · [MIT licensed](./LICENSE)
