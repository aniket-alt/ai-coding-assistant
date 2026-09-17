# Data Science Full Stack, Built With an AI Coding Assistant

Sixteen full-stack data science applications — each one a real dataset, a real
model tournament, a CRISP-DM lifecycle, a leakage audit, and a React front end.
Built with [Claude Code](https://claude.com/claude-code) as the pair programmer,
with every prompt recorded in [`PROMPTS.md`](./PROMPTS.md).

> **▶ Video walkthrough:** _link goes here once recorded_ — a chaptered tour of
> the code and UX of every project.

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

| # | Project | Domain | Dataset | API | Web | Status |
|---|---|---|---|:---:|:---:|:---:|
| 00 | Task Workspace | Full-stack reactive UX, SSE telemetry | — | 8000 | 5100 | ⬜ |
| 01 | [**NYC Taxi Trip Prediction**](./01_nyc_taxi_prediction) | Booking-time spatial regression | NYC TLC (2.96M trips) | 8001 | 5101 | ✅ |
| 02 | Nano LLM Transformer | Autoregressive transformer, RoPE + SwiGLU | TinyStories | 8002 | 5102 | ⬜ |
| 03 | [**Customer Segmentation**](./03_customer_segmentation) | Clustering judged on next-year behaviour | Online Retail II (1,067,371 lines) | 8003 | 5103 | ✅ |
| 04 | [**Market Basket Mining**](./04_market_basket) | FP-Growth rules judged on a later season | Online Retail II (shared with 03) | 8004 | 5104 | ✅ |
| 05 | DS Skills Lab | Analytical skills catalogue | multiple | 8005 | 5105 | ⬜ |
| 06 | [**Anomaly Detection**](./06_anomaly_detection) | Fraud ranking under 1:578 imbalance | ULB Credit Card Fraud (284,807) | 8006 | 5106 | ✅ |
| 07 | AutoML Stacking | Multi-level stacking, ensembling | Adult / Covertype | 8007 | 5107 | ⬜ |
| 08 | DS Visual Foundations | Interactive maths simulators | — | 8008 | 5108 | ⬜ |
| 09 | FlowForge DAG Engine | Advanced TypeScript, Kahn's algorithm | — | 8009 | 5109 | ⬜ |
| 10 | [**CRISP-DM Curriculum**](./10_crispdm_curriculum) | Seven lessons, fairness through unawareness tested, LSH | Adult census (48,842 people) | 8010 | 5110 | ✅ |
| 11 | [**Portfolio Audit**](./11_portfolio_audit) | Benchmark of the leakage auditor itself | Fraud data with planted leaks | 8011 | 5111 | ✅ |
| 12 | [**Energy Demand Forecasting**](./12_energy_forecasting) | Day-ahead quantiles, measured leak | UCI household power (2.08M minutes) | 8012 | 5112 | ✅ |
| 13 | [**MLOps & Governance**](./13_mlops_governance) | Retraining policies, drift, SHAP, load test | NYC TLC (23.4M trips, Jan–Jul 2024) | 8013 | 5113 | ✅ |
| 14 | Multimodal AutoML | Vision–language–tabular fusion | multiple | 8014 | 5114 | ⬜ |
| 15 | [**SPY Forecasting**](./15_spy_forecasting) | Probabilistic returns, walk-forward, purged | SPY/VIX/TNX daily, 2015–2026 | 8015 | 5115 | ✅ |

✅ complete · 🚧 in progress · ⬜ not started

**Bold** projects are built to full depth; the rest share their architecture.
This table reflects what is actually in the repository, not what is planned.

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
