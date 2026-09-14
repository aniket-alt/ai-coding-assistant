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
| Documentation cannot drift | `paper.md`, `abstract.md`, the README tables and the live API **all read the same `metrics.json`** the training run wrote. Nothing is hand-typed |
| Leakage is caught, not asserted | An AST pass plus **runtime assertions that raise during training** — a critical finding fails CI |
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
| 03 | Customer Segmentation | Clustering, topological partitioning | Online Retail II | 8003 | 5103 | ⬜ |
| 04 | Market Basket Mining | Apriori, FP-Growth | Instacart | 8004 | 5104 | ⬜ |
| 05 | DS Skills Lab | Analytical skills catalogue | multiple | 8005 | 5105 | ⬜ |
| 06 | **Anomaly Detection** | Multi-backbone outlier scoring | Credit Card Fraud | 8006 | 5106 | ⬜ |
| 07 | AutoML Stacking | Multi-level stacking, ensembling | Adult / Covertype | 8007 | 5107 | ⬜ |
| 08 | DS Visual Foundations | Interactive maths simulators | — | 8008 | 5108 | ⬜ |
| 09 | FlowForge DAG Engine | Advanced TypeScript, Kahn's algorithm | — | 8009 | 5109 | ⬜ |
| 10 | CRISP-DM Curriculum | Seven-phase analytics, LSH | US Census | 8010 | 5110 | ⬜ |
| 11 | Enterprise DS Audit | Quality & leakage scorecard | this repo | 8011 | 5111 | ⬜ |
| 12 | Time Series Forecasting | Multi-horizon fans, ACF/PACF | Energy demand | 8012 | 5112 | ⬜ |
| 13 | CRISP-DM Audit Platform | Governance, explainability | NYC TLC | 8013 | 5113 | ⬜ |
| 14 | Multimodal AutoML | Vision–language–tabular fusion | multiple | 8014 | 5114 | ⬜ |
| 15 | **SPY Forecasting** | Probabilistic finance, walk-forward backtest | Yahoo Finance | 8015 | 5115 | ⬜ |

✅ complete · 🚧 in progress · ⬜ not started

**Bold** projects are the three built to full depth; the rest share their
architecture. This table is updated as work lands — it reflects what is actually
in the repository, not what is planned.

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
│   ├── leakage.py           AST audit + runtime assertions that raise
│   └── app.py               FastAPI factory: 8 standard endpoints per project
└── web/                     React + TypeScript kernel (@dsfs/ui)
    ├── styles/tokens.css    design system + colourblind-validated palette
    ├── types.ts             TS mirrors of the Python dataclasses
    ├── hooks/useApi.ts      typed fetch layer
    └── components/          ProvenanceBadge · CrispDmPanel · AuditPanel
                             Leaderboard · StatTile · MarkdownMath · AppShell
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

Each project additionally carries its own `README.md`, `DESIGN_DOC.md`,
`paper.md`, `abstract.md`, `article.md`, `audit_report.md` and screenshot tour.

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
