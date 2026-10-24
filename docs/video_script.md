# Video walkthrough — recording script

One video, sixteen chapters, roughly **38–42 minutes**. Every figure below is
taken from the committed `artifacts/metrics.json` of the project it belongs to,
so what you say on camera matches what the repository shows.

**Before recording**

```bash
python ml/train.py            # only if artifacts are missing; otherwise skip
python -m uvicorn main:app --app-dir server --port 80NN
npm run dev -w NN_project/client
```

Each chapter has the same shape: **open the app → show the one thing that makes
the project worth looking at → show the code behind it → say the number.**
Resist narrating every tab; the README does that.

**Two habits that make the whole video land**

1. Say the baseline before the model. "A lookup table gets 7.51; the model gets 5.35."
2. When a number is unimpressive or negative, say so plainly. That is the
   portfolio's argument, and the recording is where it is most convincing.

---

## 00 · Opening (0:00 – 1:30)

> This is a sixteen-project data science portfolio, built with an AI coding
> assistant over about two weeks. Every project is a real dataset, a model
> tournament with baselines, a CRISP-DM lifecycle, a leakage audit, and a React
> front end on a shared design system.
>
> One rule holds it together: **every published number traces back to real data
> and a run that actually happened.** Nothing here is invented, and where a
> result is disappointing, the README says so.

**Show:** the repository root README — the index table, then scroll to the
platform section. Point out that `platform/` is about a third of the engineering
effort, which is why sixteen apps were tractable.

**Say:** CI runs the tests, the leakage audit, and a check that every generated
document still matches its artifacts, on every push.

---

## 01 · NYC Taxi Trip Prediction (1:30 – 4:00)

**Port 8001 / 5101 · Tabs: Trip estimator · Model tournament · CRISP-DM · Audit**

**Show first:** the Trip estimator. Pick a pickup and dropoff zone on the map,
move the time, watch the estimate change.

> 2.96 million January 2024 taxi trips. The question is what a rider can be told
> **at the moment they book** — so the feature rule is that nothing recorded
> after the meter starts may be used. Trip distance is the obvious trap: it is
> only known once the trip is over.

**Numbers:** global median baseline **11.77 minutes RMSE**; a zone-pair lookup
table **7.51**; LightGBM **5.35**. XGBoost is 5.35 too — effectively a tie.

**Then show:** `ml/features.py`, and the assertions in the Audit tab:
`assert_temporal_split`, `assert_disjoint`, `assert_no_target_leak`.

> The audit tab is not decoration. Those three checks ran during training, and
> the detail text is their measurement, not a label.

---

## 02 · Nano LLM Transformer (4:00 – 7:00)

**Port 8002 / 5102 · Tabs: Story studio · Training & baselines · Ablations · Attention**

**Show first:** the Story studio. Type a prompt, generate a story live.

> An 8.7-million-parameter transformer written from scratch — rotary embeddings,
> SwiGLU, RMSNorm — trained for 26 minutes on a laptop GPU on TinyStories.

**Numbers:** validation perplexity **4.03**, against **34.0** for an interpolated
bigram and **358** for unigram frequencies.

**Then the Ablations tab — this is the chapter's real content:**

> Each component was swapped for its older equivalent. But a single run tells you
> nothing, so the reference was trained twice with different seeds, and that gap —
> **0.006 nats per token** — is the noise floor. Rotary embeddings and SwiGLU beat
> it; **replacing RMSNorm with LayerNorm did not**, so on this scale I can't claim
> it mattered.

**Finish on honesty:** the KV cache is verified correct — identical greedy
continuations — but **it was not faster** at this context length: 4.7 ms per
token against 4.2 without. The README says so.

---

## 03 · Customer Segmentation (7:00 – 9:00)

**Port 8003 / 5103**

**Show first:** the Segments view, then assign a customer.

> A million invoice lines from Online Retail II. Most segmentation demos stop at
> a silhouette score, which only says the clusters are tidy. This one asks a
> harder question: **do the segments predict what customers do next year?**

**Numbers:** the metric is ω² on next-year behaviour. The decades-old **RFM
scoring rule wins at 0.306**; the best k-means is **0.284**; random segments
**−0.0001**.

> The classical rule beat every clustering algorithm I threw at it. That is the
> result, and it is in the README as the headline.

---

## 04 · Market Basket Mining (9:00 – 11:00)

**Port 8004 / 5104 · Tabs: Build a basket · Rules · Recommender tournament**

**Show first:** build a basket, watch recommendations appear.

> Association rules mined on one season, then judged on the **following**
> Christmas season — not on the data they were mined from.

**Numbers:** hit rate at 5 — random products **0.1%**, bestsellers **1.9%**,
mined FP-Growth rules **3.3% to 14.3%** depending on thresholds, and plain
**item co-occurrence 18.6%**, which beats every mined rule set.

> Four assertions ran here, including one that no invoice appears on both sides
> of the split, and one that the item being predicted was genuinely hidden.

---

## 05 · Data Science Skills Lab (11:00 – 14:00)

**Port 8005 / 5105 · Tabs: Catalogue · What flatters most · Datasets**

> A catalogue of techniques where every entry is a **measurement, not advice**.
> Each one runs the careless way and the careful way on a real dataset, 25 seeds,
> with a bootstrap interval on the paired difference.

**Show:** the Catalogue, click into *Oversampling before the split*, then run it
live on fewer seeds and watch the interval widen.

**Numbers:** six of twelve entries measurably flatter the careless version —
reading accuracy on an imbalanced target **+0.221**, oversampling before the
split **+0.110 AUC**, threshold chosen on the test set **+0.035 F1**, target
encoding without out-of-fold means **+0.022**.

**The pair to dwell on:**

> The same mistake twice. Oversampling before the split costs **0.110 AUC** with
> five-nearest-neighbours, which finds a duplicate at distance zero — and
> **nothing measurable** with a logistic regression, which has nowhere to store
> one. How much a leak costs is a property of the pipeline, not the mistake.

**And the honest half:** five entries show no difference the experiment can
resolve. Imputing before the split moves the score by **2e-05**; scaling by
**1e-16**. Those are reported, not hidden.

---

## 06 · Anomaly Detection (14:00 – 16:00)

**Port 8006 / 5106 · Tabs: Transaction scorer · Review budget · Detector tournament**

**Show first:** the Review budget view — this is the one that lands.

> 284,807 card transactions, 1 fraud in every 578. The interesting question is
> not AUC, it is: **if your review team can look at 100 transactions a day, how
> much fraud do you catch?**

**Numbers:** average precision — predicting all legitimate **0.001**, Isolation
Forest **0.033**, Local Outlier Factor **0.0007**, and plain **logistic
regression 0.755**.

> The purpose-built anomaly detectors lost badly to a supervised linear model.
> When you have labels, use them.

At a budget of 25 alerts, precision is **96%** and recall **19%** — the trade-off
the budget curve shows directly.

---

## 07 · AutoML Stacking (16:00 – 19:00)

**Port 8007 / 5107 · Tabs: Tournament · Score a client · Inside the stack · Shift & leak**

> 41,188 bank marketing calls. Everything trains on calls before 2010 and is
> scored on 2010 — the year it would have been deployed into, not a random slice.

**Numbers:** a **single tuned LightGBM wins at 0.697** AUC. AutoGluon, given 15
minutes and 13 models, gets **0.683**; my hand-built Caruana stack **0.680**. The
paired bootstrap says the tuned model is genuinely better than both.

**The point of the chapter — the Shift & leak tab:**

> Every method overestimated itself. Each one's own validation promised more than
> 2010 delivered, and AutoGluon's estimate was the most optimistic of them.
> And `duration` — the length of the call — is a documented leak: include it and
> cross-validation reports **0.948**, a model that looks near-perfect and cannot
> be used, because you don't know how long a call lasts before you make it.

**Mention:** AutoGluon runs in a separate virtual environment, because installing
it would have downgraded pandas for every other project.

---

## 08 · DS Visual Foundations (19:00 – 22:00)

**Port 8008 / 5108 · Tabs: Probability · Evaluation · Gradient descent · Backpropagation · Quiz**

**Show:** type a message into the Naive Bayes lesson and watch each word's
evidence; then the threshold slider; then run gradient descent live.

> Four lessons on real data — SMS spam and power-plant readings — where every
> algorithm is **written out by hand and then checked against a reference**.

**Numbers worth saying:** the written-out Naive Bayes matches scikit-learn to
**1.6e-14**; gradient descent reaches the exact least-squares solution to
**1.1e-16**; hand-derived backpropagation matches finite differences to **1.1e-07**
across all 49 parameters.

**The nicest demonstration:** the stability limit.

> The largest stable learning rate is **0.41** on scaled features and **9.7e-07**
> on raw ones. At 99% of the limit it converges; at 101% it diverges. The theory
> predicted exactly where it breaks, and you can push the slider past it on camera.

---

## 09 · FlowForge DAG Engine (22:00 – 25:00)

**Port 8009 / 5109 · Tabs: Run a workflow · Cycle detection · Evidence · Benchmarks · Type design**

**Show first:** run the workflow, watch nodes execute in parallel, then tick the
box to add a back edge and watch validation reject it with a concrete cycle.

> A typed workflow engine in TypeScript, run by Node in the tests and by the
> browser in the app, from one source.

**The chapter's argument is the Evidence tab:**

> **77,531 property checks** over random graphs, **zero failures**. But tests that
> pass prove nothing until you've seen them fail — so ten realistic bugs were
> planted in copies of the engine, and **all ten were caught**.

**And the bug worth telling:**

> The first executor was correct and quadratic — the benchmark showed it. The
> rewrite is linear, **1,500× faster at 3,000 nodes**. I kept the slow one as an
> oracle and compared them on random workflows, and **that comparison found a
> real bug in the rewrite**: a retrying task could be started twice at once.

---

## 10 · CRISP-DM Curriculum (25:00 – 28:00)

**Port 8010 / 5110 · Tabs: seven lessons + Audit**

**Go straight to lesson 4, Models & fairness.**

> 48,842 people from the 1994 census. The central question: if a model never sees
> a person's sex, is it fair?

**Numbers:** LightGBM reaches **0.930 AUC**. Its disparate impact for women is
**0.32**. Remove sex and race from the inputs entirely and it becomes **0.34** —
essentially unchanged.

> Because the remaining features still carry sex: a model trained on them
> predicts it with **0.935 AUC**. Relationship status alone does much of it.
> "Fairness through unawareness" is tested here rather than assumed, and it fails.

**Also worth 20 seconds:** the isolation forest flagged outliers, and 83% of them
simply have a capital gain or loss. Unusual is not the same as wrong.

---

## 11 · Portfolio Audit (28:00 – 30:30)

**Port 8011 / 5111 · Tabs: Leak benchmark · Detector tournament · Portfolio**

> This project points the portfolio's own leakage auditor at itself. Fourteen
> variants of one fraud pipeline — **eleven deliberate leaks and three clean
> controls** — and the question is how many the auditor catches.

**Numbers:** static code audit alone **0.591** balanced accuracy; runtime
assertions alone **0.682**; both together **0.727**, catching **5 of 11 leaks with
no false alarms** on the controls.

> It misses six. The biggest miss is a noisy copy of the label, worth **+0.159**
> average precision. I would rather publish a tool's measured recall than claim
> it catches everything.

---

## 12 · Energy Demand Forecasting (30:30 – 33:00)

**Port 8012 / 5112 · Tabs: Forecast · Load patterns · The leak · Tournament**

**Show:** the day-ahead fan chart.

> Four years of one household's electricity, a day-ahead probabilistic forecast,
> scored by pinball loss with walk-forward refits.

**Numbers:** repeating yesterday **0.2154**; a four-week hourly average
**0.1750**; LightGBM quantile **0.1428** — **33.7% better than yesterday**, and
significant in a paired bootstrap.

**The leak tab:**

> Use a one-to-three-hour lag instead of respecting the 24-hour horizon and the
> loss improves to **0.1104 — 22.7% better, and entirely fake**, because at the
> moment you forecast tomorrow evening, the last hour doesn't exist yet. It is
> measured and reported, not just avoided.

---

## 13 · MLOps & Governance (33:00 – 35:30)

**Port 8013 / 5113 · Tabs: Drift & retraining · Explain a prediction · Registry · Load test**

> A model trained in January 2024 and served through July. How fast does it go
> stale, and would drift monitoring notice?

**Numbers:** never retraining **7.05 RMSE**; retraining monthly on all history
**6.40** — about 9% better.

**The finding to lead with:**

> The PSI drift alarm **never fired** — its largest reading was 0.038 against a
> 0.1 threshold — and yet never retraining was clearly worse. Watching the inputs
> is not a substitute for measuring the errors once the answers arrive.

**Then:** a served prediction with its TreeSHAP contributions, and the load test —
about **25 requests per second** on one worker, where extra concurrency only adds
queueing.

---

## 14 · Multimodal AutoML (35:30 – 38:30)

**Port 8014 / 5114 · Tabs: Which source wins? · The name leak · Products · Search photos**

**Show first:** the Products gallery and the photo search — type "red leather
handbag".

> 44,000 fashion products with a photo, a name and catalogue fields. Which source
> of information actually matters?

**The name leak tab is the chapter:**

> **78% of product names contain their own category word** — "Men Navy Blue
> Shirt" is labelled Shirts. A name model scores **0.969 macro-F1** that way, and
> **0.592** once those words are removed. So the honest comparison masks them.

**Numbers:** image alone **0.818**; CLIP with no training at all **0.528**;
honest fusion of photo, masked name and fields **0.859** — significantly better
than the photo alone, **+2.5 points of accuracy**. AutoGluon on the same fused
features reaches 0.947 but reads the unmasked name, so it belongs with the
leak-affected row.

---

## 15 · SPY Forecasting (38:30 – 41:00)

**Port 8015 / 5115 · Tabs: Forecast · Backtest · Calibration · Tournament**

**Save this one for last — it is the most honest chapter in the portfolio.**

> Eleven years of SPY, a probabilistic next-day return forecast, walk-forward
> with purged splits.

**Numbers:** every model lands at a pinball loss of about **0.0027**. The winner
is **historical drift** — a constant. Against a random walk, LightGBM is
**inconclusive**; ridge, empirical quantiles and the EWMA model are **worse**.

> Nothing beat the trivial baseline by a margin the bootstrap can resolve. That is
> the correct result for daily equity returns, and it is the headline of the
> project rather than a footnote. A portfolio that shows a model winning here
> would be showing you a bug.

---

## Closing (41:00 – 42:00)

> Sixteen projects, one shared platform, every number traceable to a run.
> Along the way the honest answer was often the unimpressive one: a classical
> rule beating k-means, a tuned single model beating AutoML, a drift alarm that
> never fired, and a market that cannot be predicted. Those are in the READMEs
> with the same prominence as the wins.
>
> The prompts that produced all of it are in `PROMPTS.md`, including the
> corrections — the times the assistant claimed something the artifacts did not
> support, and what was done about it.

**Show:** `PROMPTS.md` briefly, then the CI badge / green run, then stop.

---

## YouTube chapter markers

Paste into the video description once the real timings are known. YouTube turns
these into chapters automatically if the first is `0:00`.

```
0:00 Introduction — one rule for the whole portfolio
1:30 01 NYC Taxi — predicting at booking time
4:00 02 Nano LLM — a transformer from scratch, ablated against seed noise
7:00 03 Customer Segmentation — judged on next year's behaviour
9:00 04 Market Basket — rules judged on a later season
11:00 05 Skills Lab — careless vs careful, measured
14:00 06 Anomaly Detection — fraud under a review budget
16:00 07 AutoML Stacking — AutoGluon vs one tuned model
19:00 08 Visual Foundations — algorithms checked against references
22:00 09 FlowForge — an engine with evidence
25:00 10 CRISP-DM Curriculum — fairness through unawareness, tested
28:00 11 Portfolio Audit — auditing my own auditor
30:30 12 Energy Forecasting — day-ahead quantiles and a measured leak
33:00 13 MLOps — retraining policies and a silent drift alarm
35:30 14 Multimodal — what each modality really adds
38:30 15 SPY Forecasting — when nothing beats the baseline
41:00 Closing
```

After uploading, replace the sixteen link targets at the bottom of the root
`README.md` with timestamped URLs, for example:

```markdown
[v01]: https://youtu.be/VIDEO_ID?t=90
```
