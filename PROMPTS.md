# Prompt log

Every prompt given to the AI coding assistant ([Claude Code](https://claude.com/claude-code))
while building this repository, in order and verbatim. Where the assistant
offered a multiple-choice question, the choice made is recorded as a
**decision**. The only edit is a redacted local file path.

Each entry notes what the prompt produced and, where one exists, the commit.

---

## Session 1 — understanding the assignment and planning

### 1. The assignment

```text
Please take on the role of a highly skilled specialist and an expert in the topics related to the given assignment. I’ve provided you with my assignment today and the objective of the assignment. Your job is to help me complete the assignment. First tell me what is to be submitted as part of this assignment.
I have included all the documents that the links lead to.
ASSIGNMENT:
Data science full stack with ai coding assistant
Replicate the data science experiments i did in your favorite coding assistant - i provided the prompts in the repository
https://github.com/dlmastery/data_science_examples
make sure you publish them in github repo and walk thru these in a youtube video and upload and link to README.md

Note: use your creativity and not need to do exactly like above - improvise if possible. the minimum bar is above
Checking all artifacts
Publish a link to YouTube video which walkthrough the code and ux of each of the projects thoroughly in readme.md

I provided prompts.md with what prompts I used

Use your creativity to do more in each app and explain

thanks
vijay eranti (You can find the github repo downloaded at: <local path redacted>)
```

**Produced:** a checklist of deliverables — a public repository with sixteen
projects, a per-project artifact set (README, design doc, paper, abstract,
article, audit report, skills, screenshots), root documentation, and a YouTube
walkthrough linked from the README.

### 2. Approach

```text
So how do we go about it?
```

**Produced:** an inspection of the source repository before any plan. It found
that several headline results were measured on generated data — the SPY
project's prices come from geometric Brownian motion starting on a future date,
and its "Chronos-T5" model is a NumPy simulation. That finding became the
portfolio's central decision: use real data throughout.

**Decisions:**
- Rigor: **Real data everywhere**
- GitHub: **You create repo, I push via git**
- Video: **Decide later**

The assistant then began building the platform without being asked to, which
prompted the next question.

### 3. Course correction

```text
what just happened? What did you do/
```

**Produced:** a plain inventory of every change made to the machine — folders
created, packages installed, data downloaded, one local commit — and of what had
*not* happened (nothing pushed, the source repository untouched), with the
option to discard it all.

### 4–7. Repository setup

```text
So how would that look? So the ds-fullstack-ai will be pushed to the github repo?
```

```text
Looks good. First, help me create the github repo and then we can continue building
```

```text
Just give me the steps to create, I'll do it
```

```text
https://github.com/aniket-alt/ai-coding-assistant - This is the link, now you can continue building the project
```

**Produced:** the repository layout, step-by-step creation instructions, and
the first push. Building resumed with the shared web kernel.

### 8. Status

```text
What's the status now?
```

### 9. Attribution

```text
Contributors
2
 (2)
@claude
claudeClaude
@aniket-alt
aniket-altAniket Anil Naik I see that on GitHub it says Claude as one of the Contributors. I am not sure I want that
```

**Produced:** an explanation — the assistant had been adding a co-author
trailer to commit messages — and a saved preference never to add one again.

**Decision:** **Delete the GitHub repo, I re-push clean**

```text
https://github.com/aniket-alt/ai-coding-assistant - New repository
```

**Produced:** history rebuilt locally with identical content and clean commit
messages, then pushed to the recreated repository:
[`e0e0f07`](https://github.com/aniket-alt/ai-coding-assistant/commit/e0e0f07) platform kernel,
[`9444331`](https://github.com/aniket-alt/ai-coding-assistant/commit/9444331) web kernel.

### 10. Checking the deliverable

```text
Just cross-checking, I have to submit the github repo as my assignment right?
```

**Produced:** confirmation that the repository URL is the submission and the
video link belongs inside the README.

---

## Session 2 — the first three projects

### 11. Resuming

```text
Okay, now where are we on the assignment?
```

```text
Yes, just start.
```

**Produced:** the root README, packaging for the platform kernel, a Python 3.11
environment, and Project 01 end to end on 2,964,624 real NYC taxi records.
The defining decision — excluding meter distance, fare and arrival time because
none are known at booking — was made here.
[`2a67f88`](https://github.com/aniket-alt/ai-coding-assistant/commit/2a67f88)

### 12. Component API review

```text
Build the Project 01 React client, but build it as the first consumer of
@dsfs/ui, not as a standalone app.

Five components go in @dsfs/ui, driven by API data, reusable across all 16:
  MetricsPanel      — from /api/metrics
  ModelLeaderboard  — the tournament table, any project
  CrispDmWalkthrough — phases as data, not hardcoded
  AuditPanel        — findings from the leakage scanner
  PredictionForm    — schema-driven

Only two things are project-01-specific: the zone map and the trip
estimator view.

Import tokens from ui/tokens.css. Before writing components, show me the
component API you're proposing so I can check it generalizes to clustering
and forecasting, not just this project.
```

**Produced:** a proposed API for each component, tested on paper against
clustering and forecasting, plus an honest note that these components would fit
about eleven of the sixteen projects rather than all of them.

### 13. Approval with changes

```text
Approved, with four changes.

1. Baseline ownership. MetricsPanel.showBaselineLift and ModelLeaderboard
   both render baseline data — that's two components that can disagree
   about what the baseline is. Leaderboard owns baselines as rows;
   MetricsPanel reads the computed lift from the run, doesn't derive it.

2. showEnvironment must read provenance captured at TRAINING time from
   metrics.json — never the serving process's own environment. If the
   backend reports its own python/lib versions, they'll drift from
   whatever trained the model and we'd be publishing a false provenance
   claim. Confirm RunRecord already populates it that way; if not, fix
   the training side first.

3. Token path: keep @dsfs/ui/styles.css. Don't restructure.

4. Build the Chart wrapper now as a sixth primitive, before 01 ships.
   03, 12 and 06 all need plots and will each invent their own otherwise.
   Keep it thin — palette, direct labels, table fallback. It wraps a
   charting library, it isn't one.

Hard constraint on all six: no component computes its own aggregate.
Every number rendered must come from metrics.json or an API response,
so every published figure traces to an artifact.

Before building, check that the backend actually populates
run.audit.runtime_assertions with real executed checks. If it's an empty
array, fix that first — an audit panel with a nice empty state is the
same hollow certification we caught in the source repo.

Then build the six, then the zone map and estimator view on top.
```

**Produced:** the most consequential fix in the repository. The check requested
in the last paragraph found that `runtime_assertions` was a hardcoded list of
three strings — the checks had run, but the list would have survived deleting
them. The assertion ledger was built so that each check records what it
measured. The six shared components and the Project 01 client followed.
[`ff9b407`](https://github.com/aniket-alt/ai-coding-assistant/commit/ff9b407)

### 14. Project 06

```text
go ahead
```

**Produced:** fraud detection on 284,807 real card transactions. Along the way:
class reweighting was found to collapse LightGBM's average precision, the API
was found to be serving a different model from the one that won, and the chart
component was found not to render axis labels.
[`a3334b8`](https://github.com/aniket-alt/ai-coding-assistant/commit/a3334b8), [`e7b6737`](https://github.com/aniket-alt/ai-coding-assistant/commit/e7b6737)

### 15. Project 15

```text
go ahead
```

**Produced:** SPY forecasting on 2,931 real trading days, where a baseline wins
and the gradient-boosted model cannot be distinguished from predicting zero.
This project also exposed a silent false pass in the shared leakage auditor.
[`3d83606`](https://github.com/aniket-alt/ai-coding-assistant/commit/3d83606)

---

## Session 3 — documentation

### 16. Status

```text
where are we now?
```

**Produced:** a status report that flagged two gaps a grader would find: the
finished projects lacked most of their documents, and the root README linked to
files that did not exist.

### 17. Generated documentation

```text
Go ahead
```

**Produced:** `scripts/generate_docs.py`, which writes each project's paper,
abstract, article, design document, audit report and reproduction skill from
its artifacts, fills the result tables in each README, and fails in `--check`
mode if anything drifts. Also: platform tests, a correction to how elapsed time
is reported in Project 06, and this file.

### 18. Interruption

```text
Sorry, unknowingly I hit close.
```

**Produced:** confirmation that nothing was lost, and a finding from the
interrupted CI check: the repository was private, so graders would have seen a
404.

### 19. Visibility fixed

```text
try now
```

**Produced:** the first CI run, which failed. Windows and Linux sort file paths
differently, so generated documents differed between the local machine and CI.
Paths now sort identically everywhere, and CI passed.
[`799695b`](https://github.com/aniket-alt/ai-coding-assistant/commit/799695b)

---

## Session 4 — projects 03, 04 and 12

### 20. Next batch

```text
Go ahead
```

**Produced:** three projects, each confirmed by CI on GitHub before the next
began.

- **03 Customer Segmentation.** Segments are judged on the following year's
  behaviour, and the decades-old RFM scoring rule beat every clustering
  algorithm. Also fixed: charts drawn mid-animation misaligned their bars.
  [`3d7f8d5`](https://github.com/aniket-alt/ai-coding-assistant/commit/3d7f8d5)
- **04 Market Basket Mining.** Rules are judged on the following Christmas
  season, and plain item co-occurrence beat every mined rule set. Also fixed:
  results that changed between identical runs because of Python's randomised
  string hashing, and a leaderboard crash on metrics some models lack.
  [`39e3212`](https://github.com/aniket-alt/ai-coding-assistant/commit/39e3212)
- **12 Energy Demand Forecasting.** Day-ahead forecasts with every lag at least
  the horizon, plus a measured leak: a too-recent lag would have claimed a much
  better score. Forecast evaluation moved into the shared platform, leaving
  project 15's results byte-identical.
  [`2c4a097`](https://github.com/aniket-alt/ai-coding-assistant/commit/2c4a097)

---

## Session 5 — projects 11, 13 and 10

### 21. Retry

```text
try again
```

The previous batch had stopped partway; work resumed from the
last verified step.

### 22. Status

```text
Where are we now?
```

### 23. Next batch

```text
Carry on with the next batch
```

**Produced:** three projects, each checked locally with the full test, audit,
documentation and type-check suite before pushing.

- **11 Portfolio Audit.** A benchmark of the repository's own leakage auditor on
  fourteen planted variants. Static and runtime checks together caught five of
  eleven leaks with no false alarms; the misses are reported with how much each
  inflated the score.
  [`ada6490`](https://github.com/aniket-alt/ai-coding-assistant/commit/ada6490)
- **13 MLOps & Governance.** Retraining policies judged month by month on
  unseen months. Retraining on all history won, and the drift alarm never fired
  even though never retraining was clearly worse. Reviewing the screenshots
  caught chart colours that did not match the leaderboard, which led to stable
  per-model colours in the shared components, and a "Retrains" column that
  counted the initial fit.
  [`3ee3e28`](https://github.com/aniket-alt/ai-coding-assistant/commit/3ee3e28)
- **10 CRISP-DM Curriculum.** Seven lessons on the Adult census. Removing sex
  and race barely changed disparate impact, because the remaining features
  predict sex almost perfectly. Checking the README against the artifacts
  corrected an assumed claim about the outliers, and the repository audit
  learned that a cross-sectional project can declare it has no time order.
  [`06afa32`](https://github.com/aniket-alt/ai-coding-assistant/commit/06afa32)

---

## What made prompting work

**Review before build.** Prompts 12 and 13 asked for the design before the code
and then changed it. That was cheaper than any later fix, and the instruction to
*verify* rather than trust (the check of `runtime_assertions`) caught a defect
that would otherwise have shipped looking correct.

**Constraints stated as invariants.** "No component computes its own aggregate"
is checkable against every file, where "make the numbers accurate" is not.

**Short prompts only after conventions exist.** "go ahead" produced a complete
project because the platform, the artifact contract and the verification steps
were already in place. The same two words at the start of the session would
have produced something much less predictable.

**Correcting the assistant early.** Prompt 3 stopped work that had run ahead of
the plan, and prompt 9 removed an attribution default before it spread through
the history. Both were cheaper to fix at that point than later.
