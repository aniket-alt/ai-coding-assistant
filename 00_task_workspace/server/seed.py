"""
Project 00 -- demonstration tasks.

A fixed, seeded set of tasks so an empty workspace has something to show. It is
generated, not real usage data, and the project's provenance says so. Dates are
placed relative to the day the workspace is first started, so analytics always
have a recent history.
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta, timezone

from store import Store

TITLES = [
    ("Draft the quarterly churn report", ["work", "analytics"]), ("Review pull request for the feature store", ["work", "code"]),
    ("Book dentist appointment", ["personal"]), ("Prepare slides for model review", ["work"]),
    ("Clean up the experiment tracking bucket", ["work", "infra"]), ("Renew passport", ["personal", "admin"]),
    ("Write unit tests for the parser", ["code"]), ("Plan weekend hike", ["personal"]),
    ("Read the conformal prediction paper", ["learning"]), ("Fix flaky integration test", ["code"]),
    ("Update the on-call runbook", ["work", "infra"]), ("Pay electricity bill", ["personal", "admin"]),
    ("Label 200 support tickets", ["work", "data"]), ("Rotate API keys", ["infra"]),
    ("Call the landlord about the heater", ["personal"]), ("Benchmark the new tokenizer", ["code", "learning"]),
    ("Sketch dashboard layout", ["design"]), ("Back up photo library", ["personal"]),
    ("Interview prep: probability questions", ["learning"]), ("Migrate cron jobs to the scheduler", ["infra"]),
    ("Write retro notes", ["work"]), ("Order new keyboard", ["personal"]),
    ("Check data drift report", ["work", "data"]), ("Refactor feature pipeline", ["code"]),
]


def seed(store: Store, today: date, n_done: int = 34, seed_value: int = 7) -> None:
    rng = random.Random(seed_value)
    stamp = lambda d, h: datetime.combine(d, time(h, rng.randint(0, 59)), tzinfo=timezone.utc).isoformat(timespec="milliseconds")
    # Completed history over the last five weeks, for the heatmap and streaks.
    for i in range(n_done):
        title, tags = TITLES[i % len(TITLES)]
        created = today - timedelta(days=rng.randint(3, 34))
        done_day = min(today, created + timedelta(days=rng.randint(0, 6)))
        est = rng.choice([15, 30, 45, 60, 90])
        task = store.create({"title": title, "tags": tags,
                             "priority": rng.choice([1, 2, 2, 3, 3, 3, 4]), "estimate_min": est,
                             "status": "done", "important": rng.random() < 0.5, "urgent": rng.random() < 0.3},
                            created_at=stamp(created, 9))
        store._db.execute("UPDATE tasks SET completed_at = ?, spent_min = ? WHERE id = ?",
                          (stamp(done_day, rng.randint(10, 18)), max(5, int(est * rng.uniform(0.6, 1.8))), task["id"]))
    # Open work across the board and the matrix.
    for i, (title, tags) in enumerate(TITLES[:14]):
        status = ["todo", "todo", "doing", "review"][i % 4]
        due_offset = rng.choice([-2, -1, 0, 0, 1, 2, 3, 5, 8, None])
        store.create({"title": title, "tags": tags, "status": status, "priority": rng.choice([1, 2, 3, 3, 4]),
                      "important": i % 3 != 2, "urgent": i % 2 == 0,
                      "due": (today + timedelta(days=due_offset)).isoformat() if due_offset is not None else None,
                      "estimate_min": rng.choice([None, 20, 30, 45, 60, 120])},
                     created_at=stamp(today - timedelta(days=rng.randint(0, 5)), 8))
    store._db.commit()
