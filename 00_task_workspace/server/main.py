"""
Project 00 -- FastAPI application: the task workspace.

Every change is written to SQLite and then broadcast to every open tab as a
server-sent event carrying a sequence number and the server's timestamp, so a
client can detect a missed event and resynchronise, and latency can be
measured end to end.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dsfs.app import ProjectMeta, create_app

from parser import PRIORITY_NAMES, parse
from store import STATUSES, Conflict, Invalid, NotFound, Store

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("TASKS_DB", ROOT / "data" / "tasks.db"))

META = ProjectMeta(
    slug="00_task_workspace",
    number=0,
    title="Task Workspace",
    tagline="A live task workspace: every tab stays in sync, and no edit silently overwrites another.",
    domain="Full-stack · real-time sync · concurrency",
    api_port=8000,
    web_port=5100,
    dataset="Seeded demonstration tasks (generated)",
    techniques=(
        "server-sent events with sequence numbers",
        "optimistic concurrency with versions",
        "quick-entry parser with a labelled test set",
        "Kanban, Eisenhower matrix, calendar, focus timer",
    ),
)

app = create_app(META)
store = Store(DB_PATH)
if not store.list() and os.environ.get("TASKS_SEED", "1") == "1":
    from seed import seed

    seed(store, date.today())


class Broker:
    """
    Fan-out of change events to every connected tab, in one global order.

    Publishing touches asyncio queues, which are not thread-safe, so every
    endpoint that publishes is `async` and runs on the event loop; their
    SQLite writes are short and serialised by the store's lock.
    """

    def __init__(self) -> None:
        self.seq = 0
        self.subscribers: set[asyncio.Queue] = set()

    def publish(self, kind: str, payload: dict[str, Any], origin: str | None) -> dict[str, Any]:
        self.seq += 1
        event = {"seq": self.seq, "type": kind, "origin": origin,
                 "at": datetime.now(timezone.utc).timestamp() * 1000, **payload}
        for q in list(self.subscribers):
            q.put_nowait(event)
        return event


broker = Broker()


def today() -> date:
    return date.today()


class NewTask(BaseModel):
    text: str | None = Field(None, description="Quick-entry text, parsed on the server")
    title: str | None = None
    status: str = "todo"
    important: bool = False
    urgent: bool = False


class Patch(BaseModel):
    version: int
    fields: dict[str, Any]


class Focus(BaseModel):
    task_id: int | None = None
    minutes: int = Field(..., ge=1, le=180)


def _fail(exc: Exception):
    if isinstance(exc, Conflict):
        raise HTTPException(status_code=409, detail={"message": str(exc), "current": exc.current}) from exc
    if isinstance(exc, NotFound):
        raise HTTPException(status_code=404, detail="task not found") from exc
    if isinstance(exc, Invalid):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


@app.get("/api/parse", tags=["tasks"])
def get_parse(text: str) -> dict[str, Any]:
    """Preview what quick entry would create, resolved against today's date."""
    return parse(text, today()).to_dict()


@app.get("/api/tasks", tags=["tasks"])
def list_tasks() -> dict[str, Any]:
    """Every task, with the sequence number of the latest event, so a client knows where to resume."""
    return {"tasks": store.list(), "seq": broker.seq, "today": today().isoformat()}


@app.post("/api/tasks", tags=["tasks"])
async def create_task(body: NewTask, x_client_id: str | None = Header(None)) -> dict[str, Any]:
    try:
        if body.text:
            p = parse(body.text, today())
            fields = {"title": p.title, "priority": p.priority, "tags": p.tags, "due": p.due,
                      "estimate_min": p.estimate_min, "status": body.status, "important": body.important,
                      "urgent": body.urgent or p.priority == 1}
        else:
            fields = {"title": body.title, "status": body.status, "important": body.important, "urgent": body.urgent}
        task = store.create(fields)
    except Exception as exc:  # noqa: BLE001 -- mapped to HTTP errors
        _fail(exc)
    broker.publish("created", {"task": task}, x_client_id)
    return task


@app.patch("/api/tasks/{task_id}", tags=["tasks"])
async def patch_task(task_id: int, body: Patch, x_client_id: str | None = Header(None)) -> dict[str, Any]:
    """Update a task. `version` must be the version the change was based on, or the update is refused with 409."""
    try:
        task = store.update(task_id, body.version, body.fields)
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    broker.publish("updated", {"task": task}, x_client_id)
    return task


@app.delete("/api/tasks/{task_id}", tags=["tasks"])
async def delete_task(task_id: int, version: int, x_client_id: str | None = Header(None)) -> dict[str, Any]:
    try:
        task = store.delete(task_id, version)
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    broker.publish("deleted", {"task": task}, x_client_id)
    return {"deleted": task_id}


@app.post("/api/focus", tags=["focus"])
async def post_focus(body: Focus, x_client_id: str | None = Header(None)) -> dict[str, Any]:
    """Record a finished focus session, adding its minutes to the task."""
    try:
        task = store.add_focus(body.task_id, body.minutes)
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    if task:
        broker.publish("updated", {"task": task}, x_client_id)
    return {"task": task or None, "minutes": body.minutes}


@app.get("/api/events", tags=["live"])
async def events(request: Request) -> StreamingResponse:
    """Server-sent events: one `data:` line per change, in sequence order."""
    queue: asyncio.Queue = asyncio.Queue()
    broker.subscribers.add(queue)

    async def stream():
        try:
            yield f"event: hello\ndata: {json.dumps({'seq': broker.seq})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            broker.subscribers.discard(queue)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/analytics", tags=["analytics"])
def analytics(days: int = 35) -> dict[str, Any]:
    """Completions per day, streaks, workload by status and quadrant, and estimate accuracy."""
    tasks = store.list()
    t = today()
    done_days = Counter(datetime.fromisoformat(x["completed_at"]).date() for x in tasks if x["completed_at"])
    heat = [{"date": (t - timedelta(days=i)).isoformat(), "completed": done_days.get(t - timedelta(days=i), 0)}
            for i in range(days - 1, -1, -1)]
    streak = 0
    day = t if done_days.get(t) else t - timedelta(days=1)
    while done_days.get(day):
        streak += 1
        day -= timedelta(days=1)
    week = lambda offset: sum(done_days.get(t - timedelta(days=offset + i), 0) for i in range(7))
    open_tasks = [x for x in tasks if x["status"] != "done"]
    estimated = [x for x in tasks if x["status"] == "done" and x["estimate_min"] and x["spent_min"]]
    focus = store.focus_sessions()
    return {
        "today": t.isoformat(), "heatmap": heat, "streak_days": streak,
        "completed_this_week": week(0), "completed_last_week": week(7),
        "by_status": [{"status": s, "tasks": sum(1 for x in tasks if x["status"] == s)} for s in STATUSES],
        "by_priority": [{"priority": PRIORITY_NAMES[p], "open": sum(1 for x in open_tasks if x["priority"] == p)} for p in (1, 2, 3, 4)],
        "overdue": sum(1 for x in open_tasks if x["due"] and x["due"] < t.isoformat()),
        "estimate_accuracy": {
            "tasks": len(estimated),
            "estimated_min": sum(x["estimate_min"] for x in estimated),
            "spent_min": sum(x["spent_min"] for x in estimated),
        },
        "focus_minutes_week": sum(f["minutes"] for f in focus
                                  if datetime.fromisoformat(f["finished_at"]).date() > t - timedelta(days=7)),
        "live_subscribers": len(broker.subscribers),
    }


@app.get("/api/evidence", tags=["results"])
def evidence() -> dict[str, Any]:
    """Parser accuracy, live-update latency and conflict tests from the recorded verification run."""
    p = ROOT / "artifacts" / "metrics.json"
    if not p.exists():
        raise HTTPException(status_code=503, detail="artifacts/metrics.json not found -- run ml/train.py.")
    return json.loads(p.read_text(encoding="utf-8"))["extra"]
