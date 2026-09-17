"""
Project 00 -- task storage with optimistic concurrency.

SQLite, one table. Every task carries a version number. An update must name
the version it was based on; if another client has changed the task since,
the update is refused with a conflict instead of silently overwriting the
other change. Writes are serialised by a lock, so the version check and the
write happen as one step.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUSES = ("todo", "doing", "review", "done")
EDITABLE = {"title", "notes", "status", "priority", "important", "urgent", "tags", "due", "estimate_min"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'todo',
    priority INTEGER NOT NULL DEFAULT 3,
    important INTEGER NOT NULL DEFAULT 0,
    urgent INTEGER NOT NULL DEFAULT 0,
    tags TEXT NOT NULL DEFAULT '[]',
    due TEXT,
    estimate_min INTEGER,
    spent_min INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS focus_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    minutes INTEGER NOT NULL,
    finished_at TEXT NOT NULL
);
"""


class Conflict(Exception):
    def __init__(self, current: dict[str, Any]):
        super().__init__("task changed since it was read")
        self.current = current


class NotFound(Exception):
    pass


class Invalid(ValueError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Store:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    @staticmethod
    def _row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["tags"] = json.loads(d["tags"])
        d["important"], d["urgent"] = bool(d["important"]), bool(d["urgent"])
        return d

    @staticmethod
    def _validate(fields: dict[str, Any]) -> dict[str, Any]:
        unknown = set(fields) - EDITABLE
        if unknown:
            raise Invalid(f"unknown fields: {sorted(unknown)}")
        if "title" in fields and not str(fields["title"]).strip():
            raise Invalid("title cannot be empty")
        if "status" in fields and fields["status"] not in STATUSES:
            raise Invalid(f"status must be one of {STATUSES}")
        if "priority" in fields and fields["priority"] not in (1, 2, 3, 4):
            raise Invalid("priority must be 1 to 4")
        if "tags" in fields:
            if not isinstance(fields["tags"], list):
                raise Invalid("tags must be a list")
            fields = {**fields, "tags": json.dumps(fields["tags"])}
        return fields

    def list(self) -> list[dict[str, Any]]:
        return [self._row(r) for r in self._db.execute("SELECT * FROM tasks ORDER BY id")]

    def get(self, task_id: int) -> dict[str, Any]:
        r = self._db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if r is None:
            raise NotFound(task_id)
        return self._row(r)

    def create(self, fields: dict[str, Any], created_at: str | None = None) -> dict[str, Any]:
        fields = self._validate({k: v for k, v in fields.items() if v is not None})
        if "title" not in fields:
            raise Invalid("title is required")
        stamp = created_at or now()
        fields["created_at"] = fields["updated_at"] = stamp
        if fields.get("status") == "done":
            fields["completed_at"] = stamp
        with self._lock:
            cols = ", ".join(fields)
            cur = self._db.execute(f"INSERT INTO tasks ({cols}) VALUES ({', '.join('?' * len(fields))})",
                                   tuple(fields.values()))
            self._db.commit()
            return self.get(cur.lastrowid)

    def update(self, task_id: int, base_version: int, fields: dict[str, Any]) -> dict[str, Any]:
        fields = self._validate(fields)
        with self._lock:
            current = self.get(task_id)
            if current["version"] != base_version:
                raise Conflict(current)
            if "status" in fields:
                if fields["status"] == "done" and current["status"] != "done":
                    fields["completed_at"] = now()
                elif fields["status"] != "done":
                    fields["completed_at"] = None
            fields["updated_at"] = now()
            sets = ", ".join(f"{k} = ?" for k in fields)
            self._db.execute(f"UPDATE tasks SET {sets}, version = version + 1 WHERE id = ? AND version = ?",
                             (*fields.values(), task_id, base_version))
            self._db.commit()
            return self.get(task_id)

    def delete(self, task_id: int, base_version: int) -> dict[str, Any]:
        with self._lock:
            current = self.get(task_id)
            if current["version"] != base_version:
                raise Conflict(current)
            self._db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            self._db.commit()
            return current

    def add_focus(self, task_id: int | None, minutes: int, finished_at: str | None = None) -> dict[str, Any]:
        if not 1 <= minutes <= 180:
            raise Invalid("minutes must be between 1 and 180")
        with self._lock:
            if task_id is not None:
                self.get(task_id)
                self._db.execute("UPDATE tasks SET spent_min = spent_min + ?, updated_at = ?, version = version + 1 "
                                 "WHERE id = ?", (minutes, now(), task_id))
            self._db.execute("INSERT INTO focus_sessions (task_id, minutes, finished_at) VALUES (?, ?, ?)",
                             (task_id, minutes, finished_at or now()))
            self._db.commit()
        return self.get(task_id) if task_id is not None else {}

    def focus_sessions(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute("SELECT * FROM focus_sessions ORDER BY finished_at")]

    def reset(self) -> None:
        with self._lock:
            self._db.executescript("DELETE FROM focus_sessions; DELETE FROM tasks; DELETE FROM sqlite_sequence;")
            self._db.commit()
