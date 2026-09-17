"""
Project 00 -- the quick-entry parser.

    "Draft the Q3 report !high #work #finance @friday ~45m"
    -> title "Draft the Q3 report", priority high, tags work and finance,
       due the coming Friday, estimate 45 minutes

Markers can appear anywhere in the text:

    !urgent !high !normal !low        priority (also !1 .. !4)
    #tag                              any number of tags, starting with a letter
    @today @tomorrow @monday ..       due date; also @next-week, @in-3-days,
    @2026-10-01, @oct-1               and @none to clear it
    ~30m ~1h ~1h30m ~90               time estimate (bare numbers are minutes)

Relative dates are resolved against an explicit `today`, never the system
clock, so the same text always parses the same way in tests. A marker that
does not parse is left in the title rather than silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

PRIORITIES = {"urgent": 1, "high": 2, "normal": 3, "low": 4, "1": 1, "2": 2, "3": 3, "4": 4}
PRIORITY_NAMES = {1: "urgent", 2: "high", 3: "normal", 4: "low"}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

TOKEN = re.compile(r"(?<!\S)([!#@~])(\S+)")


@dataclass
class Parsed:
    title: str
    priority: int = 3
    tags: list[str] = field(default_factory=list)
    due: str | None = None
    estimate_min: int | None = None
    unparsed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["priority_name"] = PRIORITY_NAMES[self.priority]
        return d


def parse_due(value: str, today: date) -> date | None | str:
    """A date, None for @none, or the string 'invalid'."""
    v = value.lower().rstrip(".,;")
    if v == "none":
        return None
    if v == "today":
        return today
    if v in ("tomorrow", "tmrw"):
        return today + timedelta(days=1)
    if v in ("next-week", "nextweek"):
        return today + timedelta(days=7 - today.weekday())
    m = re.fullmatch(r"in-(\d{1,3})-(day|days|week|weeks)", v)
    if m:
        n = int(m.group(1))
        return today + timedelta(days=n * (7 if m.group(2).startswith("week") else 1))
    for i, name in enumerate(WEEKDAYS):
        if v in (name, name[:3]):
            ahead = (i - today.weekday()) % 7 or 7  # "friday" on a Friday means next Friday
            return today + timedelta(days=ahead)
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", v)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return "invalid"
    m = re.fullmatch(r"([a-z]{3})[a-z]*-(\d{1,2})", v)
    if m and m.group(1) in MONTHS:
        month, day = MONTHS.index(m.group(1)) + 1, int(m.group(2))
        try:
            candidate = date(today.year, month, day)
        except ValueError:
            return "invalid"
        # A month-day already past this year means next year.
        return candidate if candidate >= today else date(today.year + 1, month, day)
    return "invalid"


def parse_estimate(value: str) -> int | None:
    v = value.lower().rstrip(".,;")
    m = re.fullmatch(r"(?:(\d{1,2})h)?(?:(\d{1,3})m?)?", v)
    if not m or not v or (m.group(1) is None and m.group(2) is None):
        return None
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    total = hours * 60 + minutes
    return total if 0 < total <= 24 * 60 else None


def parse(text: str, today: date) -> Parsed:
    result = Parsed(title="")
    keep: list[str] = []
    last = 0
    for m in TOKEN.finditer(text):
        keep.append(text[last:m.start()])
        last = m.end()
        kind, value = m.group(1), m.group(2)
        raw = m.group(0)
        ok = True
        if kind == "!":
            p = PRIORITIES.get(value.lower().rstrip(".,;"))
            if p is None:
                ok = False
            else:
                result.priority = p
        elif kind == "#":
            tag = value.lower().rstrip(".,;")
            # Tags start with a letter, so "#3" in "fix issue #3" stays in the title.
            if re.fullmatch(r"[a-z][a-z0-9_-]*", tag):
                if tag not in result.tags:
                    result.tags.append(tag)
            else:
                ok = False
        elif kind == "@":
            due = parse_due(value, today)
            if due == "invalid":
                ok = False
            else:
                result.due = due.isoformat() if isinstance(due, date) else None
        elif kind == "~":
            est = parse_estimate(value)
            if est is None:
                ok = False
            else:
                result.estimate_min = est
        if not ok:
            keep.append(raw)
            result.unparsed.append(raw)
    keep.append(text[last:])
    result.title = re.sub(r"\s+", " ", "".join(keep)).strip()
    return result
