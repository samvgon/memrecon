"""Probes: the drift measure's answer key, built from what the user said (harness side).

The judge model reads the flagged user lines of each add, update or delete session and lists every
checkable detail, copied exactly (prompts/probe_extract_v1.md). Each detail becomes one row, stored
append-only and never edited:

    record, field, detail, qualifier, who, kind, status, session, line, date

The record is the one the session is about (given by the caller, from Memora's log, A10); "fields so
far" shows the extractor the field names already used for that record, so a changed value reuses its
field. Code, not the model, derives from each row:

    number, unit        from the detail, for kind "number" ("10k steps" -> 10000, "steps")
    resolved            for kind "date": the calendar date, from the message's conversation date, when it
                        names one day ("next Tuesday"); None for a span ("next week") or unknown text

and `build_table` computes what is true when from all rows, in session order: each entry has the
detail, its state ("true", "false", "done", "no longer") and the session and date it held from and
until. "updated" closes the field's earlier values; "added" does not (a list gains an item); "replaced"
and "removed" close the matching value; "done" marks the matching task done; "no longer" closes the
matching liked item and records it as no longer liked. Nothing in this module is ever shown to the agent.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import asdict, dataclass

from .ingest import MAX_ATTEMPTS, UnusableAnswer, answer_format, extract_block, load_prompt, render_conversation
from .llm import ModelCallError, OutputTruncated

PROMPT = "probe_extract_v1"
KINDS = ("number", "date", "name", "title", "item", "task", "preference")
STATUSES = ("new", "added", "updated", "replaced", "removed", "done", "no longer")
WHO = ("user", "user and others", "someone else", "")
KEYS = {"line", "detail", "qualifier", "field", "who", "kind", "status"}


@dataclass
class Row:
    record: tuple
    field: str
    detail: str
    qualifier: str
    who: str
    kind: str
    status: str
    session: int
    line: int
    date: str
    number: float | None = None
    unit: str | None = None
    resolved: str | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["record"] = list(self.record)
        return d


# -- derived columns -----------------------------------------------------------------------------

_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mil": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
_NUMBER = re.compile(r"(?P<cur>[$€£¥])?\s*(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\s*(?P<mult>k|thousand|million|mil|bn|billion|m|b)?\b\s*(?P<rest>.*)$", re.I)


def split_number(detail: str) -> tuple[float | None, str | None]:
    """'10k steps' -> (10000.0, 'steps'); '$1.2 million' -> (1200000.0, '$'); '15%' -> (15.0, '%')."""
    m = _NUMBER.search(detail)
    if not m:
        return None, None
    value = float(m.group("num").replace(",", "")) * _MULT.get((m.group("mult") or "").lower(), 1)
    rest = m.group("rest").strip(" .,;:")
    unit = " ".join(x for x in (m.group("cur") or "", rest) if x)
    return round(value, 6), unit or None


_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                        "september", "october", "november", "december"], 1)}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9


def resolve_date(detail: str, conversation_date: str) -> str | None:
    """The calendar date a date detail names, from the conversation date; None if it names no single day.

    Conventions (to be checked against the data): "next <weekday>" and "<weekday>" are the first such day
    after the conversation date; "this <weekday>" is the first on or after it; a month and day without a
    year take the conversation's year.
    """
    try:
        base = dt.date.fromisoformat(conversation_date)
    except (TypeError, ValueError):
        return None
    t = detail.strip().lower().rstrip(".")
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        return m.group(0)
    if re.fullmatch(r"(today|tonight|this (morning|afternoon|evening)|earlier today|later today)", t):
        return base.isoformat()
    if t == "tomorrow":
        return (base + dt.timedelta(days=1)).isoformat()
    if t == "yesterday":
        return (base - dt.timedelta(days=1)).isoformat()
    m = re.fullmatch(r"in (\d+) days?", t)
    if m:
        return (base + dt.timedelta(days=int(m.group(1)))).isoformat()
    m = re.fullmatch(r"(next |this |on )?(" + "|".join(_WEEKDAYS) + r")", t)
    if m:
        target = _WEEKDAYS.index(m.group(2))
        ahead = (target - base.weekday()) % 7
        if ahead == 0 and m.group(1) != "this ":
            ahead = 7
        return (base + dt.timedelta(days=ahead)).isoformat()
    month = "|".join(sorted(_MONTHS, key=len, reverse=True))
    m = re.fullmatch(rf"({month})\.? (\d{{1,2}})(?:st|nd|rd|th)?(?:,? (\d{{4}}))?", t) or None
    if m:
        mo, day, year = _MONTHS[m.group(1)], int(m.group(2)), int(m.group(3) or base.year)
    else:
        m = re.fullmatch(rf"(\d{{1,2}})(?:st|nd|rd|th)? (?:of )?({month})(?:,? (\d{{4}}))?", t)
        if not m:
            return None
        mo, day, year = _MONTHS[m.group(2)], int(m.group(1)), int(m.group(3) or base.year)
    try:
        return dt.date(year, mo, day).isoformat()
    except ValueError:
        return None


# -- extraction ----------------------------------------------------------------------------------


def parse_probe_answer(reply: str, readable: dict[int, str]) -> list[dict]:
    """The details, each checked: keys, allowed values, a line that was to be read, and the detail and
    qualifier copied exactly from that line. The same row twice is unusable."""
    data = extract_block(reply, "details")
    if set(data) != {"details"} or not isinstance(data["details"], list):
        raise UnusableAnswer('The object must be {"details": [...]}.')
    out, seen = [], set()
    for item in data["details"]:
        if not isinstance(item, dict) or set(item) != KEYS:
            raise UnusableAnswer('Each detail must have exactly the keys "line", "detail", "qualifier", "field", "who", "kind" and "status".')
        line = item["line"]
        if not isinstance(line, int) or isinstance(line, bool) or line not in readable:
            raise UnusableAnswer(f'"line" must be one of the lines to read ({sorted(readable)}).')
        for key in ("detail", "qualifier", "field", "who", "kind", "status"):
            if not isinstance(item[key], str):
                raise UnusableAnswer(f'"{key}" must be a string.')
        if not item["detail"].strip() or item["detail"] not in readable[line]:
            raise UnusableAnswer(f'Each "detail" must be copied exactly from its line; line {line} does not contain "{item["detail"]}".')
        if item["qualifier"] and item["qualifier"] not in readable[line]:
            raise UnusableAnswer(f'Each "qualifier" must be copied exactly from its line; line {line} does not contain "{item["qualifier"]}".')
        if not item["field"].strip():
            raise UnusableAnswer('"field" must name what the detail is.')
        if item["kind"] not in KINDS:
            raise UnusableAnswer(f'"kind" must be one of {", ".join(KINDS)}.')
        if item["status"] not in STATUSES:
            raise UnusableAnswer(f'"status" must be one of {", ".join(STATUSES)}.')
        if item["who"] not in WHO:
            raise UnusableAnswer('"who" must be "user", "user and others", "someone else" or "".')
        key = (line, item["detail"], item["status"], item["field"])
        if key in seen:
            raise UnusableAnswer("The same detail appears more than once; list each detail once.")
        seen.add(key)
        out.append(item)
    return out


@dataclass
class ProbeOutcome:
    session_id: int
    ok: bool
    attempts: int
    rows: list[Row]
    error: str | None = None


class ProbeExtractor:
    """Runs the extractor on sessions in order, keeping the field names used per record."""

    def __init__(self, client, persona: str = "", prompt: str = PROMPT, max_tokens: int = 2048, tag: str | None = None) -> None:
        self.client = client
        self.persona = persona
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.tag = tag
        self.fields: dict[tuple, list[str]] = {}
        self.calls: list[dict] = []
        self.feedback: list[str] = []  # why answers were sent back (for reporting)

    def extract(self, session: dict, record: tuple) -> ProbeOutcome:
        conversation, lines = render_conversation(session)
        readable = {n: t["message"] for n, t in lines.items() if t.get("speaker") == "user_agent" and t.get("share_memory")}
        sid = session["session_id"]
        if not readable:
            return ProbeOutcome(sid, True, 0, [])
        system, template = load_prompt(self.prompt)
        fields = self.fields.get(record, [])
        user = template.format(date=session.get("date", ""), lines=", ".join(str(n) for n in sorted(readable)),
                               fields=", ".join(fields) if fields else "(none)", conversation=conversation)
        items, attempts, error = self._ask(system, user, lambda r: parse_probe_answer(r, readable), sid)
        if items is None:
            return ProbeOutcome(sid, False, attempts, [], error)
        rows = []
        for it in items:
            row = Row(record, it["field"].strip(), it["detail"], it["qualifier"], it["who"], it["kind"], it["status"],
                      sid, it["line"], session.get("date", ""))
            if row.kind == "number":
                row.number, row.unit = split_number(row.detail)
            if row.kind == "date":
                row.resolved = resolve_date(row.detail, row.date)
            rows.append(row)
            if row.field not in fields:
                fields.append(row.field)
        self.fields[record] = fields
        return ProbeOutcome(sid, True, attempts, rows)

    def _ask(self, system: str, user: str, parse, sid: int):
        """As ingestion: up to MAX_ATTEMPTS, feedback with the answer format, twice the room if cut off."""
        reminder = " " + answer_format(system)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        limit, error = self.max_tokens, None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = self.client.chat(messages, max_tokens=limit, tag=self.tag,
                                          meta={"step": "probe_extract", "persona": self.persona, "session": sid, "attempt": attempt})
                usage = getattr(result, "usage", None) or {}
                self.calls.append({"session": sid, "attempt": attempt, "cached": getattr(result, "cached", False),
                                   "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")})
            except OutputTruncated:
                limit *= 2
                error = f"cut off at {limit // 2} tokens"
                continue
            except ModelCallError as e:
                error = str(e)
                if "reasoning" in str(e) or "unreadable" in str(e):
                    continue
                return None, attempt, error
            try:
                return parse(result.content), attempt, None
            except UnusableAnswer as e:
                error = str(e)
                self.feedback.append(error)
                messages = messages + [
                    {"role": "assistant", "content": result.content},
                    {"role": "user", "content": f"That answer could not be used: {e} Answer again, following the instructions exactly.{reminder}"},
                ]
        return None, MAX_ATTEMPTS, error


# -- what is true when ---------------------------------------------------------------------------


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'“”‘’*`]", "", text)).strip(" .,!?;:").lower()


def _matches(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def build_table(rows: list[Row], cross_record_kinds: tuple[str, ...] = ("preference",)) -> list[dict]:
    """What is true when, computed from all rows (never stored, never edited).

    Rows are applied in (session, line) order. A "replaced", "removed", "done" or "no longer" row closes
    or marks a matching open entry of the same record; for kinds in `cross_record_kinds` (preferences,
    whose old and new items are different records under A10) it may match any record. With no match it
    is kept as an entry of its own (a value that should not be, or a task done, from that session).
    """
    entries: list[dict] = []

    def open_entries(row, any_record=False):
        return [e for e in entries if e["until_session"] is None and e["state"] in ("true", "done")
                and (any_record or e["record"] == row.record)]

    def close(e, row, state="false"):
        e["until_session"], e["until_date"] = row.session, row.date
        if state != "false":
            new = {**e, "state": state, "from_session": row.session, "from_date": row.date,
                   "until_session": None, "until_date": None, "status": row.status}
            entries.append(new)

    def add(row, state):
        entries.append({"record": row.record, "field": row.field, "detail": row.detail, "qualifier": row.qualifier,
                        "who": row.who, "kind": row.kind, "number": row.number, "unit": row.unit, "resolved": row.resolved,
                        "status": row.status, "state": state, "from_session": row.session, "from_date": row.date,
                        "until_session": None, "until_date": None})

    for row in sorted(rows, key=lambda r: (r.session, r.line)):
        if row.status in ("new", "added"):
            add(row, "true")
        elif row.status == "updated":
            for e in open_entries(row):
                if e["field"] == row.field and e["state"] == "true":
                    close(e, row)
            add(row, "true")
        else:
            wide = row.kind in cross_record_kinds
            hits = [e for e in open_entries(row, wide) if _matches(e["detail"], row.detail)]
            state = {"replaced": "false", "removed": "false", "done": "done", "no longer": "no longer"}[row.status]
            for e in hits:
                close(e, row, state)
            if not hits:
                add(row, state)
    return entries


def true_at(entries: list[dict], session_id: int) -> list[dict]:
    """The entries holding just after `session_id` (a session's changes apply from that session on)."""
    return [e for e in entries if e["from_session"] <= session_id and (e["until_session"] is None or e["until_session"] > session_id)]
