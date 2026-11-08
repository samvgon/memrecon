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
until. A linked "updated" (version 2's attached_to) closes the field's earlier values linked to the
same thing; an unlinked one closes all the field's earlier values; "added" does not (a list gains an item); "replaced"
and "removed" close the matching value; "done" marks the matching task done; "no longer" closes the
matching liked item and records it as no longer liked. Nothing in this module is ever shown to the agent.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import asdict, dataclass

from .ingest import MAX_ATTEMPTS, UnusableAnswer, answer_format, extract_block, load_prompt, render_conversation
from .llm import ModelCallError, OutputTruncated

PROMPT = "probe_extract_v1"
KINDS = ("number", "date", "name", "title", "item", "task", "preference")
STATUSES = ("new", "added", "updated", "replaced", "removed", "done", "no longer")
WHO = ("user", "user and others", "someone else", "")  # version 1; version 2 drops "someone else" for a name copied from the line
KEYS = {"line", "detail", "qualifier", "field", "who", "kind", "status"}
KEYS_V2 = KEYS | {"id", "attached_to"}
KEYS_V3 = KEYS_V2 | {"replaces"}
REPLACING = ("updated", "removed", "done", "no longer")  # statuses that may name the earlier row they replace (version 3)


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
    row_id: str = ""  # unique across the answer key: persona:session:id (version 2); empty for version 1 rows
    attached_to: str | None = None  # row_id of the detail this one belongs to (version 2); a root id; "unknown" (v3)
    replaces: str | None = None  # row_id of the earlier row it replaces, removes, finishes or no longer holds (v3)

    def to_json(self) -> dict:
        d = asdict(self)
        d["record"] = list(self.record)
        return d


# -- derived columns -----------------------------------------------------------------------------

_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mil": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
_NUMBER = re.compile(r"(?P<cur>[$€£¥])?\s*(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\s*(?P<mult>k|thousand|million|mil|bn|billion|m|b)?\b\s*(?P<rest>.*)$", re.I)


_WORD_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                         "fifteen sixteen seventeen eighteen nineteen".split())}
_WORD_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_WORD_SCALES = {"hundred": 100, "thousand": 1e3, "million": 1e6, "billion": 1e9}


def words_to_number(text: str) -> tuple[float | None, str]:
    """'ten thousand steps' -> (10000.0, 'steps'); 'five days' -> (5.0, 'days'); no number words -> (None, text)."""
    tokens = text.split()
    total, current, used = 0.0, 0.0, 0
    for j, tok in enumerate(tokens):
        parts = [x for x in re.sub(r"[^a-z-]", "", tok.lower()).split("-") if x]
        known = all(x in _WORD_UNITS or x in _WORD_TENS or x in _WORD_SCALES or (x == "and" and used) for x in parts)
        if not parts or not known:
            break
        for x in parts:
            if x in _WORD_UNITS:
                current += _WORD_UNITS[x]
            elif x in _WORD_TENS:
                current += _WORD_TENS[x]
            elif x == "hundred":
                current = (current or 1) * 100
            elif x in _WORD_SCALES:
                total += (current or 1) * _WORD_SCALES[x]
                current = 0
        used = j + 1
    if used == 0:
        return None, text
    return total + current, " ".join(tokens[used:]).strip(" .,;:")


def split_number(detail: str) -> tuple[float | None, str | None]:
    """'10k steps' -> (10000.0, 'steps'); '$1.2 million' -> (1200000.0, '$'); '15%' -> (15.0, '%');
    number words too: 'ten thousand steps' -> (10000.0, 'steps')."""
    m = _NUMBER.search(detail)
    if not m:
        value, rest = words_to_number(re.sub(r"^(a|an)\s+(?=hundred|thousand|million|billion)", "one ", detail.strip(), flags=re.I))
        return (round(value, 6), rest or None) if value is not None else (None, None)
    value = float(m.group("num").replace(",", "")) * _MULT.get((m.group("mult") or "").lower(), 1)
    rest = m.group("rest").strip(" .,;:").lstrip("-").strip()  # "18-month timeline" -> "month timeline"
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
    if re.search(r"\b(today|tonight|this (morning|afternoon|evening))\b", t):
        return base.isoformat()
    if re.search(r"\btomorrow\b", t):
        return (base + dt.timedelta(days=1)).isoformat()
    if re.search(r"\b(yesterday|last night)\b", t):
        return (base - dt.timedelta(days=1)).isoformat()
    m = re.search(r"\bin (\d+) days?\b", t)
    if m:
        return (base + dt.timedelta(days=int(m.group(1)))).isoformat()
    m = re.search(r"\b(next |this |on )?(" + "|".join(_WEEKDAYS) + r")\b", t)
    if m:
        target = _WEEKDAYS.index(m.group(2))
        ahead = (target - base.weekday()) % 7
        if ahead == 0 and m.group(1) != "this ":
            ahead = 7
        return (base + dt.timedelta(days=ahead)).isoformat()
    month = "|".join(sorted(_MONTHS, key=len, reverse=True))
    m = re.search(rf"\b({month})\.? (\d{{1,2}})(?:st|nd|rd|th)?\b(?:,? (\d{{4}}))?", t)
    if m:
        mo, day, year = _MONTHS[m.group(1)], int(m.group(2)), int(m.group(3) or base.year)
    else:
        m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? (?:of )?({month})\b(?:,? (\d{{4}}))?", t)
        if not m:
            return None
        mo, day, year = _MONTHS[m.group(2)], int(m.group(1)), int(m.group(3) or base.year)
    try:
        return dt.date(year, mo, day).isoformat()
    except ValueError:
        return None


# -- extraction ----------------------------------------------------------------------------------


def parse_probe_answer(reply: str, readable: dict[int, str], version: int = 1, existing: dict[str, str] | None = None) -> list[dict]:
    """The details, each checked: keys, allowed values, a line that was to be read, and the detail and
    qualifier copied exactly from that line. The same row twice is unusable.
    Version 2 also checks ids (1, 2, ... each once), attached_to (null or another detail's id) and a named
    "who" copied exactly from the detail's line. Version 3: attached_to may also be "R", "unknown" or an E
    label of `existing` (label -> row_id), and "replaces" is null or an E label, for REPLACING statuses only."""
    existing = existing or {}
    data = extract_block(reply, "details")
    if set(data) != {"details"} or not isinstance(data["details"], list):
        raise UnusableAnswer('The object must be {"details": [...]}.')
    keys = KEYS_V3 if version >= 3 else KEYS_V2 if version >= 2 else KEYS
    out, seen = [], set()
    for item in data["details"]:
        if not isinstance(item, dict) or set(item) != keys:
            names = ('"id", "line", "detail", "qualifier", "field", "who", "kind", "status", "attached_to" and "replaces"' if version >= 3
                     else '"id", "line", "detail", "qualifier", "field", "who", "kind", "status" and "attached_to"' if version >= 2
                     else '"line", "detail", "qualifier", "field", "who", "kind" and "status"')
            raise UnusableAnswer(f"Each detail must have exactly the keys {names}.")
        line = item["line"]
        if not isinstance(line, int) or isinstance(line, bool):
            raise UnusableAnswer(f'"line" must be a number without quotes, one of {sorted(readable)}; got {json.dumps(line)}.')
        if line not in readable:
            raise UnusableAnswer(f'"line" must be one of the lines to read ({sorted(readable)}); got {line}.')
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
        if version >= 2:
            if item["who"] not in ("user", "user and others", "") and item["who"] not in readable[line]:
                raise UnusableAnswer(f'"who" must be "user", "user and others", "", or a name copied exactly from its line; line {line} does not contain "{item["who"]}".')
        elif item["who"] not in WHO:
            raise UnusableAnswer('"who" must be "user", "user and others", "someone else" or "".')
        key = (line, item["detail"], item["status"], item["field"])
        if key in seen:
            raise UnusableAnswer("The same detail appears more than once; list each detail once.")
        seen.add(key)
        out.append(item)
    if version >= 2:
        ids = [it["id"] for it in out]
        if not all(isinstance(i, int) and not isinstance(i, bool) and i > 0 for i in ids) or len(ids) != len(set(ids)):
            raise UnusableAnswer('Each "id" must be a different whole number: 1, 2, 3 and so on.')
        for it in out:
            a = it["attached_to"]
            if version >= 3 and isinstance(a, str):
                if a not in ("R", "unknown") and a not in existing:
                    raise UnusableAnswer(f'"attached_to" must be the id of another detail, "R", an E label shown ({", ".join(existing) or "none"}), "unknown" or null; detail {it["id"]} has {json.dumps(a)}.')
            elif a is not None and (not isinstance(a, int) or isinstance(a, bool) or a not in ids or a == it["id"]):
                what = 'the id of another detail in this answer, "R", an E label, "unknown" or null' if version >= 3 else "null or the id of another detail in this answer"
                raise UnusableAnswer(f'"attached_to" must be {what}; detail {it["id"]} has {json.dumps(a)}.')
            if version >= 3:
                r = it["replaces"]
                if r is not None and (it["status"] not in REPLACING or r not in existing):
                    raise UnusableAnswer(f'"replaces" must be null, or for "updated", "removed", "done" or "no longer" an E label shown ({", ".join(existing) or "none"}); detail {it["id"]} has {json.dumps(r)}.')
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
        self.rows: list[Row] = []  # every row so far, for "already recorded" (version 3)
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
        version = int(self.prompt.rsplit("_v", 1)[1]) if "_v" in self.prompt else 1
        recorded, existing = self.recorded(record, readable) if version >= 3 else ("", {})
        # placeholders a version's template does not use are ignored (recorded is used from version 3 on)
        user = template.format(date=session.get("date", ""), lines=", ".join(str(n) for n in sorted(readable)),
                               fields=", ".join(fields) if fields else "(none)", conversation=conversation, recorded=recorded)
        items, attempts, error = self._ask(system, user, lambda r: parse_probe_answer(r, readable, version, existing), sid)
        if items is None:
            return ProbeOutcome(sid, False, attempts, [], error)
        rows = []
        rid = lambda i: f"{self.persona}:{sid}:{i}"  # noqa: E731
        for it in items:
            row = Row(record, it["field"].strip(), it["detail"], it["qualifier"], it["who"], it["kind"], it["status"],
                      sid, it["line"], session.get("date", ""))
            if version >= 2:
                row.row_id = rid(it["id"])
                a = it["attached_to"]
                row.attached_to = (None if a is None else rid(a) if isinstance(a, int) else root_id(self.persona, record) if a == "R"
                                   else "unknown" if a == "unknown" else existing[a])
            if version >= 3 and it["replaces"] is not None:
                row.replaces = existing[it["replaces"]]
            if row.kind == "number":
                row.number, row.unit = split_number(row.detail)
            if row.kind == "date":
                row.resolved = resolve_date(row.detail, row.date)
            rows.append(row)
            if row.field not in fields:
                fields.append(row.field)
        self.fields[record] = fields
        self.rows += rows
        return ProbeOutcome(sid, True, attempts, rows)

    def recorded(self, record: tuple, readable: dict[int, str]) -> tuple[str, dict[str, str]]:
        """What is already recorded, as shown to version 3: R, then E1, E2 ... for the entries that still
        hold for this record, and for other records' entries whose detail appears in the lines to read
        (a liked item the user now says they no longer like is another record under A10). Returns the
        text and the map from E label to row_id."""
        if not self.rows:
            return "R: the thing this conversation is about (nothing recorded yet)", {}
        now = true_at(build_table(self.rows), max(r.session for r in self.rows))
        text = norm(" ".join(readable.values()))
        own = [e for e in now if e["record"] == record and e.get("row_id")]
        others = [e for e in now if e["record"] != record and e.get("row_id") and len(norm(e["detail"])) >= 3 and norm(e["detail"]) in text]
        shown, seen = [], set()
        for e in own + others[:15]:
            if e["row_id"] not in seen:
                seen.add(e["row_id"])
                shown.append(e)
        labels = {f"E{i}": e["row_id"] for i, e in enumerate(shown, 1)}
        lines = ["R: the thing this conversation is about"] + [
            f"E{i} ({e['field']}{', ' + e['state'] if e['state'] != 'true' else ''}): {e['detail']}" for i, e in enumerate(shown, 1)]
        return "\n".join(lines), labels

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


def root_id(persona: str, record: tuple) -> str:
    """The row_id standing for a record itself (version 3's "R"): no row of its own, a link target."""
    return f"{persona}:R:" + "/".join(str(x) for x in record)


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
    detail_of = {r.row_id: r.detail for r in rows if r.row_id}

    def open_entries(row, any_record=False):
        return [e for e in entries if e["until_session"] is None and e["state"] in ("true", "done")
                and (any_record or e["record"] == row.record)]

    def close(e, row, state="false"):
        e["until_session"], e["until_date"] = row.session, row.date
        if state != "false":
            new = {**e, "state": state, "from_session": row.session, "from_date": row.date,
                   "until_session": None, "until_date": None, "status": row.status}
            entries.append(new)

    def parent(row):
        if row.attached_to and (":R:" in row.attached_to or row.attached_to == "unknown"):
            return row.attached_to  # the record itself, or unknown
        return norm(detail_of.get(row.attached_to, "")) if row.attached_to else ""

    def entry_parent(e):
        a = e.get("attached_to")
        if a and (":R:" in a or a == "unknown"):
            return a
        return norm(e.get("attached_detail") or "")

    def add(row, state):
        entries.append({"record": row.record, "field": row.field, "detail": row.detail, "qualifier": row.qualifier,
                        "row_id": row.row_id, "attached_to": row.attached_to, "attached_detail": (row.attached_to if row.attached_to and (":R:" in row.attached_to or row.attached_to == "unknown")
                                            else detail_of.get(row.attached_to) if row.attached_to else None),
                        "replaces": row.replaces,
                        "who": row.who, "kind": row.kind, "number": row.number, "unit": row.unit, "resolved": row.resolved,
                        "status": row.status, "state": state, "from_session": row.session, "from_date": row.date,
                        "until_session": None, "until_date": None})

    for row in sorted(rows, key=lambda r: (r.session, r.line)):
        if row.replaces and row.status in REPLACING:
            # version 3: the earlier row is named, so exactly that one closes
            hits = [e for e in entries if e.get("row_id") == row.replaces and e["until_session"] is None and e["state"] in ("true", "done")]
            if hits:
                state = {"updated": "false", "removed": "false", "done": "done", "no longer": "no longer"}[row.status]
                for e in hits:
                    close(e, row, state)
                if row.status == "updated":
                    add(row, "true")
                continue
        if row.status in ("new", "added"):
            add(row, "true")
        elif row.status == "updated":
            # a linked update closes the field's earlier values linked to the same thing (one action item's
            # deadline, not another's); an unlinked one closes every earlier value of the field, as in version 1
            for e in open_entries(row):
                if e["field"] == row.field and e["state"] == "true" and (not row.attached_to or entry_parent(e) == parent(row)):
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


# -- what is graded (D45) --------------------------------------------------------------------------

PRIMARY_KINDS = ("number", "date", "name", "title", "item")  # code-gradable: the pre-registered probe types
HEDGES = ("about", "approximately", "approx.", "around", "roughly", "nearly", "almost", "at least", "at most",
          "under", "over", "up to", "more than", "less than", "fewer than", "maximum", "minimum", "max", "min",
          "only if", "every day", "each day", "daily", "per day", "per week", "a week", "a day", "per month",
          "or more", "or less", "no more than", "no less than", "below", "above", "within")


def graded_qualifier(qualifier: str) -> str:
    """The qualifier if it is a real hedge, limit or condition (from HEDGES), else "" (not graded)."""
    q = qualifier.strip().lower()
    return qualifier.strip() if q and any(q == h or q.startswith(h + " ") or q.endswith(" " + h) for h in HEDGES) else ""


def grading_view(entries: list[dict]) -> list[dict]:
    """The probes to grade, derived from table entries (the rows are never changed).

    Primary probes (graded by code): kinds number (value and unit), date (the resolved day, or the text
    when unresolved), name, title and list item, in states "true" (should be in memory) and "false"
    (should not). Each keeps its qualifier only when it is a number or date and the qualifier is a real
    hedge or limit (D47: qualifiers are collected for every kind, scored only there), and its "who" only when
    it is not "user" (we, someone else). Secondary probes (graded by the judge): kinds task and
    preference, and every entry in state "done" or "no longer" (the field is kept: "no longer" means the
    field's claim no longer holds). Number, unit and resolved day are recomputed from the detail here, so
    rows extracted before a parser fix are graded with the current parser.
    """
    out = []
    for e in entries:
        primary = e["kind"] in PRIMARY_KINDS and e["state"] in ("true", "false")
        derived = {}
        if e["kind"] == "number":  # recomputed with the current parser, whatever the row stored
            derived["number"], derived["unit"] = split_number(e["detail"])
        if e["kind"] == "date":
            derived["resolved"] = resolve_date(e["detail"], e.get("from_date") or "")
        out.append({**e, **derived, "grade": "code" if primary else "judge",
                    "qualifier": graded_qualifier(e.get("qualifier") or "") if e["kind"] in ("number", "date") else "",
                    "who": e.get("who") if e.get("who") not in ("user", None) else ""})
    return out
