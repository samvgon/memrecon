"""E14: how many code-gradable probe candidates do the flagged turns give, per persona?

Question (D10, power): the pre-registered test assumes about 500 probed facts. Probes come from the
dialogue: turns flagged `share_memory: True` in add and update sessions (E07, A2). How many of those
turns contain something code can check, how many such elements are there, and how many belong to facts
that are never changed or deleted later (the cleanest probes)?
Method: for each weekly persona, take the flagged turns of add and update sessions and find candidate
elements with simple patterns:
  number   - digits with an optional currency sign before and an optional unit word after
             ("$8.45", "9,045 steps", "15-month", "12+"); number words ("three phases") are not counted;
  quoted   - text in double quotes, *asterisks*, or single quotes after a space or colon
             (titles: "Thinking, Fast and Slow", *Super 8*, 'Marketing Director (Automotive Sector)');
  hashtag  - #Words;
  name     - two or more capitalised words in a row ("Project Horizon", "Regional Leadership Team"),
             or one capitalised word that does not start a sentence ("Seychelles");
  qualifier- maximum, minimum, at least, at most, up to, no more than, no later than, exactly, only,
             and high/medium/low followed by risk or priority;
  record value - the log's own key string for the record, found in the turn: a preference's item
             (with like/dislike wording, code-gradable as item + polarity), a todo's description or a
             calendar event's name.
A name that repeats a quoted title is not counted twice.
A fact is one add or update session (one record). It is "changed later" if a later update or delete
session concerns the same record, matched as in E12 (preference item; todo description; calendar
event name; food amount; step count) or, for content records, by the log's record ID (e.g.
project_proposal_1). Goal sessions are counted but their identity is not matched (bare numbers).
Limits: these are candidates found by patterns, not checked probes. Patterns over-count (any
capitalised phrase counts as a name) and under-count (number words, paraphrased facts, unflagged turns
that state facts). Building real probes needs judgement on each candidate; this only sizes the pool.
Counts on test personas are sizes only; nothing is tuned on them.
"""
import json
import random
import re
from collections import Counter, defaultdict

from common import ROOT, flagged_turns, iter_sessions, personas

UNIT = (
    r"(?:%|k\b|K\b|M\b|million|billion|thousand|steps?|days?|weeks?|months?|years?|hours?|minutes?|mins?"
    r"|languages?|locales?|people|participants|users|KB|MB|GB|TB|km|miles?|kg|lbs?|calories|pages?|words?"
    r"|slides?|screens?|countries|regions?|phases?|am|pm|AM|PM)"
)
PATTERNS = {
    "number": re.compile(r"(?<![\w#])(?:[$€£¥]\s?)?\d+(?:[.,]\d+)*\+?(?:\s?-?\s?" + UNIT + r")?"),
    "quoted": re.compile(r"[\"“*]([^\"”*]{3,80})[\"”*]|(?<=[\s:])'([^']{3,120})'(?=[\s.,;!?]|$)"),
    "hashtag": re.compile(r"#\w+"),
    "qualifier": re.compile(
        r"\b(?:maximum|minimum|at least|at most|up to|no more than|no later than|exactly|only"
        r"|(?:high|medium|low)(?:-|\s)(?:risk|priority))\b",
        re.I,
    ),
}
MULTI_NAME = re.compile(r"\b[A-Z][\w&'’.-]*(?:\s+(?:of|the|for|de|on|in|&)?\s*[A-Z][\w&'’.-]*)+")
SINGLE_NAME = re.compile(r"(?<=[a-z,;:]\s)[A-Z][\w&'’.-]{2,}")
NOT_NAMES = {"I", "I'm", "I've", "I'd", "I'll", "I’m", "I’ve", "Oh", "OK", "Okay"}
CONTENT_KEYS = ["project_title", "meeting_title", "email_subject", "email_purpose"]


def _bare(x):
    return re.sub(r"[\"“”*'.,!?:;]", "", x).strip().lower()


def elements(text, record_value=None):
    found = []
    for kind, pat in PATTERNS.items():
        for m in pat.finditer(text):
            found.append((kind, m.group(0).strip()))
    quoted = {_bare(v) for k, v in found if k == "quoted"}
    names = set()
    for m in MULTI_NAME.finditer(text):
        names.add(m.group(0).rstrip(".,;:!?"))
    for m in SINGLE_NAME.finditer(text):
        w = m.group(0).rstrip(".,;:!?")
        if w not in NOT_NAMES and not any(w in n for n in names):
            names.add(w)
    found += [("name", n) for n in sorted(names) if _bare(n) not in quoted and n not in NOT_NAMES]
    if record_value and record_value.strip().lower() in text.lower():
        found.append(("record value", record_value.strip()))
    return found


def record_key(s):
    """Identity of the record an add/update/delete session is about, as in E12 (plus content record IDs)."""
    d = s.get("operation_details") or {}
    t = s.get("session_type")
    if t == "preference":
        item = d.get("item")
        return ("preference", item.strip().lower()) if isinstance(item, str) else None
    if t != "activity":
        return None
    cat, it = d.get("category"), d.get("item")
    if cat is None and isinstance(it, str):
        return ("content", it)
    if not isinstance(it, dict):
        return None
    if cat == "todo_list" and it.get("description"):
        return (cat, it["description"].strip().lower())
    if cat == "calendar_event" and it.get("event_name"):
        return (cat, it["event_name"].strip().lower())
    if cat == "food_expenses" and it.get("amount") is not None:
        return (cat, f"{float(it['amount']):.2f}")
    if cat == "step_tracker" and it.get("step_count") is not None:
        return (cat, str(int(it["step_count"])))
    return None


split = json.loads((ROOT / "notes" / "split.json").read_text())
role = {p: "dev" for p in split["dev"]} | {p: "test" for p in split["test"]}
kinds = ["number", "quoted", "name", "hashtag", "qualifier", "record value"]

per = {}
examples = []
for persona in personas():
    sessions = [s for _, _, s in iter_sessions(persona)]
    keys = [record_key(s) if s.get("operation") in ("add", "update", "delete") else None for s in sessions]
    row = Counter()
    for i, s in enumerate(sessions):
        if s.get("operation") not in ("add", "update"):
            continue
        turns = flagged_turns(s)
        if not turns:
            continue
        changed = keys[i] is not None and any(keys[j] == keys[i] for j in range(i + 1, len(sessions)))
        d = s.get("operation_details") or {}
        it = d.get("item")
        if s.get("session_type") == "preference" and isinstance(it, str):
            value = it
        elif isinstance(it, dict):
            value = it.get("description") or it.get("event_name")
        else:
            value = None
        fact_elems = 0
        for t in turns:
            els = elements(t["message"], value if isinstance(value, str) else None)
            row["flagged turns"] += 1
            if els:
                row["turns with elements"] += 1
            for kind, _ in els:
                row[kind] += 1
            fact_elems += len(els)
            if role[persona] == "dev":
                examples.append((persona, s["session_id"], s.get("session_type"), s.get("operation"), t["message"], els))
        row["facts"] += 1
        if fact_elems:
            row["facts with elements"] += 1
            row["elements"] += fact_elems
            if keys[i] is None and s.get("session_type") == "goal":
                row["goal facts (identity not matched)"] += 1
            elif not changed:
                row["clean facts"] += 1
                row["clean elements"] += fact_elems
    per[persona] = row

cols = ["facts", "facts with elements", "clean facts", "flagged turns", "turns with elements", "elements", "clean elements"]
head = f"{'persona':<22} {'set':<4} " + " ".join(f"{c:>12}" for c in ["facts", "w/ elements", "clean", "turns", "turns w/ el", "elements", "clean el"])
print(head)
totals = defaultdict(Counter)
for persona in personas():
    r = per[persona]
    totals[role[persona]].update(r)
    totals["all"].update(r)
    print(f"{persona:<22} {role[persona]:<4} " + " ".join(f"{r[c]:>12}" for c in cols))
print()
for name in ("dev", "test", "all"):
    r = totals[name]
    print(f"{'TOTAL ' + name:<27} " + " ".join(f"{r[c]:>12}" for c in cols))
print()
print("facts: add/update sessions with flagged turns; clean: never updated or deleted later (identity as in E12);")
print("elements: code-gradable candidates found by pattern (see the docstring); goal facts are not identity-matched.")
print()
print("Elements by kind (all personas):")
for k in kinds:
    print(f"  {k:<16} {totals['all'][k]:>6}")
print(f"  goal facts with elements (identity not matched): {totals['all']['goal facts (identity not matched)']}")
print()
random.seed(14)
print("20 random flagged turns from development personas, with the candidates found:")
for persona, sid, st, op, msg, els in random.sample(examples, 20):
    print(f"- {persona} session {sid} ({st}, {op}): {msg[:160]}")
    print(f"    -> {els if els else 'none'}")
