"""E26: does the probe extractor (judge model) give a trustworthy answer key, checked against Memora's log?

Question (drift measure): on the development personas, the extractor (prompts/probe_extract_v1.md, the
judge role) lists every checkable detail of the flagged user lines of each add, update and delete
session, copied exactly, with field, qualifier, who, kind and status; code derives number, unit and
resolved date and computes what is true when (memrecon.probes). Is it reliable (failures, retries and
why), and do its rows agree with the log where the log can tell (E25): documents' added and removed
list items and changed fields, preference items and changes, expense and step amounts, goal amounts,
calendar events, to-dos (whose logged description is usually reworded in the dialogue, so matched by
shared words)? Also prints a sample of sessions with their rows for Sam's hand check.
Method: sessions in file order (A6); the record of a session is its A10 key, or ("goal", subcategory)
for goals, or ("session", id) otherwise. The flags choose the lines to read (harness side; the model is
shown only line numbers). Rows are written to results/probes/E26_<persona>.jsonl. Calls are cached.
Pilot: MEMRECON_E26_PILOT=N extracts only the first N sessions of the first persona (not evidence).
Limits: matching against the log by normalised text, number value or shared words, so a reworded
detail can count as missing; the log says what changed, not what the user said, so a disagreement can
be the log's or the dialogue's; the hand check is the judge of last resort.
"""
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

from common import ROOT, flagged_turns, iter_sessions, record_key
from memrecon.llm import client as model_client
from memrecon.probes import ProbeExtractor, build_table, norm, true_at

PILOT = int(os.environ.get("MEMRECON_E26_PILOT", "0") or 0)
OUT = ROOT / "results" / "probes"
STOP = {"the", "a", "an", "to", "my", "for", "and", "of", "on", "in", "i", "need", "plan", "up"}


def record_of(s):
    k = record_key(s)
    if k is not None:
        return k
    d = s.get("operation_details") or {}
    if s.get("session_type") == "goal":
        return ("goal", str(d.get("subcategory")))
    return ("session", s["session_id"])


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


def match_text(value, rows):
    v = norm(str(value))
    return any(v and (v in norm(r.detail) or norm(r.detail) in v) and len(norm(r.detail)) >= 3 for r in rows)


def match_number(value, rows):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return any(r.number is not None and abs(r.number - x) < 1e-6 for r in rows)


def match_words(value, rows):
    w = {t for t in re.findall(r"[a-z]+", str(value).lower()) if t not in STOP}
    return bool(w) and any(len(w & {t for t in re.findall(r"[a-z]+", r.detail.lower())}) / len(w) >= 0.5 for r in rows)


split = json.loads((ROOT / "notes" / "split.json").read_text())
judge = model_client("judge")
OUT.mkdir(parents=True, exist_ok=True)
personas = split["dev"][:1] if PILOT else split["dev"]
by_session = {}
all_rows, outcomes, feedback, calls = [], [], Counter(), []
start = time.monotonic()
for persona in personas:
    ex = ProbeExtractor(judge, persona=persona, tag="E26")
    sessions = [s for _, _, s in iter_sessions(persona) if s.get("operation") in ("add", "update", "delete") and flagged_turns(s)]
    if PILOT:
        sessions = sessions[:PILOT]
    rows_p = []
    for s in sessions:
        out = ex.extract(s, record_of(s))
        outcomes.append((persona, s, out))
        by_session[(persona, s["session_id"])] = (s, out.rows)
        rows_p += out.rows
    with open(OUT / f"E26_{persona}.jsonl", "w", encoding="utf-8") as f:
        for r in rows_p:
            f.write(json.dumps({"persona": persona, **r.to_json()}) + "\n")
    all_rows.append((persona, rows_p))
    feedback.update(x.split(";")[0][:90] for x in ex.feedback)
    calls += ex.calls
elapsed = time.monotonic() - start

if PILOT:
    print(f"PILOT RUN: not evidence. {len(outcomes)} sessions in {elapsed:.0f} s, {sum(1 for c in calls if not c['cached'])} fresh calls\n")

print("1. RELIABILITY")
print(f"  {len(outcomes)} sessions, {sum(not o.ok for _, _, o in outcomes)} failed, attempts {dict(sorted(Counter(o.attempts for *_, o in outcomes).items()))}, "
      f"{sum(1 for c in calls if not c['cached'])} fresh calls, {elapsed:.0f} s")
for p, s, o in outcomes:
    if not o.ok:
        print(f"    failed: {p} session {s['session_id']}: {o.error}")
print("  why answers were sent back: " + ("; ".join(f"{n} x {r}" for r, n in feedback.most_common()) or "never"))
print()

rows = [r for _, rs in all_rows for r in rs]
turns = sum(len(flagged_turns(s)) for _, s, _ in outcomes)
print("2. ROWS")
print(f"  {len(rows)} rows from {turns} flagged turns ({len(rows) / max(1, turns):.2f} per turn)")
print("  kind:   " + ", ".join(f"{k} {n}" for k, n in Counter(r.kind for r in rows).most_common()))
print("  status: " + ", ".join(f"{k} {n}" for k, n in Counter(r.status for r in rows).most_common()))
print("  who:    " + ", ".join(f"{k or '(empty)'} {n}" for k, n in Counter(r.who for r in rows).most_common()))
q = [r for r in rows if r.qualifier]
print(f"  with a qualifier: {len(q)}: " + ", ".join(f"{k} {n}" for k, n in Counter(r.qualifier.lower() for r in q).most_common(12)))
nums = [r for r in rows if r.kind == "number"]
print(f"  numbers: {len(nums)}, parsed {sum(r.number is not None for r in nums)}; units: "
      + ", ".join(f"{k} {n}" for k, n in Counter(r.unit for r in nums).most_common(10)))
dates = [r for r in rows if r.kind == "date"]
print(f"  dates: {len(dates)}, resolved to a day {sum(r.resolved is not None for r in dates)}; not resolved: "
      + ", ".join(sorted({r.detail for r in dates if r.resolved is None})[:25]))
fields = Counter(r.field for r in rows)
print(f"  {len(fields)} field names; most used: " + ", ".join(f"{k} {n}" for k, n in fields.most_common(15)))
print()

print("3. AGREEMENT WITH THE LOG (where it can tell; E25)")
c = Counter()
miss = defaultdict(list)
last_doc = {}
for p, s, o in outcomes:
    d = s.get("operation_details") or {}
    t, op, rs = s.get("session_type"), s.get("operation"), o.rows
    if not o.ok:
        continue
    if t == "activity" and d.get("category") is None and isinstance(d.get("content_data"), dict):
        item, cd = d.get("item"), d["content_data"]
        prev = last_doc.get((p, item))
        last_doc[(p, item)] = cd
        if op == "add" or prev is None:
            continue
        for f in sorted(set(prev) | set(cd)):
            a, b = prev.get(f), cd.get(f)
            if a == b:
                continue
            if isinstance(a, list) and isinstance(b, list):
                for x in b:
                    if x not in a:
                        ok = match_text(x, [r for r in rs if r.status in ("added", "new", "updated")])
                        c[("document: list item added", ok)] += 1
                        ok or miss["document: list item added"].append((p, s["session_id"], x))
                for x in a:
                    if x not in b:
                        ok = match_text(x, [r for r in rs if r.status in ("removed", "replaced")])
                        c[("document: list item removed", ok)] += 1
                        ok or miss["document: list item removed"].append((p, s["session_id"], x))
            else:
                ok = match_number(b, [r for r in rs if r.status in ("updated", "new", "added")]) or match_text(b, [r for r in rs if r.status in ("updated", "new", "added")])
                c[("document: field new value", ok)] += 1
                ok or miss["document: field new value"].append((p, s["session_id"], f"{f}: {b}"))
                ok = match_number(a, [r for r in rs if r.status == "replaced"]) or match_text(a, [r for r in rs if r.status == "replaced"])
                c[("document: field old value", ok)] += 1
                ok or miss["document: field old value"].append((p, s["session_id"], f"{f}: {a}"))
    elif t == "preference":
        item = d.get("item", "")
        if op == "add":
            ok = match_text(item, [r for r in rs if r.status in ("new", "added")])
            c[("preference add: item, as new", ok)] += 1
            ok or miss["preference add"].append((p, s["session_id"], item))
        elif op == "delete":
            ok = match_text(item, [r for r in rs if r.status in ("no longer", "removed")])
            c[("preference delete: item, as no longer", ok)] += 1
            ok or miss["preference delete"].append((p, s["session_id"], item))
        else:
            if d.get("old_item"):
                ok = match_text(d["old_item"], [r for r in rs if r.status in ("no longer", "replaced", "removed")])
                c[("preference update: old item, as no longer/replaced", ok)] += 1
                ok or miss["preference update: old item"].append((p, s["session_id"], d["old_item"]))
                ok = match_text(item, [r for r in rs if r.status in ("new", "updated", "added")])
                c[("preference update: new item, as new/updated", ok)] += 1
            else:
                to_dislike = d.get("preference") == "dislike"
                ok = match_text(item, [r for r in rs if (r.status == "no longer") == to_dislike])
                c[(f"preference update: same item, now {d.get('preference')}", ok)] += 1
                ok or miss["preference update: same item"].append((p, s["session_id"], f"{item} -> {d.get('preference')}"))
    elif t == "goal":
        ok = match_number(d.get("item"), rs)
        c[("goal: amount", ok)] += 1
    elif isinstance(d.get("item"), dict):
        cat, it = d.get("category"), d["item"]
        if cat == "food_expenses":
            ok = match_number(it.get("amount"), rs)
        elif cat == "step_tracker":
            ok = match_number(it.get("step_count"), rs)
        elif cat == "todo_list":
            want = ("done", "removed") if op == "delete" else ("new", "added", "updated")
            ok = match_words(it.get("description"), [r for r in rs if r.status in want])
        elif cat == "calendar_event":
            ok = match_words(it.get("event_name"), rs)
        else:
            continue
        c[(f"{cat} {op}", ok)] += 1
        ok or miss[f"{cat} {op}"].append((p, s["session_id"], it.get("description") or it.get("event_name") or it.get("amount") or it.get("step_count")))
for name in sorted({k for k, _ in c}):
    a, n = c[(name, True)], c[(name, True)] + c[(name, False)]
    print(f"  {name:<52} {a:4d} of {n:4d} {pct(a, n)}")
print("  not matched (up to 4 each):")
for name, xs in sorted(miss.items()):
    for p, sid, x in xs[:4]:
        print(f"    {name}: {p} session {sid}: {str(x)[:140]}")
print()

print("4. THE TABLE at the end of each persona's history")
for persona, rs in all_rows:
    t = build_table(rs)
    last = max(r.session for r in rs) if rs else 0
    now = true_at(t, last)
    print(f"  {persona}: {len(t)} entries; holding at the end {len(now)}: " + ", ".join(f"{k} {n}" for k, n in Counter(e["state"] for e in now).most_common()))
print()

random.seed(26)
print("5. HAND-CHECK SAMPLE: 30 sessions at random, each flagged line then its rows")
print("   (detail | qualifier | field | who | kind | status | number unit | resolved date)")
for key in random.sample(sorted(by_session), min(30, len(by_session))):
    s, rs = by_session[key]
    print(f"- {key[0]} session {key[1]} ({s.get('session_type')}, {s.get('operation')}, {s.get('date')})")
    lines = {i + 1: t for i, t in enumerate(s["conversation"])}
    for n, t in lines.items():
        if t.get("share_memory") and t.get("speaker") == "user_agent":
            print(f"    {n}. {t['message']}")
            for r in [r for r in rs if r.line == n]:
                extra = f" | {r.number:g} {r.unit or ''}".rstrip() if r.number is not None else ""
                extra += f" | {r.resolved}" if r.resolved else ""
                print(f"        * {r.detail} | {r.qualifier} | {r.field} | {r.who} | {r.kind} | {r.status}{extra}")
