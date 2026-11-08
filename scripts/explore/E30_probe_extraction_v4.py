"""E30: the probe extractor, version 4, on the development personas, and the same hand-check sessions.

Question (D48): version 4 (prompts/probe_extract_v4.md) changes what Sam's hand check of version 3 found
wrong (notes/labels/2026-11-07-probe-handcheck-v3.jsonl, 116 of 152 rows ok): "who" is who does the
thing, never the user asking for a document to be edited; the kinds are defined one by one ("name" a
proper name only, "date" includes deadlines, durations and timeframes, which now carry a parsed number);
the numbers, dates and timeframes inside a longer item are listed too, attached to it, and things joined
by "and" are split; "replaced" is refused for an old value shown as an E row; "added" is dropped. Does it
stay reliable and agree with the log as version 3 did (E29)? How do who, kinds and the details inside
items change? And, for Sam's hand check, the same 37 sessions as E27 with version 4's rows.
Method: as E29 (sessions in file order, A6; records as A10, goals by subcategory; the flags choose the
lines to read; the model sees only line numbers). Rows are written to results/probes/E30_<persona>.jsonl;
the hand-check sample to results/probes/E30_handcheck.json. Calls are cached.
Pilot: MEMRECON_E30_PILOT=N extracts only the first N sessions of the first persona (not evidence).
Limits: as E29; what is shown as already recorded depends on the model's own earlier rows.
"""
import json
import os
import re
import time
from collections import Counter, defaultdict

from common import ROOT, flagged_turns, iter_sessions, record_key
from memrecon.llm import client as model_client
from memrecon.probes import ProbeExtractor, build_table, norm, true_at

PILOT = int(os.environ.get("MEMRECON_E30_PILOT", "0") or 0)
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
    ex = ProbeExtractor(judge, persona=persona, tag="E30", prompt="probe_extract_v4")
    sessions = [s for _, _, s in iter_sessions(persona) if s.get("operation") in ("add", "update", "delete") and flagged_turns(s)]
    if PILOT:
        sessions = sessions[:PILOT]
    rows_p = []
    for s in sessions:
        out = ex.extract(s, record_of(s))
        outcomes.append((persona, s, out))
        by_session[(persona, s["session_id"])] = (s, out.rows)
        rows_p += out.rows
    with open(OUT / f"E30_{persona}.jsonl", "w", encoding="utf-8") as f:
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
named = [r for r in rows if r.who not in ("user", "user and others", "")]
label = lambda w: w if w in ("user", "user and others") else ("(empty)" if not w else "(a name)")  # noqa: E731
print("  who:    " + ", ".join(f"{k} {n}" for k, n in Counter(label(r.who) for r in rows).most_common())
      + "; names: " + (", ".join(f"{k} {n}" for k, n in Counter(r.who for r in named).most_common(10)) or "none"))
by_id = {r.row_id: r for r in rows}


def link_kind(r, target):
    """What a row's attached_to or replaces points at: R, unknown, same session, earlier session, none."""
    if not target:
        return "none"
    if target == "unknown":
        return "unknown"
    if ":R:" in target:
        return "R (the thing itself)"
    t = by_id.get(target)
    if t is None:
        return "(missing)"
    where = "same session" if t.session == r.session else "earlier session"
    return where + ("" if t.record == r.record else ", another record")


print("  attached to: " + ", ".join(f"{k} {n}" for k, n in Counter(link_kind(r, r.attached_to) for r in rows).most_common()))
att = [r for r in rows if r.attached_to in by_id]
print("  kinds attached to a detail: " + ", ".join(f"{k} {n}" for k, n in Counter(r.kind for r in att).most_common())
      + "; attached to kinds: " + ", ".join(f"{k} {n}" for k, n in Counter(by_id[r.attached_to].kind for r in att).most_common()))
rep = [r for r in rows if r.status in ("updated", "removed", "done", "no longer")]
print(f"  replacing statuses: {len(rep)}; naming the row replaced: " + ", ".join(f"{k} {n}" for k, n in Counter(link_kind(r, r.replaces) for r in rep).most_common())
      + "; by status (named of all): " + ", ".join(f"{st} {sum(1 for r in rep if r.status == st and r.replaces)} of {sum(1 for r in rep if r.status == st)}"
                                                    for st in ("updated", "removed", "done", "no longer")))
print(f"  'replaced' rows (an old value not recorded as a row): {sum(1 for r in rows if r.status == 'replaced')}")
inner = [r for r in rows if r.attached_to in by_id and by_id[r.attached_to].kind in ("item", "task") and r.kind in ("number", "date")]
print(f"  numbers and dates attached to an item or task (details inside a longer item, or a task's deadline): {len(inner)}; "
      + ", ".join(f"{k} {n}" for k, n in Counter(f"{r.kind} in {by_id[r.attached_to].kind}" for r in inner).most_common()))
print("  who by kind: " + "; ".join(f"{k}: " + ", ".join(f"{w or '(empty)'} {n}" for w, n in Counter(
    (r.who if r.who in ("", "user", "user and others") else "(a name)") for r in rows if r.kind == k).most_common()) for k in sorted({r.kind for r in rows})))
dates = [r for r in rows if r.kind == "date"]
print(f"  dates with a duration parsed: {sum(1 for r in dates if r.number is not None)} of {len(dates)}: "
      + ", ".join(sorted({r.detail for r in dates if r.number is not None})[:20]))
generic = Counter(r.field for r in rows if r.field.lower() in ("title", "name", "date", "item", "time", "preference"))
print("  bare field names (title, name, date, item, time, preference): " + (", ".join(f"{k} {n}" for k, n in generic.most_common()) or "none"))
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
            ok = match_text(item, [r for r in rs if r.status in ("new", "added")]) or match_words(item, [r for r in rs if r.status in ("new", "added")])
            c[("preference add: item, as new", ok)] += 1
            ok or miss["preference add"].append((p, s["session_id"], item))
        elif op == "delete":
            ok = match_text(item, [r for r in rs if r.status in ("no longer", "removed")]) or match_words(item, [r for r in rs if r.status in ("no longer", "removed")])
            c[("preference delete: item, as no longer", ok)] += 1
            ok or miss["preference delete"].append((p, s["session_id"], item))
        else:
            if d.get("old_item"):
                olds = [r for r in rs if r.status in ("no longer", "replaced", "removed")]
                ok = match_text(d["old_item"], olds) or match_words(d["old_item"], olds)
                c[("preference update: old item, as no longer/replaced", ok)] += 1
                ok or miss["preference update: old item"].append((p, s["session_id"], d["old_item"]))
                news = [r for r in rs if r.status in ("new", "updated", "added")]
                ok = match_text(item, news) or match_words(item, news)
                c[("preference update: new item, as new/updated", ok)] += 1
            else:
                to_dislike = d.get("preference") == "dislike"
                same = [r for r in rs if (r.status == "no longer") == to_dislike]
                ok = match_text(item, same) or match_words(item, same)
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

print("4. THE TABLE at the end of each persona's history, and updates the links do not reach")
for persona, rs in all_rows:
    t = build_table(rs)
    last = max(r.session for r in rs) if rs else 0
    now = true_at(t, last)
    attached_fields = {(r.record, r.field) for r in rs if r.attached_to}
    loose = [r for r in rs if r.status == "updated" and not r.attached_to and (r.record, r.field) in attached_fields]
    print(f"  {persona}: {len(t)} entries; holding at the end {len(now)}: " + ", ".join(f"{k} {n}" for k, n in Counter(e["state"] for e in now).most_common())
          + f"; updates with no link in a field whose values are linked (closing every earlier value of the field): {len(loose)}")
    for r in loose[:4]:
        print(f"      session {r.session}: {r.field}: {r.detail}")
print()

print("5. HAND-CHECK SAMPLE: E27's sessions with version 4's rows (each flagged line, then its rows)")
print("   ([id] detail | qualifier | field | who | kind | status | number unit | resolved date | attached to | replaces)")
print("   (an id from another session is shown as s<session>:<id> with its detail; R is the thing itself)")
picked = json.loads((OUT / "E27_handcheck.json").read_text(encoding="utf-8"))
sample = []
for sk in picked:
    key = (sk["persona"], sk["session"])
    if key not in by_session:
        continue
    s, rs = by_session[key]
    lines = []
    print(f"- {key[0]} session {key[1]} ({s.get('session_type')}, {s.get('operation')}, said on {s.get('date')}) [picked for: {sk['why']}]")
    for n, t in enumerate(s["conversation"], 1):
        if not (t.get("share_memory") and t.get("speaker") == "user_agent"):
            continue
        print(f"    {n}. {t['message']}")
        lrows = []
        for r in [r for r in rs if r.line == n]:
            num = f"{r.number:,.6g} {r.unit or ''}".strip() if r.number is not None else ""

            def show(x, r=r):
                if not x:
                    return ""
                if x == "unknown" or ":R:" in x:
                    return "unknown" if x == "unknown" else "R"
                t = by_id.get(x)
                if t is None or t.session == r.session:
                    return x.rsplit(":", 1)[-1]
                return f"s{t.session}:{x.rsplit(':', 1)[-1]} ({t.detail})"
            print(f"        * [{r.row_id.rsplit(':', 1)[-1]}] {r.detail} | {r.qualifier} | {r.field} | {r.who} | {r.kind} | {r.status} | {num} | {r.resolved or ''} | {show(r.attached_to)} | {show(r.replaces)}")
            lrows.append({"id": r.row_id.rsplit(":", 1)[-1], "detail": r.detail, "qualifier": r.qualifier, "field": r.field, "who": r.who, "kind": r.kind,
                          "status": r.status, "number": num, "resolved": r.resolved or "", "said_on": r.date, "attached_to": show(r.attached_to),
                          "replaces": show(r.replaces)})
        lines.append({"n": n, "text": t["message"], "rows": lrows})
    sample.append({"persona": key[0], "session": key[1], "type": s.get("session_type"), "op": s.get("operation"), "date": s.get("date"),
                   "why": sk["why"], "lines": lines})
(OUT / "E30_handcheck.json").write_text(json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nwritten: results/probes/E30_handcheck.json")
