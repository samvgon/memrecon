"""E31: the split pass on version 4's rows: which long details with "and" are several things?

Question (D49; Sam's idea from the hand checks of versions 3 and 4): some details join separate things
("provide a preliminary financial analysis ... and outline key considerations ..."), which should be
separate probes, but most details with "and" are one thing ("account management and fraud detection
modules"), so code cannot split them. For each row of E30 (results/probes/E30_<persona>.jsonl, development
personas) that is at least 6 words long, contains "and", is not a number, date or name, and holds rather
than removes or finishes something, the judge (prompts/probe_split_v1.md) answers with an empty list (one
thing) or the parts, each copied exactly from the detail (code checks). How many candidates are there, how
many are split, into how many parts, and do the splits look right? How does the table change when the
parts are added and closing a row also closes what belongs to it (D49)? And, for Sam's hand check, the
decisions on E27's 37 sessions.
Method: E30's rows are read, never changed; each part becomes a new row attached to its whole (row_id
<whole>.<k>, status "new", origin "split"), written to results/probes/E31_<persona>.jsonl; every decision
(one thing, or the parts) to results/probes/E31_decisions_<persona>.jsonl; the hand-check sample to
results/probes/E31_handcheck.json. Calls are cached.
Pilot: MEMRECON_E31_PILOT=N asks only the first N candidates of the first persona (not evidence).
Limits: the candidate rule (6 words, "and") misses separate things joined only by commas, and long
details joined by "or"; the judge's "one thing" is not checked by code.
"""
import json
import os
import time
from collections import Counter

from common import ROOT, iter_sessions
from memrecon.llm import client as model_client
from memrecon.probes import ProbeSplitter, Row, build_table, split_candidate, true_at

PILOT = int(os.environ.get("MEMRECON_E31_PILOT", "0") or 0)
OUT = ROOT / "results" / "probes"


def load_rows(persona):
    rows = []
    for x in (OUT / f"E30_{persona}.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            d = json.loads(x)
            d.pop("persona", None)
            d["record"] = tuple(d["record"])
            rows.append(Row(**d))
    return rows


split = json.loads((ROOT / "notes" / "split.json").read_text())
judge = model_client("judge")
personas = split["dev"][:1] if PILOT else split["dev"]
start = time.monotonic()
decisions, parts_all, base_all, calls, feedback, failed = [], {}, {}, [], Counter(), []
for persona in personas:
    rows = load_rows(persona)
    base_all[persona] = rows
    lines = {(s["session_id"], n): t["message"] for _, _, s in iter_sessions(persona)
             for n, t in enumerate(s.get("conversation", []), 1)}
    cands = [r for r in rows if split_candidate(r)]
    if PILOT:
        cands = cands[:PILOT]
    sp = ProbeSplitter(judge, persona=persona, tag="E31")
    parts_p, dec_p = [], []
    for r in cands:
        out = sp.split(r, lines[(r.session, r.line)])
        if not out.ok:
            failed.append((persona, r, out.error))
        dec = {"persona": persona, "row_id": r.row_id, "session": r.session, "line": r.line, "field": r.field, "kind": r.kind,
               "detail": r.detail, "ok": out.ok, "parts": [p.detail for p in out.rows], "attempts": out.attempts, "error": out.error}
        dec_p.append(dec)
        parts_p += out.rows
    with open(OUT / f"E31_{persona}.jsonl", "w", encoding="utf-8") as f:
        for p in parts_p:
            f.write(json.dumps({"persona": persona, **p.to_json()}) + "\n")
    with open(OUT / f"E31_decisions_{persona}.jsonl", "w", encoding="utf-8") as f:
        for dec in dec_p:
            f.write(json.dumps(dec, ensure_ascii=False) + "\n")
    decisions += dec_p
    parts_all[persona] = parts_p
    calls += sp.calls
    feedback.update(x[:90] for x in sp.feedback)
elapsed = time.monotonic() - start

if PILOT:
    print(f"PILOT RUN: not evidence. {len(decisions)} candidates in {elapsed:.0f} s, {sum(1 for c in calls if not c['cached'])} fresh calls\n")

print("1. RELIABILITY")
print(f"  {len(decisions)} candidates asked, {len(failed)} failed (kept whole), attempts {dict(sorted(Counter(d['attempts'] for d in decisions).items()))}, "
      f"{sum(1 for c in calls if not c['cached'])} fresh calls, {elapsed:.0f} s")
for p, r, e in failed:
    print(f"    failed: {p} {r.row_id}: {e}")
print("  why answers were sent back: " + ("; ".join(f"{n} x {r}" for r, n in feedback.most_common()) or "never"))
print()

total = sum(len(v) for v in base_all.values())
splits = [d for d in decisions if d["parts"]]
print("2. CANDIDATES AND DECISIONS")
print(f"  {total} rows in E30; candidates (6+ words, 'and', not number/date/name, holding): {len(decisions)}")
print("  candidates by kind: " + ", ".join(f"{k} {n}" for k, n in Counter(d["kind"] for d in decisions).most_common()))
print(f"  split: {len(splits)} ({len(splits) / max(1, len(decisions)):.0%}); one thing: {sum(1 for d in decisions if d['ok'] and not d['parts'])}")
print("  parts per split: " + ", ".join(f"{k} parts: {n}" for k, n in sorted(Counter(len(d["parts"]) for d in splits).items())))
print("  split by kind: " + ", ".join(f"{k} {n}" for k, n in Counter(d["kind"] for d in splits).most_common())
      + "; by field (top 10): " + ", ".join(f"{k} {n}" for k, n in Counter(d["field"] for d in splits).most_common(10)))
print("  with ', and' in the detail: split {} of {}; without: split {} of {}".format(
    sum(1 for d in splits if ", and" in d["detail"]), sum(1 for d in decisions if ", and" in d["detail"]),
    sum(1 for d in splits if ", and" not in d["detail"]), sum(1 for d in decisions if ", and" not in d["detail"])))
print()

print("3. EVERY SPLIT (whole -> parts)")
for d in splits:
    print(f"  - {d['persona']} {d['row_id']} [{d['kind']}, {d['field']}] {d['detail']}")
    for p in d["parts"]:
        print(f"        > {p}")
print()
print("4. KEPT WHOLE (up to 40, in order)")
for d in [d for d in decisions if d["ok"] and not d["parts"]][:40]:
    print(f"  - {d['persona']} {d['row_id']} [{d['kind']}, {d['field']}] {d['detail']}")
print()

print("5. THE TABLE at the end of each persona's history: E30's rows alone, then with the parts (both with D49's closing)")
for persona, rows in base_all.items():
    last = max(r.session for r in rows)
    a = true_at(build_table(rows), last)
    b = true_at(build_table(rows + parts_all[persona]), last)
    print(f"  {persona}: holding at the end {len(a)} -> {len(b)}; "
          + ", ".join(f"{k} {n}" for k, n in sorted(Counter(e['state'] for e in b).items())))
print()

print("6. HAND-CHECK SAMPLE: decisions on E27's 37 sessions")
picked = {(x["persona"], x["session"]) for x in json.loads((OUT / "E27_handcheck.json").read_text(encoding="utf-8"))}
sample = [d for d in decisions if (d["persona"], d["session"]) in picked]
for d in sample:
    print(f"  - {d['persona']} session {d['session']} line {d['line']} [{d['row_id'].rsplit(':', 1)[-1]}] {d['detail']}")
    print("        " + ("one thing" if d["ok"] and not d["parts"] else "FAILED (kept whole)" if not d["ok"] else "split: " + " | ".join(d["parts"])))
(OUT / "E31_handcheck.json").write_text(json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nwritten: results/probes/E31_handcheck.json")
