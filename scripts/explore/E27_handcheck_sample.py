"""E27: a stratified hand-check sample of E26's probe rows, so every case to be judged appears.

Question (D44, D45; Sam's hand check of the answer key): E26 printed 30 sessions at random, which
showed mostly the common case (every "who" in it was "user"). Which sessions to hand-check so that each
case the extractor is meant to handle appears at least a few times? Strata, each filled with up to N
rows' sessions at random (seed 27): every status (new, added, updated, replaced, removed, done, no
longer); who "user and others" and "someone else"; every kind; qualifiers on the hedge list (graded)
and off it (not graded, D45); dates resolved and not resolved; numbers with a unit; document update and
delete sessions; preference updates that switch item and that flip like/dislike; flagged lines with no
rows (possible misses); then random sessions. A session is shown whole (every flagged line and its rows)
and is counted once, under the first stratum that picked it. Strata are filled rarest first, and a stratum
only gets more sessions while fewer than N of those already picked show it, so the sample stays short.
Method: reads E26's rows (results/probes/E26_<persona>.jsonl, development personas) and the sessions;
no model calls. Prints the sample and writes it as JSON to results/probes/E27_handcheck.json for the
hand-check page.
Limits: strata overlap, so a session picked for one stratum also shows rows of others; counts per
stratum are rows found, not a measure of anything.
"""
import json
import random
from collections import defaultdict

from common import ROOT, flagged_turns, iter_sessions
from memrecon.probes import PRIMARY_KINDS, graded_qualifier

PER_STRATUM = 3  # N
RANDOM_SESSIONS = 6
split = json.loads((ROOT / "notes" / "split.json").read_text())
rows = []
for persona in split["dev"]:
    path = ROOT / "results" / "probes" / f"E26_{persona}.jsonl"
    rows += [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
sessions = {(p, s["session_id"]): s for p in split["dev"] for _, _, s in iter_sessions(p)}
by_session = defaultdict(list)
for r in rows:
    by_session[(r["persona"], r["session"])].append(r)
extracted = sorted(k for k in sessions if sessions[k].get("operation") in ("add", "update", "delete") and flagged_turns(sessions[k]))

strata = []
for st in ("new", "added", "updated", "replaced", "removed", "done", "no longer"):
    strata.append((f"status {st}", lambda r, st=st: r["status"] == st))
strata += [("who: user and others", lambda r: r["who"] == "user and others"),
           ("who: someone else", lambda r: r["who"] == "someone else")]
for k in ("number", "date", "name", "title", "item", "task", "preference"):
    strata.append((f"kind {k}", lambda r, k=k: r["kind"] == k))
strata += [("qualifier on the hedge list (graded)", lambda r: bool(graded_qualifier(r["qualifier"]))),
           ("qualifier off the list (not graded)", lambda r: bool(r["qualifier"]) and not graded_qualifier(r["qualifier"])),
           ("date resolved to a day", lambda r: r["kind"] == "date" and r["resolved"]),
           ("date not resolved", lambda r: r["kind"] == "date" and not r["resolved"]),
           ("number with a unit", lambda r: r["kind"] == "number" and r["unit"])]
session_strata = [
    ("document update", lambda s: s.get("session_type") == "activity" and s.get("operation") == "update" and (s.get("operation_details") or {}).get("category") is None),
    ("document delete", lambda s: s.get("session_type") == "activity" and s.get("operation") == "delete" and (s.get("operation_details") or {}).get("category") is None),
    ("preference update, item switched", lambda s: s.get("session_type") == "preference" and s.get("operation") == "update" and (s.get("operation_details") or {}).get("old_item")),
    ("preference update, like/dislike flipped", lambda s: s.get("session_type") == "preference" and s.get("operation") == "update" and not (s.get("operation_details") or {}).get("old_item")),
]

random.seed(27)
picked, reason, found = [], {}, {}


def top_up(name, keys):
    """Add sessions from `keys` until PER_STRATUM picked sessions show this stratum."""
    have = sum(1 for k in picked if k in keys)
    pool = [k for k in keys if k not in reason]
    for k in random.sample(pool, max(0, min(PER_STRATUM - have, len(pool)))):
        picked.append(k)
        reason[k] = name


# rarest strata first, so their sessions also cover the common ones
row_strata = sorted(strata, key=lambda st: sum(1 for r in rows if st[1](r)))
for name, test in row_strata:
    found[name] = sum(1 for r in rows if test(r))
    top_up(name, {(r["persona"], r["session"]) for r in rows if test(r)})
for name, test in session_strata:
    keys = {k for k in extracted if test(sessions[k])}
    found[name] = len(keys)
    top_up(name, keys)
missed = {k for k in extracted if any(n not in {r["line"] for r in by_session[k]} for n, t in enumerate(sessions[k]["conversation"], 1)
                                      if t.get("share_memory") and t.get("speaker") == "user_agent")}
found["flagged line with no rows"] = len(missed)
top_up("flagged line with no rows", missed)
rest = [k for k in extracted if k not in reason]
for k in random.sample(rest, min(RANDOM_SESSIONS, len(rest))):
    picked.append(k)
    reason[k] = "random"

print(f"{len(rows)} rows from {len(extracted)} sessions; {len(picked)} sessions picked\n")
print("per stratum: rows (or sessions) found, sessions in the sample that show it, sessions picked for it")
shows = {n: t for n, t in strata}
for name in [n for n, _ in strata] + [n for n, _ in session_strata] + ["flagged line with no rows", "random"]:
    if name in shows:
        n_show = sum(1 for k in picked if any(shows[name](r) for r in by_session[k]))
    elif name == "flagged line with no rows":
        n_show = sum(1 for k in picked if k in missed)
    elif name == "random":
        n_show = sum(1 for k in picked if reason[k] == "random")
    else:
        n_show = sum(1 for k in picked if dict(session_strata)[name](sessions[k]))
    print(f"  {name:<44} found {found.get(name, len(rest)):4d}, shown in {n_show:3d}, picked for it {sum(1 for k in picked if reason[k] == name)}")
print()

out = []
for k in picked:
    s = sessions[k]
    lines = []
    for n, t in enumerate(s["conversation"], 1):
        if not (t.get("share_memory") and t.get("speaker") == "user_agent"):
            continue
        rs = [r for r in by_session[k] if r["line"] == n]
        lines.append({"n": n, "text": t["message"], "rows": [
            {"detail": r["detail"], "qualifier": r["qualifier"], "field": r["field"], "who": r["who"], "kind": r["kind"],
             "status": r["status"], "number": "" if r["number"] is None else f"{r['number']:g} {r['unit'] or ''}".strip(),
             "resolved": r["resolved"] or "", "said_on": r["date"],
             "grade": "code" if r["kind"] in PRIMARY_KINDS and r["status"] not in ("done", "no longer") else "judge"} for r in rs]})
    out.append({"persona": k[0], "session": k[1], "type": s.get("session_type"), "op": s.get("operation"), "date": s.get("date"),
                "why": reason[k], "lines": lines})
    print(f"- {k[0]} session {k[1]} ({s.get('session_type')}, {s.get('operation')}, said on {s.get('date')}) [picked for: {reason[k]}]")
    for ln in lines:
        print(f"    {ln['n']}. {ln['text']}")
        for r in ln["rows"]:
            print(f"        * {r['detail']} | {r['qualifier']} | {r['field']} | {r['who']} | {r['kind']} | {r['status']} | {r['number']} | {r['resolved']}")
(ROOT / "results" / "probes" / "E27_handcheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nwritten: results/probes/E27_handcheck.json")
