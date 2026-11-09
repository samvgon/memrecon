"""The code-grader fixes of D62, checked where they matter: every verdict they change.

D62 fixed three things in the code grader (src/memrecon/probes.py) found in the Stage 0 check (E40): a name
followed by 's ("Victor Fleming's films"), a short item matched only as an exact phrase ("quick $5.56
breakfast" against "a quick breakfast for $5.56"), and a short parent required as an exact phrase ("update my
professional certifications" against "update their professional certifications"). Re-checking the fixed code
on E40's labels would grade it on the very items the fixes came from, so instead this recomputes the code's
verdict on every probe in every end-of-history store of the frozen control run (as make_grader_sample.py),
with the grader as it was before D62 (`--before`, a git revision; loaded from git) and as it is now, and
writes out every item whose verdict changed, for Sam to label. Each shows the detail, its field and parent,
the user's sentence it came from (from the answer key's row, not a search), the session's user lines, and five
memories: the one the code now matches (or matched before), and the four most like the detail.

Needs E30's and E31's rows in results/probes/ and the control run's snapshots (run on the EVO-X2).
Usage:  uv run python scripts/make_grader_recheck.py --before <revision before D62>
Writes notes/labels/<date>-grader-recheck.jsonl (refuses to overwrite one) and prints how many verdicts changed.
"""
import argparse
import datetime
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "explore"))
from common import flagged_turns, iter_sessions, record_key  # noqa: E402
from memrecon.memora import ROOT, dev_personas  # noqa: E402
from memrecon.memory import MemoryStore  # noqa: E402
from memrecon.probes import (LONG_WORDS, OVERLAP, Row, build_table, content_words, grading_view,  # noqa: E402
                             match_with_parent, numbers_in, parent_entry, true_at)
from memrecon.recall import REWRITE_PROMPT, SELECT_PROMPT  # noqa: E402

SHOWN = 5
CONTROL = ROOT / "results" / "stage0" / "control"
PROBES = ROOT / "results" / "probes"

ap = argparse.ArgumentParser()
ap.add_argument("--before", required=True, help="git revision with the grader as it was before D62")
ap.add_argument("--config", default=None)
args = ap.parse_args()
out = ROOT / "notes" / "labels" / f"{datetime.date.today().isoformat()}-grader-recheck.jsonl"
if out.exists():
    sys.exit(f"{out.name} already exists; it is frozen. Not overwritten.")

old_src = subprocess.run(["git", "-C", str(ROOT), "show", f"{args.before}:src/memrecon/probes.py"], capture_output=True, text=True, check=True).stdout
tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
tmp.write(old_src)
tmp.close()
spec = importlib.util.spec_from_file_location("memrecon._probes_before_d62", tmp.name)
old = importlib.util.module_from_spec(spec)
old.__package__ = "memrecon"
sys.modules[spec.name] = old  # dataclasses look the module up while it runs
spec.loader.exec_module(old)
if not hasattr(old, "match_with_parent") or hasattr(old, "parent_found"):
    sys.exit(f"{args.before} does not look like the grader before D62.")


def current_config():
    for p in sorted(CONTROL.iterdir()):
        f = p / "config.json"
        if f.exists():
            c = json.loads(f.read_text())
            if c.get("sessions") == "all" and c.get("rewrite_prompt") == REWRITE_PROMPT and c.get("select_prompt") == SELECT_PROMPT \
                    and (p / "report.md").exists():
                return p.name
    sys.exit("No finished full control run with the current prompts.")


def load_rows(name, persona):
    rows = []
    for x in (PROBES / f"{name}_{persona}.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            d = json.loads(x)
            d.pop("persona", None)
            d["record"] = tuple(d["record"])
            rows.append(Row(**d))
    return rows


def likeness(p, parent, text):
    want = content_words(p["detail"]) | (content_words(parent["detail"]) if parent else set())
    share = len(want & content_words(text)) / max(1, len(want))
    nums = numbers_in(p["detail"])
    return share + (1.0 if nums and nums & numbers_in(text) else 0.0)


def why(p, parent):
    if "'s" in p["detail"] or p["kind"] in ("name", "title"):
        return "name or title (possessive 's)"
    if parent is not None:
        return "attached detail (short parent)" if len(parent["detail"].split()) < LONG_WORDS else "attached detail"
    return "short item (content words, any order)" if p["kind"] == "item" and len(p["detail"].split()) < LONG_WORDS else "other"


config = args.config or current_config()
changed, totals = [], Counter()
for persona in dev_personas():
    rows = load_rows("E30", persona) + load_rows("E31", persona)
    by_id = {r.row_id: r for r in rows if r.row_id}
    sessions = {s["session_id"]: s for _, _, s in iter_sessions(persona)}
    with_rows = {r.session for r in rows}
    failed = {sid for sid, s in sessions.items() if s.get("operation") in ("add", "update", "delete") and flagged_turns(s) and sid not in with_rows}
    bad = set()
    for sid in failed:
        k = record_key(sessions[sid])
        bad.add(k if k is not None else ("session", sid))
    entries = build_table([r for r in rows if r.record not in bad])
    by_row = {e["row_id"]: e for e in entries if e.get("row_id")}
    last = max(sessions)
    probes = [p for p in grading_view(true_at(entries, last)) if p["grade"] == "code"]
    for arm in ("note", "log"):
        run = sorted((CONTROL / config / persona / arm).glob("run_*"), key=lambda p: int(p.name[4:]))[-1]
        snap = run / "B" / "snapshots" / f"session_{last:03d}.json"
        live = MemoryStore.load(snap).active()
        for p in probes:
            parent = parent_entry(p, by_row)
            new_hits = [m for m in live if match_with_parent(p, m.text, parent, OVERLAP)]
            old_hits = [m for m in live if old.match_with_parent(p, m.text, parent, OVERLAP)]
            totals[(arm, "before")] += bool(old_hits)
            totals[(arm, "now")] += bool(new_hits)
            totals[(arm, "probes")] += 1
            if bool(new_hits) == bool(old_hits):
                continue
            hit = (new_hits or old_hits)[0]
            ranked = sorted(live, key=lambda m: -likeness(p, parent, m.text))
            shown = [hit] + [m for m in ranked if m.id != hit.id][: SHOWN - 1]
            row = by_id.get(p.get("row_id"))
            s = sessions[p["from_session"]]
            users = [(n, t["message"]) for n, t in enumerate(s["conversation"], start=1) if t.get("speaker") == "user_agent"]
            said = s["conversation"][row.line - 1]["message"] if row and 0 < row.line <= len(s["conversation"]) else ""
            changed.append({"persona": persona, "arm": arm, "kind": p["kind"], "state": p["state"], "field": p.get("field", ""),
                            "detail": p["detail"], "attached_to": parent["detail"] if parent else "", "said_on": p.get("from_date", ""),
                            "session": p["from_session"], "line": row.line if row else None, "said": said,
                            "user_lines": [f"{n}. {m}" for n, m in users], "why": why(p, parent),
                            "before": bool(old_hits), "now": bool(new_hits), "code_memory": hit.id,
                            "memories": [{"id": m.id, "date": m.date, "text": m.text} for m in shown],
                            "probe": p.get("row_id"), "snapshot": f"{config}/{persona}/{arm}/{snap.name}"})

print(f"Control run {config}; grader before D62: {args.before}")
for arm in ("note", "log"):
    print(f"  {arm}: {totals[(arm, 'probes')]} probes in end-of-history stores; found before {totals[(arm, 'before')]}, now {totals[(arm, 'now')]}")
print(f"Verdicts changed: {len(changed)} (not found -> found {sum(c['now'] for c in changed)}, found -> not found {sum(not c['now'] for c in changed)})")
for k, v in sorted(Counter((c["arm"], c["why"]) for c in changed).items()):
    print(f"  {k[0]:5} {k[1]:38} {v}")
import random  # noqa: E402
random.Random(62).shuffle(changed)
for i, c in enumerate(changed, start=1):
    c["item"] = f"R{i:02d}"
    random.Random(62 + i).shuffle(c["memories"])
out.write_text("".join(json.dumps(c) + "\n" for c in changed), encoding="utf-8")
print(f"-> {out.relative_to(ROOT)}")
