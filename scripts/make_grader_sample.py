"""Make the frozen sample of code-grader items for Sam to label (D21: the code graders face hand labels too).

An item is one probe (a detail from the frozen answer key, D50, D51, that code grades: kinds number, date,
name, title and item) at the end of a development persona's history, in one arm's end-of-history store from
the frozen Stage 0 control run (E38). The code grader (`match_with_parent`, D50, D51) decides whether any
live memory holds it. Sam is shown the detail (with what it is attached to, if anything) and five memories,
and labels whether one of them states it. The five are the memory the code matched, if any, and the others
most like the detail by shared words and numbers, in random order; she is not told the code's verdict or
which memory it matched.

Stratified (CLAUDE.md): details that should be in memory (state "true") by arm (note, log), by the code's
verdict (found, not found) and by kind (numbers; dates; short names, titles and items, matched exactly; long
items, matched by 80% word overlap), 4 items per cell where it has them, half of them near misses (found only
through an alias, a date form or partial overlap; or not found though a memory shares most of its words or one
of its numbers), the rest at random; plus, for each arm, 6 old values (state "false": should no longer be in
memory), half of them ones the code found. Seed 40.
Version 2 (Nov 8, before any labelling): version 1 (`2026-11-08-grader-sample.jsonl`, kept unused) put numbers
and dates in one group and old values in none, and drew 2 numbers and 1 old value of 55 items.

Needs E30's and E31's rows in results/probes/ and the control run's snapshots (run on the EVO-X2).
Usage:  uv run python scripts/make_grader_sample.py [--config <control config id>]
Writes notes/labels/<date>-grader-sample-v2.jsonl and refuses to overwrite one: the sample is frozen once made.
"""
import argparse
import datetime
import json
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "explore"))
from common import flagged_turns, iter_sessions, record_key  # noqa: E402
from memrecon.memora import ROOT, dev_personas  # noqa: E402
from memrecon.memory import MemoryStore  # noqa: E402
from memrecon.probes import (LONG_WORDS, OVERLAP, Row, build_table, content_words, grading_view,  # noqa: E402
                             match_with_parent, norm, numbers_in, parent_entry, true_at)
from memrecon.recall import REWRITE_PROMPT, SELECT_PROMPT  # noqa: E402

SEED = 40
PER_CELL = 4
OLD_PER_ARM = 6
SHOWN = 5
CONTROL = ROOT / "results" / "stage0" / "control"
PROBES = ROOT / "results" / "probes"

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
args = ap.parse_args()
labels_dir = ROOT / "notes" / "labels"
if list(labels_dir.glob("*-grader-sample-v2.jsonl")):
    sys.exit("A grader sample already exists in notes/labels/; it is frozen (D21). Not overwritten.")


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


def group(p):
    if p["kind"] in ("number", "date"):
        return p["kind"]
    return "long item" if len(p["detail"].split()) >= LONG_WORDS else "short name, title or item"


def likeness(p, parent, text):
    want = content_words(p["detail"]) | (content_words(parent["detail"]) if parent else set())
    share = len(want & content_words(text)) / max(1, len(want))
    nums = numbers_in(p["detail"])
    return share + (1.0 if nums and nums & numbers_in(text) else 0.0), share


config = args.config or current_config()
items = []
for persona in dev_personas():
    rows = load_rows("E30", persona) + load_rows("E31", persona)
    sessions = [s for _, _, s in iter_sessions(persona)]
    with_rows = {r.session for r in rows}
    failed = {s["session_id"] for s in sessions if s.get("operation") in ("add", "update", "delete")
              and flagged_turns(s) and s["session_id"] not in with_rows}
    bad = set()
    for s in sessions:
        if s["session_id"] in failed:
            k = record_key(s)
            bad.add(k if k is not None else ("session", s["session_id"]))
    entries = build_table([r for r in rows if r.record not in bad])
    by_row = {e["row_id"]: e for e in entries if e.get("row_id")}
    last = sessions[-1]["session_id"]
    probes = [p for p in grading_view(true_at(entries, last)) if p["grade"] == "code"]
    for arm in ("note", "log"):
        run = sorted((CONTROL / config / persona / arm).glob("run_*"), key=lambda p: int(p.name[4:]))[-1]
        snap = run / "B" / "snapshots" / f"session_{last:03d}.json"
        store = MemoryStore.load(snap)
        live = store.active()
        for p in probes:
            parent = parent_entry(p, by_row)
            hits = [m for m in live if match_with_parent(p, m.text, parent, OVERLAP)]
            ranked = sorted(live, key=lambda m: -likeness(p, parent, m.text)[0])
            shown = ([hits[0]] if hits else []) + [m for m in ranked if not hits or m.id != hits[0].id]
            shown = shown[:SHOWN]
            best_share = max((likeness(p, parent, m.text)[1] for m in live), default=0.0)
            nums = numbers_in(p["detail"])
            if hits:
                literal = norm(p["detail"]) in norm(hits[0].text)
                near = not literal or (group(p) == "long item" and likeness(p, None, hits[0].text)[1] < 0.9)
            else:
                near = best_share >= 0.5 or bool(nums and any(nums & numbers_in(m.text) for m in live))
            items.append({"persona": persona, "arm": arm, "kind": p["kind"], "state": p["state"], "field": p.get("field", ""),
                          "detail": p["detail"], "attached_to": parent["detail"] if parent else "",
                          "said_on": p.get("from_date", ""), "group": group(p), "code_found": bool(hits),
                          "code_memory": hits[0].id if hits else "", "near": near, "best_share": round(best_share, 3),
                          "memories": [{"id": m.id, "date": m.date, "text": m.text} for m in shown],
                          "probe": p.get("row_id") or f"{p['record']}|{p['field']}|{p['detail']}", "snapshot": f"{config}/{persona}/{arm}/{snap.name}"})

print(f"Control run {config}: {len(items)} probe-in-store items (code-graded probes at the end, both arms)")
cells = defaultdict(list)
for it in items:
    if it["state"] == "true":
        cells[(it["arm"], it["code_found"], it["group"])].append(it)
old = defaultdict(list)
for it in items:
    if it["state"] == "false":
        old[(it["arm"], it["code_found"])].append(it)
for k in sorted(cells):
    print(f"  {k[0]:5} {'found' if k[1] else 'not found':10} {k[2]:26} {len(cells[k]):4} (near {sum(x['near'] for x in cells[k])})")
for k in sorted(old):
    print(f"  {k[0]:5} {'found' if k[1] else 'not found':10} {'old value (any kind)':26} {len(old[k]):4}")

rng = random.Random(SEED)
chosen = []
for k in sorted(cells):
    pool = sorted(cells[k], key=lambda x: (x["persona"], x["probe"]))
    near = [x for x in pool if x["near"]]
    rng.shuffle(near)
    pick = near[: PER_CELL // 2 + PER_CELL % 2]
    rest = [x for x in pool if x not in pick]
    rng.shuffle(rest)
    pick += rest[: PER_CELL - len(pick)]
    chosen += pick
for arm in ("note", "log"):
    found_pool = sorted(old[(arm, True)], key=lambda x: (x["persona"], x["probe"]))
    miss_pool = sorted(old[(arm, False)], key=lambda x: (x["persona"], x["probe"]))
    rng.shuffle(found_pool)
    rng.shuffle(miss_pool)
    pick = found_pool[: OLD_PER_ARM // 2]
    pick += miss_pool[: OLD_PER_ARM - len(pick)]
    pick += found_pool[OLD_PER_ARM // 2: OLD_PER_ARM // 2 + OLD_PER_ARM - len(pick)]
    chosen += pick
rng.shuffle(chosen)
for i, it in enumerate(chosen, start=1):
    it["item"] = f"G{i:02d}"
    rng.shuffle(it["memories"])
out = labels_dir / f"{datetime.date.today().isoformat()}-grader-sample-v2.jsonl"
out.write_text("".join(json.dumps(it) + "\n" for it in chosen), encoding="utf-8")
print(f"\n{len(chosen)} items -> {out.relative_to(ROOT)}")
for key in ("arm", "code_found", "group", "kind", "state", "near"):
    print(f"  {key:10} " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(str(it[key]) for it in chosen).items())))
