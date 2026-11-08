"""Make the frozen sample of judge items for Sam to label (D21).

An item is one of Memora's yes/no sub-questions about one agent answer (from scripts/run_stage0_answers.py):
"Does the response mention the task: Buy groceries?" asked of what the agent answered to "What tasks remain
on my todo list this week?". The judge will answer the same yes/no question about the same answer, so Sam's
label is her own answer to the sub-question, about the answer as written, before seeing any judge output.

The sample is stratified (CLAUDE.md): by arm (note, log), by sub-question type ("should mention",
memory_presence; "should not mention", forgetting_absence) and by how much of the sub-question's own words
appear in the answer (none, some, most), since items where the answer half-says it are the hard ones; each
of the 12 cells gets an equal share where it has enough items, and the rest is filled at random. Questions
are spread across all three categories (remembering, reasoning, recommending) within a cell. Then 20 items,
spread across the cells, are marked `tune` (for writing the judge prompt) and the other 50 `check` (held
back for the pass/fail check). The order shown is random. Seed 21.

Usage:  uv run python scripts/make_judge_sample.py [--config <control config id>]
Writes notes/labels/<date>-judge-sample.jsonl and refuses to overwrite it: the sample is frozen once made.
"""
import argparse
import datetime
import json
import random
import sys
from collections import Counter, defaultdict

from memrecon.memora import ROOT
from memrecon.probes import content_words

SEED = 21
N, N_TUNE = 70, 20
ANSWERS = ROOT / "results" / "stage0" / "answers"
# words of the sub-question templates, not of the thing asked about
TEMPLATE = {"response", "mention", "mentions", "mentioned", "include", "includes", "included", "list", "listed", "item",
            "task", "todo", "correctly", "shown", "show", "active", "current", "remaining", "state", "states", "does",
            "answer", "reference", "references", "recommend", "recommendation", "recommends", "user", "still", "say",
            "says", "indicate", "indicates", "note", "notes", "contain", "contains", "suggest", "suggests", "refer", "refers"}


def overlap(sub, answer):
    words = content_words(sub) - TEMPLATE
    if not words or not answer:
        return "none"
    share = len(words & content_words(answer)) / len(words)
    return "none" if share < 0.34 else "most" if share >= 0.8 else "some"


ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
args = ap.parse_args()
configs = [p for p in ANSWERS.iterdir() if p.is_dir()] if ANSWERS.exists() else []
if args.config:
    configs = [ANSWERS / args.config]
if len(configs) != 1:
    sys.exit(f"Give --config: answers found for {[p.name for p in configs]}")
cfg = configs[0]
out = ROOT / "notes" / "labels" / f"{datetime.date.today().isoformat()}-judge-sample.jsonl"
if list((ROOT / "notes" / "labels").glob("*-judge-sample.jsonl")):
    sys.exit("A judge sample already exists in notes/labels/; it is frozen (D21). Not overwritten.")

items = []
for f in sorted(cfg.glob("*/*/answers.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        a = json.loads(line)
        if a["answer"] is None:
            continue
        for s in a["sub_questions"]:
            items.append({"persona": a["persona"], "arm": a["arm"], "category": a["category"], "question_id": a["question_id"],
                          "question": a["question"], "date": a["date"], "answer": a["answer"], "sub_id": s["id"],
                          "sub_question": s["text"], "type": s["type"], "expected": s["expected"],
                          "overlap": overlap(s["text"], a["answer"])})
print(f"Answers from {cfg.name}: {len(items)} sub-question items")
cells = defaultdict(list)
for it in items:
    cells[(it["arm"], it["type"], it["overlap"])].append(it)
for k in sorted(cells):
    print(f"  {k[0]:5} {k[1]:19} {k[2]:5} {len(cells[k]):4}")

rng = random.Random(SEED)
share = N // len(cells)
chosen = []
for k in sorted(cells):
    pool = cells[k][:]
    rng.shuffle(pool)
    # spread across categories: round-robin over them
    by_cat = defaultdict(list)
    for it in pool:
        by_cat[it["category"]].append(it)
    picked = []
    while len(picked) < min(share, len(pool)):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(picked) < share:
                picked.append(by_cat[cat].pop())
    chosen += picked
rest = [it for it in items if it not in chosen]
rng.shuffle(rest)
chosen += rest[: N - len(chosen)]

# tune / check: spread the 20 tune items across cells
by_cell = defaultdict(list)
for it in chosen:
    by_cell[(it["arm"], it["type"], it["overlap"])].append(it)
order = []
for k in sorted(by_cell):
    rng.shuffle(by_cell[k])
    order += by_cell[k]
for i, it in enumerate(order):
    it["set"] = "tune" if (i * N_TUNE) // N != ((i + 1) * N_TUNE) // N else "check"
rng.shuffle(chosen)
for i, it in enumerate(chosen, start=1):
    it["item"] = f"J{i:02d}"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(json.dumps(it) + "\n" for it in chosen), encoding="utf-8")
print(f"\n{len(chosen)} items ({sum(it['set'] == 'tune' for it in chosen)} tune, {sum(it['set'] == 'check' for it in chosen)} check) -> {out.relative_to(ROOT)}")
for key in ("arm", "type", "overlap", "category", "expected", "set"):
    print(f"  {key:9} " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(it[key] for it in chosen).items())))
