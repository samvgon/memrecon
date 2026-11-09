"""Stage 0 judge check (D21, D61): the judge against Sam's hand labels.

  uv run python scripts/run_stage0_judge_check.py --set tune               # the 20 tune items: adjust the prompt on these
  uv run python scripts/run_stage0_judge_check.py --set check | tee notes/evidence/E39.txt   # once, with the frozen prompt

Items: notes/labels/2026-11-08-judge-sample.jsonl (the answers and sub-questions) and Sam's labels,
notes/labels/2026-11-08-judge-labels-resolved.jsonl (her "Can't tell" labels resolved by her rules, each with
its source). Every item is judged twice: as usual, and again with request seed 1, so the second call is made
afresh rather than read from the cache; the first is the judge's answer, the pair gives the flip rate.
Pass (check set only): raw agreement >= 90%, Cohen's kappa >= 0.7, flip rate <= 5%.
The check set is judged once, with a prompt that is not changed afterwards; the tune set is for adjusting it.
Results: results/stage0/judge/<prompt>/<set>.jsonl.
"""
import argparse
import json
import subprocess
import time
from collections import Counter

from memrecon.judge import JUDGE_PROMPT, Judge, cohen_kappa
from memrecon.llm import client as model_client, endpoint
from memrecon.memora import ROOT
from memrecon.monitor import wait_for_memory

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["tune", "check"])
ap.add_argument("--prompt", default=JUDGE_PROMPT)
args = ap.parse_args()

LABELS = ROOT / "notes" / "labels"
sample = {json.loads(x)["item"]: json.loads(x) for x in (LABELS / "2026-11-08-judge-sample.jsonl").read_text().splitlines() if x.strip()}
labels = {json.loads(x)["item"]: json.loads(x) for x in (LABELS / "2026-11-08-judge-labels-resolved.jsonl").read_text().splitlines() if x.strip()}
items = sorted(k for k, s in sample.items() if s["set"] == args.set)
try:
    code = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "scripts"], capture_output=True, text=True).stdout.strip()
except Exception:
    code, dirty = "unknown", ""
print(f"# Stage 0 judge check, {args.set} set ({len(items)} items)")
print(f"judge: {endpoint('judge').model}; prompt: {args.prompt}; memrecon: {code}{' (src or scripts have uncommitted changes)' if dirty else ''}; "
      f"run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

client = model_client("judge")
judges = [Judge(client, prompt=args.prompt, tag="stage0-judge"), Judge(client, prompt=args.prompt, tag="stage0-judge", call_settings={"seed": 1})]
rows = []
t0 = time.monotonic()
for i, k in enumerate(items):
    if i % 10 == 0:
        wait_for_memory(say=print)
    s = sample[k]
    meta = {"item": k, "sub_id": s["sub_id"], "persona": s["persona"], "arm": s["arm"]}
    first, second = (j.judge(s["answer"], s["sub_question"], meta) for j in judges)
    rows.append({"item": k, "label": labels[k]["label"], "label_source": labels[k]["source"], "judge": first["answer"],
                 "judge_again": second["answer"], "reason": first["reason"], "reason_again": second["reason"],
                 "error": first["error"] or second["error"], "type": s["type"], "arm": s["arm"], "category": s["category"],
                 "expected": s["expected"], "sub_question": s["sub_question"]})
out = ROOT / "results" / "stage0" / "judge" / args.prompt / f"{args.set}.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

ok = [r for r in rows if r["judge"] is not None]
n = len(rows)
agree = sum(r["judge"] == r["label"] for r in ok)
kappa = cohen_kappa([r["label"] for r in ok], [r["judge"] for r in ok])
pairs = [r for r in rows if r["judge"] is not None and r["judge_again"] is not None]
flips = sum(r["judge"] != r["judge_again"] for r in pairs)
print(f"{n} items judged twice in {time.monotonic() - t0:.0f} s; failed answers: {n - len(ok)} (first), "
      f"{sum(r['judge_again'] is None for r in rows)} (again)")
print(f"Raw agreement with Sam: {agree} of {n} ({100 * agree / n:.1f}%; a failed answer counts as a disagreement)   [pass: >= 90%]")
print(f"Cohen's kappa: {kappa:.3f}   [pass: >= 0.7]")
print(f"Flip rate (first against again): {flips} of {len(pairs)} ({100 * flips / max(1, len(pairs)):.1f}%)   [pass: <= 5%]")
if args.set == "check":
    passed = agree / n >= 0.90 and kappa >= 0.7 and flips / max(1, len(pairs)) <= 0.05 and len(pairs) == n
    print(f"\n**Judge check: {'PASS' if passed else 'FAIL'}**")
print()
print("Agreement by group (first answer)")
for key in ("type", "arm", "category", "label"):
    c = Counter((r[key], r["judge"] == r["label"]) for r in rows)
    print(f"  {key:9} " + "; ".join(f"{v} {c[(v, True)]} of {c[(v, True)] + c[(v, False)]}" for v in sorted({r[key] for r in rows})))
from_rule = [r for r in rows if not r["label_source"].startswith("Sam's label")]
print(f"  labels from Sam's rules or answers: {sum(r['judge'] == r['label'] for r in from_rule)} of {len(from_rule)} agree")
lab_vs_memora = sum(r["label"] == r["expected"] for r in rows)
print(f"  (for reference: Sam's label equals Memora's expected answer on {lab_vs_memora} of {n}; the judge's on "
      f"{sum(r['judge'] == r['expected'] for r in ok)})")
print()
print("== Disagreements with Sam ==")
for r in rows:
    if r["judge"] != r["label"]:
        s = sample[r["item"]]
        print(f"--- {r['item']} ({r['type']}, {r['arm']}, {r['category']}): Sam {r['label']} ({r['label_source']}), judge {r['judge']} (again: {r['judge_again']})")
        print(f"  Q: {r['sub_question']}")
        print(f"  Judge's reason: {r['reason'] or r['error']}")
        if labels[r["item"]]["note"]:
            print(f"  Sam's note: {labels[r['item']]['note']}")
        a = " ".join(s["answer"].split())
        print(f"  Response: {a[:700]}{'…' if len(a) > 700 else ''}")
print("\n== Flips ==")
for r in pairs:
    if r["judge"] != r["judge_again"]:
        print(f"--- {r['item']}: {r['judge']} then {r['judge_again']} | Q: {r['sub_question'][:150]}")
        print(f"  first: {r['reason'][:300]}\n  again: {r['reason_again'][:300]}")
