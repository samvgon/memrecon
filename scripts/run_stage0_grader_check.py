"""Stage 0 code-grader check (D21): the code grader's found / not found against Sam's labels.

  uv run python scripts/run_stage0_grader_check.py | tee notes/evidence/E40.txt

Items: notes/labels/2026-11-08-grader-sample-v2.jsonl (scripts/make_grader_sample.py); labels: notes/labels/2026-11-08-grader-labels-resolved.jsonl
(one line per item: `item`, `label` yes, no or dropped: does one of the memories shown state the detail; `source`). Pass: raw
agreement >= 90% and Cohen's kappa >= 0.7 (the code is deterministic, so there is no flip rate). No model calls.
"""
import json
from collections import Counter

from memrecon.judge import cohen_kappa
from memrecon.memora import ROOT

L = ROOT / "notes" / "labels"
sample_f, labels_f = L / "2026-11-08-grader-sample-v2.jsonl", L / "2026-11-08-grader-labels-resolved.jsonl"
sample = {json.loads(x)["item"]: json.loads(x) for x in sample_f.read_text().splitlines() if x.strip()}
labels = {json.loads(x)["item"]: json.loads(x) for x in labels_f.read_text().splitlines() if x.strip()}
rows = [{**sample[k], "label": labels[k]["label"], "note": labels[k].get("note", ""), "code": "yes" if sample[k]["code_found"] else "no"}
        for k in sorted(sample) if labels[k]["label"] in ("yes", "no")]
dropped = [k for k in sorted(sample) if labels[k]["label"] not in ("yes", "no")]
n = len(rows)
agree = sum(r["code"] == r["label"] for r in rows)
kappa = cohen_kappa([r["label"] for r in rows], [r["code"] for r in rows])
print(f"# Stage 0 code-grader check ({sample_f.name}, {labels_f.name})")
print(f"{n} items ({len(dropped)} dropped: {', '.join(dropped) or 'none'}). Raw agreement with Sam: {agree} of {n} ({100 * agree / n:.1f}%)   [pass: >= 90%]")
print(f"Cohen's kappa: {kappa:.3f}   [pass: >= 0.7]")
print(f"\n**Code-grader check: {'PASS' if agree / n >= 0.9 and kappa >= 0.7 else 'FAIL'}**\n")
print("Agreement by group")
for key in ("arm", "group", "kind", "state", "near", "code"):
    c = Counter((str(r[key]), r["code"] == r["label"]) for r in rows)
    print(f"  {key:6} " + "; ".join(f"{v} {c[(v, True)]} of {c[(v, True)] + c[(v, False)]}" for v in sorted({str(r[key]) for r in rows})))
print("  (code = the code's verdict; disagreements where the code says no are misses, where it says yes false finds)")
print("\n== Disagreements ==")
for r in rows:
    if r["code"] != r["label"]:
        print(f"--- {r['item']} ({r['arm']}, {r['kind']}, {r['group']}, state {r['state']}): Sam {r['label']}, code {r['code']}")
        print(f"  detail: {r['detail']}" + (f"   [attached to: {r['attached_to']}]" if r["attached_to"] else ""))
        if r["note"]:
            print(f"  Sam's note: {r['note']}")
        if labels[r["item"]]["source"] != "Sam's label":
            print(f"  label source: {labels[r['item']]['source']}")
        for m in r["memories"]:
            mark = " <- code's match" if m["id"] == r["code_memory"] else ""
            print(f"  ({m['date']}) {m['text'][:300]}{mark}")
