"""E25: can Memora's operation log give each record's latest state, to check the probe extractor against?

Question (drift measure, D-rule that probes come from the dialogue): the per-detail extractor (judge
model) will label each detail of a flagged turn as stated, new, old (replaced), removed, done or no
longer liked. Can the log's operation_details serve as an independent cross-check of those labels?
For each session type, operation and category: which fields does operation_details hold (all personas,
counts only)? On the development personas: for documents (activity sessions with no category, whose
operation_details holds content_data), what changes between a record's consecutive content_data (list
items added or removed, fields changed), and do the added, removed, new and old values appear in the
session's flagged turns (normalised substring)? For preference updates (old_item -> item) and deletes
(item), do the items appear in the flagged turns? Do to-do and calendar deletes name the item?
Method: read-only; file order is time order (A6); records identified as in E12-E24 (A10).
Limits: normalised substring matching (lower case, quotes and trailing punctuation stripped), so a
reworded mention counts as absent.
"""
import json
import re
from collections import Counter, defaultdict

from common import ROOT, flagged_turns, iter_sessions, personas

split = json.loads((ROOT / "notes" / "split.json").read_text())


def norm(x):
    return re.sub(r"\s+", " ", re.sub(r"[\"'“”‘’*]", "", str(x))).strip(" .,!?;:").lower()


def mentioned(value, text):
    v = norm(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        digits = re.sub(r"[^\d.]", "", f"{value:,}".replace(",", ""))
        return digits.rstrip("0").rstrip(".") in re.sub(r"[^\d.]", "", text.replace(",", "")) or f"{value:,}" in text
    return bool(v) and v in norm(text)


print("1. FIELDS IN operation_details, by session type / operation / category (all personas, counts only)")
fields = defaultdict(Counter)
n = Counter()
for _, _, s in iter_sessions():
    d = s.get("operation_details") or {}
    k = (s.get("session_type"), s.get("operation"), d.get("category") if isinstance(d, dict) else None)
    n[k] += 1
    for f, v in (d.items() if isinstance(d, dict) else []):
        fields[k][f if not isinstance(v, dict) else f + "{" + ",".join(sorted(v)) + "}"] += 1
for k in sorted(n, key=str):
    print(f"  {str(k):<45} n={n[k]:4d}  " + ", ".join(f"{f} {c}" for f, c in sorted(fields[k].items())))
print()

print("2. DOCUMENTS (development personas): what changes between consecutive content_data, and is it in the flagged turns?")
c = Counter()
examples = []
for persona in split["dev"]:
    last = {}
    for _, _, s in iter_sessions(persona):
        d = s.get("operation_details") or {}
        if s.get("session_type") != "activity" or d.get("category") is not None or not isinstance(d.get("content_data"), dict):
            continue
        item, cd = d.get("item"), d["content_data"]
        text = " ".join(t["message"] for t in flagged_turns(s))
        op = s.get("operation")
        if op == "add" or item not in last:
            c[(op, "first content_data")] += 1
            last[item] = cd
            continue
        prev = last[item]
        changes = []
        for f in sorted(set(prev) | set(cd)):
            a, b = prev.get(f), cd.get(f)
            if a == b:
                continue
            if isinstance(a, list) and isinstance(b, list):
                changes += [("list item added", f, x) for x in b if x not in a] + [("list item removed", f, x) for x in a if x not in b]
            else:
                changes += [("field new value", f, b), ("field old value", f, a)]
        c[(op, "sessions")] += 1
        if not changes:
            c[(op, "no change in content_data")] += 1
        for kind, f, x in changes:
            c[(op, kind, "total")] += 1
            c[(op, kind, "in flagged turns")] += mentioned(x, text)
        if len(examples) < 6 and changes:
            examples.append((persona, s["session_id"], op, changes, text[:300]))
        last[item] = cd
for k in sorted(c, key=str):
    print(f"  {str(k):<70} {c[k]}")
print("  examples:")
for persona, sid, op, changes, text in examples:
    print(f"  - {persona} session {sid} ({op}). User: {text}")
    for kind, f, x in changes:
        print(f"      {kind:<18} {f}: {str(x)[:140]}")
print()

print("3. PREFERENCES, TO-DOS, CALENDAR, GOALS (development personas): are the logged items in the flagged turns?")
c = Counter()
for persona in split["dev"]:
    for _, _, s in iter_sessions(persona):
        d = s.get("operation_details") or {}
        t, op = s.get("session_type"), s.get("operation")
        if op not in ("add", "update", "delete") or t not in ("preference", "activity", "goal"):
            continue
        text = " ".join(m["message"] for m in flagged_turns(s))
        if t == "preference":
            c[("preference", op, "item in turns")] += mentioned(d.get("item", ""), text)
            c[("preference", op, "n")] += 1
            if op == "update":
                c[("preference", op, "old_item in turns")] += mentioned(d.get("old_item", ""), text)
                c[("preference", op, "update_type " + str(d.get("update_type")))] += 1
                c[("preference", op, f"preference {d.get('old_preference')} -> {d.get('preference')}")] += 1
            else:
                c[("preference", op, f"preference {d.get('preference')}")] += 1
        elif t == "goal":
            c[("goal", op, "n")] += 1
            c[("goal", op, "amount in turns")] += mentioned(d.get("item", ""), text)
        elif isinstance(d.get("item"), dict):
            cat, it = d.get("category"), d["item"]
            key = it.get("description") or it.get("event_name") or it.get("amount") or it.get("step_count")
            c[(cat, op, "n")] += 1
            c[(cat, op, "key value in turns")] += mentioned(key, text) if key is not None else 0
for k in sorted(c, key=str):
    print(f"  {str(k):<60} {c[k]}")
