"""E01: what sessions exist, and what do `session_type` and `operation` contain?

Question: is the add/update/delete log really in the data, and how big is each part?
Method: count (session_type, operation) over all weekly personas and print one
example session (without its conversation) for each combination.
"""
import collections
import json

from common import iter_sessions, personas

counts = collections.Counter()
example = {}
n = 0
for persona, path, s in iter_sessions():
    n += 1
    key = (s.get("session_type"), s.get("operation"))
    counts[key] += 1
    example.setdefault(key, (persona, path.name, {k: v for k, v in s.items() if k != "conversation"}))

print(f"{len(personas())} personas, {n} sessions")
for (st, op), c in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"{c:5d}  session_type={st!r:14} operation={op!r}")
print()
for key, (persona, name, meta) in example.items():
    print(f"--- example {key}: {persona}/{name}")
    print(json.dumps(meta, indent=1, ensure_ascii=False)[:1500])
