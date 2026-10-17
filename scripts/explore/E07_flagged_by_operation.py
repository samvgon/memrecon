"""E07: how many flagged turns belong to each kind of session (add, update, delete)?

Question: the README says `share_memory` also flags turns for "forgetting sessions". How many of the
1,613 flagged turns (E04) are facts to remember, and how many are instructions to forget or edit?
Method: count flagged turns by (session_type, operation) and print a few examples from delete and
update sessions. Supersedes the unsplit count in E04 for the purpose of counting probe-able facts.
"""
import collections

from common import flagged_turns, iter_sessions

counts = collections.Counter()
sessions = collections.Counter()
examples = collections.defaultdict(list)
for persona, path, s in iter_sessions():
    key = (s.get("session_type"), s.get("operation"))
    fl = flagged_turns(s)
    counts[key] += len(fl)
    sessions[key] += 1
    for t in fl:
        if len(examples[key]) < 3:
            examples[key].append(f"{persona}/{path.name}: {t['message'][:140]}")

total = sum(counts.values())
print(f"{total} flagged turns in total")
print("(session_type, operation): flagged turns | sessions | flagged per session")
for key, c in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {str(key):30} {c:5d} | {sessions[key]:4d} | {c / sessions[key]:.2f}")
print()
print("Examples (up to 3 per group):")
for key in sorted(examples, key=str):
    for e in examples[key]:
        print(f"  {key}  {e}")
