"""E04: how many turns are flagged `share_memory: True`, and how long are they?

Question: is there enough material for ~500 probes, and are the facts short or long?
Method: count flagged turns overall, per persona and per session type; length statistics.
"""
import collections
import statistics

from common import flagged_turns, iter_sessions

lengths = []
per_session = []
per_persona = collections.Counter()
per_type = collections.Counter()
sessions_with_flag = collections.Counter()
sessions_total = collections.Counter()
for persona, path, s in iter_sessions():
    fl = flagged_turns(s)
    per_session.append(len(fl))
    per_persona[persona] += len(fl)
    per_type[s.get("session_type")] += len(fl)
    sessions_total[s.get("session_type")] += 1
    if fl:
        sessions_with_flag[s.get("session_type")] += 1
    lengths += [len(t["message"]) for t in fl]

print(f"flagged turns: {len(lengths)}")
print(f"characters per flagged turn: median {statistics.median(lengths)}, mean {statistics.mean(lengths):.0f}, max {max(lengths)}")
print(f"flagged turns per session: max {max(per_session)}, mean {statistics.mean(per_session):.2f}")
print()
print("by session_type (flagged turns | sessions with any flag / sessions):")
for st in sorted(sessions_total):
    print(f"  {st!r:14} {per_type[st]:5d} | {sessions_with_flag[st]} / {sessions_total[st]}")
print()
print("by persona:")
for p, c in sorted(per_persona.items()):
    print(f"  {p:24} {c}")
