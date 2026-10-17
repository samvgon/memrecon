"""E05: how often do flagged turns contain hedging language?

Question: are natural hedges frequent enough to use as a probe type?
Method: two keyword searches over flagged turns. v1 is the loose list first run on Oct 15
(it includes "about", which is mostly a plain preposition, so it overcounts). v2 drops
"about" and counts each word separately. Keyword search can miss hedges phrased in other words.
"""
import collections
import re

from common import flagged_turns, iter_sessions

turns = [t["message"] for _, _, s in iter_sessions() for t in flagged_turns(s)]

v1 = re.compile(r"\b(approximately|about|around|roughly|maybe|might|could|probably|tentative|tentatively|at least|up to|depends|depending|if possible|ideally|hope|plan to|expected|estimated|around)\b", re.I)
hits = [m for m in turns if v1.search(m)]
print(f"v1 (loose): {len(hits)} of {len(turns)} flagged turns match ({100 * len(hits) / len(turns):.1f}%)")
for m in hits[:8]:
    print("  -", m[:200])
print()

words = ["approximately", "roughly", "around", "maybe", "might", "probably", "tentative", "tentatively", "at least",
         "up to", "depends", "depending", "if possible", "ideally", "estimated", "about [0-9$€£]"]
pats = {w: re.compile(r"\b" + w + r"\b", re.I) for w in words}
counts = collections.Counter()
example = {}
for m in turns:
    for w, p in pats.items():
        if p.search(m):
            counts[w] += 1
            example.setdefault(w, m[:160])
print(f"v2 (tight, per word) over {len(turns)} flagged turns:")
for w, c in counts.most_common():
    print(f"  {c:3d}  {w:18} | {example[w]}")
