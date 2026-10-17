"""E06: how much of what the log records for adds and updates is actually spoken?

Question: can we trust `share_memory: True` turns as "the facts told to the agent", and how much of
`operation_details` never appears in the dialogue? (Tests assumptions A2 and A3.)
Method: for every add and update session, collect the leaf values of `operation_details`
(strings of 6+ characters and numbers of 100 or more; keys are ignored) and check whether each
appears verbatim (case-insensitive, commas removed from numbers) in (a) the whole dialogue and
(b) the flagged turns only. Strict matching misses paraphrases, so the fractions are lower
bounds on how much is spoken; they are not a measure of the truth.
"""
import collections
import re

from common import flagged_turns, iter_sessions


def leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from leaves(v)
    elif isinstance(x, bool) or x is None:
        return
    else:
        yield x


IDENT = re.compile(r"^[a-z0-9_]+$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def norm(t):
    t = t.lower().replace("’", "'")
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    return re.sub(r"\s+", " ", t)


tot = collections.defaultdict(lambda: [0, 0, 0, 0])  # sessions, values, found anywhere, found in flagged
missed = collections.defaultdict(list)
for persona, path, s in iter_sessions():
    op = s.get("operation")
    if op not in ("add", "update"):
        continue
    full = norm(" ".join(t["message"] for t in s["conversation"]))
    flag = norm(" ".join(t["message"] for t in flagged_turns(s)))
    key = (s["session_type"], op)
    tot[key][0] += 1
    for v in leaves(s.get("operation_details", {})):
        if isinstance(v, str):
            if len(v) < 6 or IDENT.match(v) or DATE.match(v):
                continue
            needle = norm(v)
        else:
            if abs(v) < 100:
                continue
            needle = str(int(v)) if float(v).is_integer() else str(v)
        tot[key][1] += 1
        if needle in full:
            tot[key][2] += 1
        elif len(missed[key]) < 3:
            missed[key].append(f"{persona}/{path.name}: {str(v)[:90]}")
        if needle in flag:
            tot[key][3] += 1

print("(session_type, operation): sessions | log values checked | found in dialogue | found in flagged turns")
for key, (ns, nv, nf, nfl) in sorted(tot.items()):
    pct = lambda a: f"{100 * a / nv:.0f}%" if nv else "n/a"
    print(f"  {key}: {ns} | {nv} | {nf} ({pct(nf)}) | {nfl} ({pct(nfl)})")
print()
print("Examples of log values NOT found verbatim in the dialogue (up to 3 per group):")
for key, ms in missed.items():
    for m in ms:
        print(f"  {key}  {m}")
