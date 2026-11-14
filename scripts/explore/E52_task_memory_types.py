"""E52: are Memora's three tasks separate from its memory types? (for the paper video)

Question: Memora reports results by task (remembering, reasoning, recommending; Tables 3-4, Figure 3) and
draws conclusions such as "agents are strongest on remembering" and "reasoning is the weakest task". Its
memory types are preference, activity (personal and work) and goal memory. If each task draws on a single
memory type, task and memory type cannot be told apart. The paper also says each question is paired with
memory-presence and forgetting-absence criteria (§3.5); E51 found reasoning questions have none of the
latter.
Method (no model calls): for every weekly question, the memory it draws on from its `question_id` prefix
(`pref_` preference; `activity_<kind>_` activity; `content_` work content, which the released sessions file
as activity, see the paper's "work activity"; `goal_` goal), by task; the distinct question texts per task
with digits masked (how many templates each task uses); and per task, how many questions have forgetting
sub-questions. On all 10 weekly personas (descriptive counts only, as E01, E08, E48 and E51; Sam, Nov 13)
and on the 3 development personas. Reads the weekly question files only; seconds.
"""
import re
from collections import Counter, defaultdict

from common import WEEKLY, load, personas
from memrecon.memora import dev_personas

CATS = ("remembering", "reasoning", "recommending")


def source(qid):
    if qid.startswith("pref_"):
        return "preference"
    if qid.startswith("goal_"):
        return "goal"
    if qid.startswith("content_"):
        return "work content"
    m = re.match(r"activity_([a-z]+)", qid)
    return f"activity: {m.group(1)}" if m else f"other ({qid.split('_')[0]})"


rows = []  # (persona, task, source, masked question text, has forgetting sub-questions)
for p in personas():
    for task, qs in load(WEEKLY / p / f"evaluation_questions_{p}.json")["questions"].items():
        for q in qs:
            subs = (q.get("evaluation") or {}).get("evaluation_questions") or []
            nf = sum(s.get("evaluation_type") != "memory_presence" for s in subs)
            text = re.sub(r"\d", "#", q.get("question", "")).strip()
            rows.append((p, task, source(q.get("question_id", "")), text, nf > 0))

for label, keep in (("all 10 weekly personas", set(personas())), ("development personas", set(dev_personas()))):
    sub = [r for r in rows if r[0] in keep]
    print(f"== {label}: {len(sub)} questions ==")
    print("  Memory each task draws on (from question_id):")
    for t in CATS:
        c = Counter(r[2] for r in sub if r[1] == t)
        print(f"    {t:13} n={sum(c.values()):3}  " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    print("  Which tasks each memory feeds:")
    by = defaultdict(Counter)
    for r in sub:
        by[r[2].split(":")[0]][r[1]] += 1
    for k in sorted(by):
        print(f"    {k:13} " + ", ".join(f"{t} {n}" for t, n in by[k].most_common()))
    print("  Templates (distinct question texts, digits masked) and questions with forgetting sub-questions:")
    for t in CATS:
        tt = [r for r in sub if r[1] == t]
        texts = Counter(r[3] for r in tt)
        nf = sum(r[4] for r in tt)
        print(f"    {t:13} {len(texts):3} distinct texts for {len(tt):3} questions; with forgetting sub-questions: {nf} of {len(tt)}")
        if t == "reasoning":
            for q, n in texts.most_common():
                print(f"        {n:3}  {q}")
    print()
