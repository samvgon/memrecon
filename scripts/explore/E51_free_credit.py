"""E51: how much FAMA does an answer earn by saying nothing about the user? (follow-up to E49)

Question: E49 found that answers written with no memory score about 30 FAMA on recommending, whether or not
they know the persona's profile. Memora's recommending questions have "dislike" presence sub-questions
("Does the response avoid recommending X (which the user dislikes)?"), which an answer that names nothing
passes, and forgetting sub-questions ("Does the response mention X?"), which it also passes. So how much of
the score is available to an answer that names no items at all?
Method (no model calls): for every weekly question, count its presence sub-questions by kind (the id's
`_eval_memory_presence_<kind>_` part: like, dislike, and the remembering/reasoning kinds) and its
forgetting sub-questions. A "silent" answer is one that recommends and mentions nothing: it is taken to pass
every dislike sub-question (it recommends nothing) and every forgetting sub-question (it mentions nothing),
and fail every other presence sub-question. Its per-question FAMA is then (dislikes / presence) with FAA = 1;
reported as Memora's table cell (mean x100) by category, on the development personas (as E49) and on all 10
weekly personas (descriptive counts only, as E01, E08 and E48; Sam, Nov 13), with E49's measured no-memory
scores for comparison. The silent answer is a construction from the rubric, not a judged answer: a judge
could still fail a dislike sub-question for an answer that names nothing (E49 measures what judges did).
Reads the weekly question files only; seconds.
"""
import re
from collections import Counter

from common import WEEKLY, load, personas
from memrecon.memora import dev_personas

CATS = ("remembering", "reasoning", "recommending")
E49 = {"generic": (29.8, 28.4), "persona": (32.0, 36.2)}  # recommending FAMA under judge_v1, Memora's judge (E49)


def kind(sub):
    m = re.search(r"_eval_memory_presence(?:_([a-z]+))?(?:_\d+)?$", sub.get("evaluation_question_id", ""))
    return (m.group(1) or "plain") if m else "?"


rows = []  # (persona, category, presence kinds Counter, n forgetting)
for p in personas():
    for cat, qs in load(WEEKLY / p / f"evaluation_questions_{p}.json")["questions"].items():
        for q in qs:
            subs = (q.get("evaluation") or {}).get("evaluation_questions") or []
            pres = Counter(kind(s) for s in subs if s.get("evaluation_type") == "memory_presence")
            nf = sum(s.get("evaluation_type") != "memory_presence" for s in subs)
            rows.append((p, cat, pres, nf))

unknown = sum(c["?"] for _, _, c, _ in rows)
print(f"{len(rows)} questions over {len(personas())} weekly personas; presence sub-questions of unrecognised kind: {unknown}\n")
for label, keep in (("development personas (as E49)", set(dev_personas())), ("all 10 weekly personas", set(personas()))):
    print(f"== {label} ==")
    print(f"  {'category':14}{'n':>4}  presence by kind{'':30}forgetting   silent-answer FAMA")
    for cat in CATS:
        sub = [r for r in rows if r[0] in keep and r[1] == cat]
        if not sub:
            continue
        kinds = sum((r[2] for r in sub), Counter())
        nf = sum(r[3] for r in sub)
        per_q = [(r[2]["dislike"] / sum(r[2].values())) if sum(r[2].values()) else 0.0 for r in sub]
        free = 100 * sum(per_q) / len(per_q)
        ks = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
        print(f"  {cat:14}{len(sub):4}  {ks:46}{nf:6}       {free:5.1f}")
        if cat == "recommending":
            nd, npres = kinds["dislike"], sum(kinds.values())
            print(f"  {'':18}dislike sub-questions: {nd} of {npres} presence sub-questions ({100 * nd / npres:.0f}%)")
    print()
print("E49, measured on the development personas' recommending questions (FAMA; judge_v1, Memora's judge):")
for k, (a, b) in E49.items():
    print(f"  {k:8} (no memory)  {a:5.1f}  {b:5.1f}")
