"""E02: how many remembered and forgotten items do Memora's evaluation questions list?

Question: are Memora's own questions enough to supply ~500 probed facts?
Method: count dicts that have both `value` and `session_id` inside `memory_evidence`
and `forgetting_evidence`. Because that counter may miss differently shaped evidence,
also list the evidence key names and show examples of questions where it found nothing.
"""
import collections
import json

from common import iter_question_files


def pairs(x):
    if isinstance(x, dict):
        if "value" in x and "session_id" in x:
            yield x
        for v in x.values():
            yield from pairs(v)
    elif isinstance(x, list):
        for v in x:
            yield from pairs(v)


nq = 0
mem = forget = 0
mem_keys = collections.Counter()
forget_keys = collections.Counter()
empty_examples = []
by_type = collections.Counter()
for persona, path, d in iter_question_files():
    for qtype, qs in d["questions"].items():
        for q in qs:
            nq += 1
            by_type[qtype] += 1
            me, fe = q.get("memory_evidence"), q.get("forgetting_evidence")
            m, f = len(list(pairs(me))), len(list(pairs(fe)))
            mem += m
            forget += f
            if isinstance(me, dict):
                mem_keys.update(me.keys())
            if isinstance(fe, dict):
                forget_keys.update(fe.keys())
            if m == 0 and len(empty_examples) < 4:
                empty_examples.append((persona, qtype, q.get("question_id"), json.dumps(me, ensure_ascii=False)[:300]))

print(f"{nq} questions {dict(by_type)}")
print(f"{mem} memory items and {forget} forgotten items matched as {{value, session_id}} dicts")
print()
print("memory_evidence keys:", dict(mem_keys.most_common(15)))
print("forgetting_evidence keys:", dict(forget_keys.most_common(15)))
print()
print("Questions where no memory item was matched (up to 4 examples):")
for e in empty_examples:
    print(" -", e)
