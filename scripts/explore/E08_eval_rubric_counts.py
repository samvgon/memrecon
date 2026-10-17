"""E08: how many scored yes/no sub-questions does Memora's rubric contain?

Question: the README says the `evaluation` block is the actual scoring rubric (the evidence blocks are
for auditing). How big is the rubric, and how is it split between "should mention" (`memory_presence`)
and "should not mention" (`forgetting_absence`)?
Method: read `evaluation.evaluation_questions` for every question; count by `evaluation_type` and task
type; check that the stated totals match the counts. Supersedes E02 for sizing the benchmark.
"""
import collections

from common import iter_question_files

by_type = collections.Counter()
by_task = collections.defaultdict(collections.Counter)
by_persona = collections.Counter()
mismatch = []
nq = 0
examples = []
for persona, path, d in iter_question_files():
    for task, qs in d["questions"].items():
        for q in qs:
            nq += 1
            ev = q.get("evaluation", {})
            subs = ev.get("evaluation_questions", [])
            c = collections.Counter(x.get("evaluation_type") for x in subs)
            for k, v in c.items():
                by_type[k] += v
                by_task[task][k] += v
            by_persona[persona] += len(subs)
            if (ev.get("total_evaluation_questions") != len(subs)
                    or ev.get("memory_presence_questions") != c.get("memory_presence", 0)
                    or ev.get("forgetting_absence_questions") != c.get("forgetting_absence", 0)):
                mismatch.append((persona, q.get("question_id")))
            if len(examples) < 4 and task == "remembering":
                examples.append(subs[0].get("evaluation_question", "")[:160] if subs else "")

print(f"{nq} questions; {sum(by_type.values())} scored sub-questions: {dict(by_type)}")
print()
print("by task type:")
for task, c in by_task.items():
    print(f"  {task:13} {dict(c)}")
print()
print("by persona (sub-questions):")
for p, n in sorted(by_persona.items()):
    print(f"  {p:24} {n}")
print()
print(f"questions whose stated totals do not match the counts: {len(mismatch)}")
for m in mismatch[:5]:
    print("  ", m)
print()
print("Example sub-questions:")
for e in examples:
    print("  -", e)
