"""Show how each judge scored chosen hand-label items, with both judge prompts, from the call cache only.

Usage, from the repo root: uv run python scripts/explore/show_judge_items.py J64 J69
For each item: the question, the sub-question, Sam's label and note, Memora's expected answer, then for
Qwen and Gemma the `judge_v1` verdict and reason (E39/E44) and the Memora-prompt reply (E45). Makes no
model calls: anything not already in the cache is reported as not cached. Not an evidence script.
"""
import ast
import json
import sys

from common import ROOT
from memrecon.judge import Judge
from memrecon.llm import cache_key, client as model_client

LABELS = ROOT / "notes" / "labels"
ROLE = {"qwen": "agent", "gemma": "judge"}


class NotCached(Exception):
    pass


class CacheOnly:
    def __init__(self, inner):
        self.inner = inner

    def chat(self, messages, *, use_cache=True, tag=None, meta=None, **settings):
        if not self.inner.path_for(cache_key(self.inner.request_body(messages, **settings))).exists():
            raise NotCached()
        return self.inner.chat(messages, use_cache=True, tag=tag, meta=meta, **settings)


# Memora's judge prompt exactly as E45 sent it (read from E45's source, so the requests match its cache).
src = ast.parse((ROOT / "scripts" / "explore" / "E45_memora_judge_prompt.py").read_text())
consts = {n.targets[0].id: ast.literal_eval(n.value) for n in src.body
          if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("MEMORA_SYSTEM", "MEMORA_USER")}

items = [a.upper() for a in sys.argv[1:]] or ["J64", "J69"]
sample = {json.loads(x)["item"]: json.loads(x) for x in (LABELS / "2026-11-08-judge-sample.jsonl").read_text().splitlines() if x.strip()}
given = {json.loads(x)["item"]: json.loads(x) for x in (LABELS / "2026-11-08-judge-labels.jsonl").read_text().splitlines() if x.strip()}
resolved = {json.loads(x)["item"]: json.loads(x) for x in (LABELS / "2026-11-08-judge-labels-resolved.jsonl").read_text().splitlines() if x.strip()}
clients = {j: CacheOnly(model_client(ROLE[j])) for j in ROLE}

for k in items:
    s = sample[k]
    print(f"=== {k}: {s['persona']} {s['arm']} ({s.get('set', resolved[k].get('set'))} set)")
    print(f"Question: {s['question']}")
    print(f"Sub-question: {s['sub_question']}")
    print(f"Memora's expected answer: {s.get('expected', '?')}")
    print(f"Sam's label: {given[k]['label']} -> {resolved[k]['label']} ({resolved[k]['source']}); note: {given[k].get('note', '').strip()}")
    print(f"Answer: {' '.join(s['answer'].split())}")
    for j in ROLE:
        meta = {"item": k, "sub_id": s["sub_id"], "persona": s["persona"], "arm": s["arm"]}
        try:
            r = Judge(clients[j], tag="show").judge(s["answer"], s["sub_question"], meta)
            print(f"  {j:5} judge_v1: {r['answer']}  | {r['reason']}")
        except NotCached:
            print(f"  {j:5} judge_v1: not cached")
        user = consts["MEMORA_USER"].replace("{generated_answer}", s["answer"]).replace("{eval_question}", s["sub_question"])
        try:
            res = clients[j].chat([{"role": "system", "content": consts["MEMORA_SYSTEM"]}, {"role": "user", "content": user}], max_tokens=1024)
            print(f"  {j:5} Memora's prompt: {' '.join(res.content.split())}")
        except NotCached:
            print(f"  {j:5} Memora's prompt: not cached")
    print()
