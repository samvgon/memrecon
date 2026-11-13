"""E47: E46's answers judged with Memora's own judge prompt. Does Memora's answer prompt do better under
Memora's own judge?

Question: E45 found that with Memora's judge prompt, an answer that names an outdated preference to say it
changed ("you're no longer keen on South America") fails the forgetting sub-question; our `judge_v1` rule
(D61) does not count that. E46 found that, judged with `judge_v1`, Memora's answer prompt (no dates, no
newer-wins rule) loses FAMA against ours, mostly in the log arm. Our prompt's newer-wins rule may lead the
agent to name old preferences in order to set them aside, which Memora's judge penalises. So does the
ranking of the answer prompts depend on which judge prompt scores them?
Method (development personas only): the answers of E46 (Stage 0's `answer_v1` answers and the three E46
conditions: memora, relevance, nodates; read from `results/e46/<condition>.jsonl`) judged by Gemma with
Memora's judge prompt, verbatim, read as Memora reads it (as in E45). The `judge_v1` verdicts are E46's.
Stage 0's answers judged with Memora's prompt are E45's calls (cached). For each condition: FAMA/MPA/FAA by
arm under both judge prompts, the FAMA difference against `answer_v1` under each with a bootstrap interval
over questions (seed 47), and the difference between those two differences (positive: Memora's judge
prompt favours the condition more than `judge_v1` does).
Conditions are judged one at a time in E46's order and each report is printed, and saved to
results/e47/<condition>-report.txt, as soon as it is done.
Pausing: every call is cached, so Ctrl-C and rerun to resume. `--cached-only` (or
MEMRECON_E47_CACHED_ONLY=1) makes no calls. `MEMRECON_E47_CONDITIONS` to choose and order them.
Needs E46's output and the judge (Gemma) server only.
"""
import json
import os
import random
import sys
import time

from common import ROOT
from memrecon.fama import fama_score, table_cell
from memrecon.llm import ModelCallError, OutputTruncated, cache_key, client as model_client, endpoint
from memrecon.monitor import wait_for_memory

sys.stdout.reconfigure(line_buffering=True)
CACHED_ONLY = "--cached-only" in sys.argv or os.environ.get("MEMRECON_E47_CACHED_ONLY") == "1"
CONDITIONS = [c.strip() for c in os.environ.get("MEMRECON_E47_CONDITIONS", "memora,relevance,nodates").split(",") if c.strip()]
E46 = ROOT / "results" / "e46"
OUT = ROOT / "results" / "e47"

# Memora's judge prompt and its reading of the reply: copied from E45 (memory_to_answer.py,
# _evaluate_with_single_judge, a6493188); the requests are identical, so E45's calls are reused.
MEMORA_SYSTEM = (
    'You are an expert evaluator assessing AI assistant responses. Your task is to answer a YES/NO evaluation '
    'question about a given response.\n\nYou must provide your answer in the following JSON format:\n{\n'
    '    "answer": "yes" or "no",\n    "confidence": 0.0 to 1.0,\n    "explanation": "Brief explanation of your '
    'reasoning"\n}\n\nBe objective and thorough in your evaluation.'
)
MEMORA_USER = (
    'Please evaluate the following AI response against the evaluation question.\n\nAI RESPONSE TO EVALUATE:\n'
    '{generated_answer}\n\nEVALUATION QUESTION:\n{eval_question}\n\nProvide your evaluation in JSON format with '
    'answer (yes/no), confidence (0.0-1.0), and explanation.'
)


class NotCached(Exception):
    pass


class Gate:
    def __init__(self, inner):
        self.inner, self.sent, self.skipped = inner, 0, 0

    def chat(self, messages, *, use_cache=True, tag=None, meta=None, **settings):
        path = self.inner.path_for(cache_key(self.inner.request_body(messages, **settings)))
        if not path.exists():
            if CACHED_ONLY:
                self.skipped += 1
                raise NotCached()
            self.sent += 1
        return self.inner.chat(messages, use_cache=use_cache, tag=tag, meta=meta, **settings)


def memora_read(content):
    response_text = (content or "").strip()
    try:
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            low = response_text.lower()
            if "yes" in low and "no" not in low:
                return "yes"
            if "no" in low and "yes" not in low:
                return "no"
            return "unclear"
        answer = result.get("answer", "").lower().strip()
        float(result.get("confidence", 0.0))
        explanation = result.get("explanation", "")
        if answer not in ("yes", "no"):
            low = explanation.lower()
            if "yes" in low and "no" not in low:
                answer = "yes"
            elif "no" in low and "yes" not in low:
                answer = "no"
            else:
                answer = "unclear"
        return answer
    except Exception:
        return "error"


def memora_judge(c, response, question, meta):
    messages = [{"role": "system", "content": MEMORA_SYSTEM},
                {"role": "user", "content": MEMORA_USER.replace("{generated_answer}", response).replace("{eval_question}", question)}]
    limit = 1024
    for attempt in range(1, 4):
        try:
            result = c.chat(messages, max_tokens=limit, tag="E47", meta={"step": "judge-memora", "attempt": attempt, **meta})
        except OutputTruncated:
            limit *= 2
            continue
        except ModelCallError as e:
            if "reasoning" in str(e) or "unreadable" in str(e):
                continue
            return None
        return memora_read(result.content)
    return None


def fama_parts(rows, key):
    f, mp, fa = [], [], []
    for r in rows:
        pres = [(s, v) for s, v in zip(r["subs"], r[key]) if s["type"] == "memory_presence"]
        forg = [(s, v) for s, v in zip(r["subs"], r[key]) if s["type"] != "memory_presence"]
        pc = sum(v == str(s["expected"]).lower() for s, v in pres)
        fc = sum(v == str(s["expected"]).lower() for s, v in forg)
        f.append(fama_score(pc, len(pres), fc, len(forg)))
        if pres:
            mp.append(pc / len(pres))
        if forg:
            fa.append(fc / len(forg))
    return table_cell(f), 100 * sum(mp) / max(1, len(mp)), 100 * sum(fa) / max(1, len(fa)), f


def boot(d, reps=2000, seed=47):
    rng = random.Random(seed)
    stats = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(reps))
    return 100 * sum(d) / len(d), 100 * stats[int(0.025 * reps)], 100 * stats[int(0.975 * reps) - 1]


def report(cond, rows):
    out = [f"== Condition: {cond} against answer_v1, judged by Gemma with judge_v1 (E46) and with Memora's judge prompt =="]
    out.append(f"Questions with every verdict: {len(rows)}")
    out.append(f"  {'':5} {'answers':10} {'judge prompt':13}{'FAMA':>7}{'MPA':>7}{'FAA':>7}")
    for arm in ("note", "log", "both"):
        sub = [r for r in rows if arm == "both" or r["arm"] == arm]
        if not sub:
            continue
        s = {(a, j): fama_parts(sub, f"{a}_{j}") for a in ("base", "cond") for j in ("v1", "mem")}
        for a, name in (("base", "answer_v1"), ("cond", cond)):
            for j, jn in (("v1", "judge_v1"), ("mem", "Memora's")):
                f, mp, fa, _ = s[(a, j)]
                out.append(f"  {arm:5} {name:10} {jn:13}{f:7.1f}{mp:7.1f}{fa:7.1f}")
        d1 = [c - b for b, c in zip(s[("base", "v1")][3], s[("cond", "v1")][3])]
        d2 = [c - b for b, c in zip(s[("base", "mem")][3], s[("cond", "mem")][3])]
        for lab, d in ((f"{cond} - answer_v1, judge_v1", d1), (f"{cond} - answer_v1, Memora's judge", d2),
                       ("difference of the two (Memora's judge - judge_v1)", [y - x for x, y in zip(d1, d2)])):
            m, lo, hi = boot(d)
            out.append(f"        {lab:52} {m:+6.1f} [{lo:+.1f}, {hi:+.1f}]")
    for who, key in (("answer_v1", "base"), (cond, "cond")):
        n = sum(1 for r in rows for s in r["subs"] if s["type"] != "memory_presence")
        named = sum(v == "yes" for r in rows for s, v in zip(r["subs"], r[f"{key}_mem"]) if s["type"] != "memory_presence")
        out.append(f"  {who:10}: forgetting sub-questions Memora's judge says the answer mentions: {named} of {n}")
    return "\n".join(out)


print(f"Answers: {E46.relative_to(ROOT)}/<condition>.jsonl (E46); judge Gemma = {endpoint('judge').model}")
print("Conditions, in order: " + ", ".join(CONDITIONS))
print("Mode: " + ("CACHED ONLY (no model calls; partial results)" if CACHED_ONLY else "full"))
print()
gemma = Gate(model_client("judge"))
OUT.mkdir(parents=True, exist_ok=True)
progress = open(OUT / "progress.log", "a", encoding="utf-8")
done = {}
for cond in CONDITIONS:
    f = E46 / f"{cond}.jsonl"
    if not f.exists():
        print(f"== Condition: {cond}: {f.relative_to(ROOT)} not found (run E46 first) ==\n")
        continue
    t0, sent0 = time.monotonic(), gemma.sent
    rows = []
    src = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    for n, r in enumerate(src, start=1):
        if r["cond_text"] is None or None in r["cond_v"] or None in r["base_v"]:
            continue
        if n % 5 == 1:
            wait_for_memory(say=print)
        try:
            for key, text in (("base", r["base_text"]), ("cond", r["cond_text"])):
                r[f"{key}_mem"] = [memora_judge(gemma, text, s["text"], {"persona": r["persona"], "arm": r["arm"], "sub_id": s["id"],
                                                                          "writer": "qwen" if key == "base" else f"qwen-{cond}"})
                                   for s in r["subs"]]
        except NotCached:
            continue
        if None in r["base_mem"] or None in r["cond_mem"]:
            continue
        r["base_v1"], r["cond_v1"] = r["base_v"], r["cond_v"]
        rows.append(r)
        progress.write(f"{time.strftime('%H:%M:%S')} {cond} {n}/{len(src)} sent {gemma.sent}\n")
        progress.flush()
    done[cond] = rows
    text = report(cond, rows) if rows else f"== Condition: {cond}: nothing judged yet =="
    text += (f"\n({time.monotonic() - t0:.0f} s; judge calls sent {gemma.sent - sent0}"
             + (f"; not yet cached: {gemma.skipped}" if CACHED_ONLY else "") + ")")
    (OUT / f"{cond}-report.txt").write_text(text + "\n", encoding="utf-8")
    with open(OUT / f"{cond}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(text + "\n")
progress.close()

print("== FAMA of every answer condition under both judge prompts (both arms; each condition's own questions) ==")
for cond, rows in done.items():
    if not rows:
        continue
    b1, bm = fama_parts(rows, "base_v1")[0], fama_parts(rows, "base_mem")[0]
    c1, cm = fama_parts(rows, "cond_v1")[0], fama_parts(rows, "cond_mem")[0]
    print(f"  {cond:10} n={len(rows):3}  judge_v1: answer_v1 {b1:5.1f}, {cond} {c1:5.1f}  |  Memora's judge: answer_v1 {bm:5.1f}, {cond} {cm:5.1f}")
print(f"\nPer-condition reports: {OUT.relative_to(ROOT)}/<condition>-report.txt; progress: {(OUT / 'progress.log').relative_to(ROOT)}")
