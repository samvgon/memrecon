"""E45: how much of the judges' disagreement on FAA comes from our labelling rules?

Question: in E44 the two judges (Qwen and Gemma, both with `judge_v1`) agreed on about 88% of
sub-questions, MPA barely moved with the judge, but FAA moved by 26-29 points: Qwen-as-judge counts an
answer that names a preference as no longer held ("you used to like thrillers") as mentioning it, which
`judge_v1`'s first rule (Sam's, D61) says does not count. Memora's own judge prompt has no such rule.
Is the forgetting question ambiguous as Memora asks it (then without the rule the judges should split at
least as often), or is the split Qwen not following our rule (then without the rule Gemma should move
toward Qwen)?
Method (development personas only): the same 454 sub-questions and the same two answer sets as E44
(Qwen's Stage 0 answers and Gemma's rewrites from the same prompts, read from `results/e44/verdicts.jsonl`),
judged again by both models with Memora's own judge prompt, copied verbatim from
evals/agent_eval/memory_to_answer.py, `_evaluate_with_single_judge` (commit a6493188), and its reply read
the way Memora reads it (fence stripping, JSON, then its yes/no inference from the text; anything else is
"unclear" and fails, as in Memora). One call per verdict at temperature 0, as Memora. The `judge_v1`
verdicts are E44's. Reported for both prompts: the writer-by-judge pass rates and E44's difference in
differences, agreement between the judges, FAMA/MPA/FAA per cell, how each judge's verdicts change between
the prompts, and both judges against Sam's 70 hand labels, split into labels she gave directly and labels
that came from her rules.
Not the same as Memora's setup: our judges are local models with thinking off, one judge per verdict
instead of a majority of three (GPT-4.1, Claude Haiku 4.5, Gemini 2.5 Flash); the prompt and the reading
of the reply are Memora's.
Needs E44 to have run (its verdicts file), and the agent (Qwen) and judge (Gemma) servers.
Pausing: every call is cached, so Ctrl-C at any time and rerun to resume. `--cached-only` (or
MEMRECON_E45_CACHED_ONLY=1) makes no calls and reports on what is done. `MEMRECON_E45_PILOT=N` takes only
N questions per persona and arm (not evidence).
"""
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

from common import ROOT
from memrecon.fama import fama_score, table_cell
from memrecon.judge import Judge, cohen_kappa
from memrecon.llm import ModelCallError, OutputTruncated, cache_key, client as model_client, endpoint
from memrecon.monitor import wait_for_memory

CACHED_ONLY = "--cached-only" in sys.argv or os.environ.get("MEMRECON_E45_CACHED_ONLY") == "1"
PILOT = int(os.environ.get("MEMRECON_E45_PILOT", "0") or 0)
E44 = ROOT / "results" / "e44" / "verdicts.jsonl"
OUT = ROOT / "results" / "e45"
LABELS = ROOT / "notes" / "labels"
WRITERS = ("qwen", "gemma")
JUDGES = ("qwen", "gemma")
PROMPTS = ("rules", "memora")  # judge_v1 (E44's verdicts) and Memora's own prompt
ROLE = {"qwen": "agent", "gemma": "judge"}

# Memora's judge prompt, verbatim (memory_to_answer.py, _evaluate_with_single_judge, a6493188).
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
    """A model client that, with --cached-only, answers only from the call cache."""

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
    """Read a reply the way Memora does: (answer, how, explanation). answer is yes, no, unclear or error."""
    response_text = (content or "").strip()
    how = "json"
    try:
        if response_text.startswith("```"):
            how = "json in a code fence"
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            low = response_text.lower()
            if "yes" in low and "no" not in low:
                return "yes", "inferred from text", ""
            if "no" in low and "yes" not in low:
                return "no", "inferred from text", ""
            return "unclear", "unreadable", ""
        answer = result.get("answer", "").lower().strip()
        float(result.get("confidence", 0.0))
        explanation = result.get("explanation", "")
        if answer not in ("yes", "no"):
            low = explanation.lower()
            if "yes" in low and "no" not in low:
                answer, how = "yes", "inferred from explanation"
            elif "no" in low and "yes" not in low:
                answer, how = "no", "inferred from explanation"
            else:
                answer, how = "unclear", "no yes/no answer"
        return answer, how, explanation
    except Exception as e:  # Memora: any other failure is an error
        return "error", f"error: {type(e).__name__}", ""


def memora_judge(c, response, question, meta):
    messages = [{"role": "system", "content": MEMORA_SYSTEM},
                {"role": "user", "content": MEMORA_USER.replace("{generated_answer}", response).replace("{eval_question}", question)}]
    limit = 1024
    for attempt in range(1, 4):
        try:
            result = c.chat(messages, max_tokens=limit, tag="E45", meta={"step": "judge-memora", "attempt": attempt, **meta})
        except OutputTruncated:
            limit *= 2
            continue
        except ModelCallError as e:
            if "reasoning" in str(e) or "unreadable" in str(e):
                continue
            return None, "call failed", str(e)
        return memora_read(result.content)
    return None, "call failed", "no usable reply after 3 attempts"


def did(per_q):
    tot = defaultdict(lambda: [0, 0])
    for cells in per_q:
        for cell, (p, n) in cells.items():
            tot[cell][0] += p
            tot[cell][1] += n
    r = {c: p / n if n else float("nan") for c, (p, n) in tot.items()}
    return (r[("qwen", "qwen")] - r[("qwen", "gemma")]) - (r[("gemma", "qwen")] - r[("gemma", "gemma")])


def boot(per_q, reps=2000, seed=44):
    rng = random.Random(seed)
    keys = list(per_q)
    stats = sorted(did([per_q[rng.choice(keys)] for _ in keys]) for _ in range(reps))
    return stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


if not E44.exists():
    sys.exit(f"{E44.relative_to(ROOT)} not found: run E44 first.")
e44 = [json.loads(x) for x in E44.read_text(encoding="utf-8").splitlines() if x.strip()]
print(f"Sub-questions and answers: {E44.relative_to(ROOT)} ({len(e44)} rows); rules = judge_v1 (E44's verdicts); "
      f"memora = Memora's judge prompt (a6493188)")
print(f"Qwen = {endpoint('agent').model}; Gemma = {endpoint('judge').model}")
print("Mode: " + ("CACHED ONLY (no model calls; partial results)" if CACHED_ONLY else "full")
      + (f"; PILOT {PILOT} questions per unit (not evidence)" if PILOT else ""))
print()

# questions in E44's order, grouped by persona and arm
order, by_q = [], defaultdict(list)
for r in e44:
    k = (r["persona"], r["arm"], r["question_id"])
    if k not in by_q:
        order.append(k)
    by_q[k].append(r)
if PILOT:
    seen = Counter()
    keep = []
    for k in order:
        if seen[k[:2]] < PILOT:
            keep.append(k)
            seen[k[:2]] += 1
    order = keep

client = {j: Gate(model_client(ROLE[j])) for j in JUDGES}
OUT.mkdir(parents=True, exist_ok=True)
progress = open(OUT / "progress.log", "a", encoding="utf-8")
rows, hows, failed = [], {j: Counter() for j in JUDGES}, 0
t0 = time.monotonic()
for n, k in enumerate(order, start=1):
    if n % 5 == 1:
        wait_for_memory(say=print)
    for r in by_q[k]:
        v = {("rules", w, j): r["v"].get(f"{w}|{j}") for w in WRITERS for j in JUDGES}
        notes = {}
        for w in WRITERS:
            for j in JUDGES:
                if not r["text"].get(w):
                    v[("memora", w, j)] = None
                    continue
                try:
                    ans, how, why = memora_judge(client[j], r["text"][w], r["sub"],
                                                 {"persona": r["persona"], "arm": r["arm"], "sub_id": r["sub_id"], "writer": w})
                except NotCached:
                    v[("memora", w, j)] = None
                    continue
                hows[j][how] += 1
                failed += ans is None
                v[("memora", w, j)] = ans
                notes[(w, j)] = why
        rows.append({**r, "v": v, "why": notes})
    progress.write(f"{time.strftime('%H:%M:%S')} {k[0]} {k[1]} {n}/{len(order)} "
                   f"sent qwen {client['qwen'].sent} gemma {client['gemma'].sent}\n")
    progress.flush()
progress.close()

complete = [r for r in rows if all(x is not None for x in r["v"].values())]
print(f"Sub-questions with all eight verdicts: {len(complete)} of {len(rows)} reached "
      f"({time.monotonic() - t0:.0f} s; calls sent: Qwen {client['qwen'].sent}, Gemma {client['gemma'].sent}"
      + (f"; not yet cached: {client['qwen'].skipped + client['gemma'].skipped}" if CACHED_ONLY else "")
      + f"; calls that failed: {failed})")
print("How the Memora-prompt replies were read (Memora's reading; 'unclear' fails, as in Memora):")
for j in JUDGES:
    print(f"  {j:5} judges: " + ", ".join(f"{how} {c}" for how, c in hows[j].most_common()))
if not complete:
    sys.exit(0)


def passed(r, p, w, j):
    return r["v"][(p, w, j)] == r["expected"]


# -- pass rates and self-preference, per prompt --------------------------------------------------------
for p in PROMPTS:
    print(f"\n== Pass rate, writer by judge: {p} prompt ==")
    print(f"{'':16}{'Qwen judges':>14}{'Gemma judges':>14}{'difference':>12}")
    for w in WRITERS:
        rq = sum(passed(r, p, w, "qwen") for r in complete) / len(complete)
        rg = sum(passed(r, p, w, "gemma") for r in complete) / len(complete)
        print(f"{w.capitalize() + ' wrote':16}{100 * rq:13.1f}%{100 * rg:13.1f}%{100 * (rq - rg):+11.1f}")
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in complete:
        for w in WRITERS:
            for j in JUDGES:
                cells[(r["persona"], r["arm"], r["question_id"])][(w, j)][0] += passed(r, p, w, j)
                cells[(r["persona"], r["arm"], r["question_id"])][(w, j)][1] += 1
    per_q = {q: {c: tuple(x) for c, x in cs.items()} for q, cs in cells.items()}
    lo, hi = boot(per_q)
    print(f"Self-preference (difference in differences): {100 * did(per_q.values()):+.1f} points; "
          f"95% bootstrap interval [{100 * lo:+.1f}, {100 * hi:+.1f}] over {len(per_q)} questions")

print("\n== By sub-question type (pass rate, writer/judge) ==")
for t in sorted({r["type"] for r in complete}):
    sub = [r for r in complete if r["type"] == t]
    for p in PROMPTS:
        cells = "  ".join(f"{w}/{j} {100 * sum(passed(r, p, w, j) for r in sub) / len(sub):5.1f}%" for w in WRITERS for j in JUDGES)
        print(f"  {t:20} n={len(sub):4}  {p:6}  {cells}")

# -- agreement between the judges ------------------------------------------------------------------------
print("\n== Agreement between the judges ==")
for p in PROMPTS:
    for w in WRITERS:
        for t in ("all", "forgetting_absence", "memory_presence"):
            sub = [r for r in complete if t == "all" or r["type"] == t]
            a = [r["v"][(p, w, "qwen")] for r in sub]
            b = [r["v"][(p, w, "gemma")] for r in sub]
            agree = sum(x == y for x, y in zip(a, b))
            print(f"  {p:6} on {w:5}'s answers, {t:18}: {agree:3} of {len(sub):3} ({100 * agree / len(sub):5.1f}%), "
                  f"kappa {cohen_kappa(a, b):.3f}; yes: Qwen {a.count('yes')}, Gemma {b.count('yes')}")

# -- FAMA, MPA, FAA ----------------------------------------------------------------------------------
print("\n== FAMA, MPA, FAA in each cell (questions with all their sub-questions judged) ==")
qs = defaultdict(list)
for r in complete:
    qs[(r["persona"], r["arm"], r["question_id"])].append(r)
for arm in ("note", "log", "both"):
    for w in WRITERS:
        for j in JUDGES:
            line = []
            for p in PROMPTS:
                f, mp, fa = [], [], []
                for (_, a, _), subs in qs.items():
                    if arm != "both" and a != arm:
                        continue
                    pres = [s for s in subs if s["type"] == "memory_presence"]
                    forg = [s for s in subs if s["type"] != "memory_presence"]
                    pc = sum(passed(s, p, w, j) for s in pres)
                    fc = sum(passed(s, p, w, j) for s in forg)
                    f.append(fama_score(pc, len(pres), fc, len(forg)))
                    if pres:
                        mp.append(pc / len(pres))
                    if forg:
                        fa.append(fc / len(forg))
                line.append(f"{p}: FAMA {table_cell(f):5.1f}  MPA {100 * sum(mp) / max(1, len(mp)):5.1f}  "
                            f"FAA {100 * sum(fa) / max(1, len(fa)):5.1f}")
            print(f"  {arm:5} {w:5} wrote, {j:5} judges | " + " | ".join(line))

# -- how each judge's verdicts change between the prompts -------------------------------------------
print("\n== Verdicts that change from the rules prompt to Memora's (both answer sets) ==")
for j in JUDGES:
    for t in ("forgetting_absence", "memory_presence"):
        ch = Counter()
        n = 0
        for r in complete:
            if r["type"] != t:
                continue
            for w in WRITERS:
                n += 1
                a, b = r["v"][("rules", w, j)], r["v"][("memora", w, j)]
                if a != b:
                    ch[f"{a}->{b}"] += 1
        print(f"  {j:5} judges, {t:18}: {sum(ch.values()):3} of {n} change ("
              + ", ".join(f"{k} {c}" for k, c in sorted(ch.items())) + ")")

# -- against Sam's hand labels -------------------------------------------------------------------------
print("\n== Both judges, both prompts, against Sam's hand labels (70 judge-check items; Qwen's answers) ==")
sample = {json.loads(x)["item"]: json.loads(x) for x in (LABELS / "2026-11-08-judge-sample.jsonl").read_text().splitlines() if x.strip()}
labels = {json.loads(x)["item"]: json.loads(x) for x in (LABELS / "2026-11-08-judge-labels-resolved.jsonl").read_text().splitlines() if x.strip()}
rules_judge = {j: Judge(client[j], tag="E45") for j in JUDGES}
hl = defaultdict(list)
for k in sorted(sample):
    s, lab = sample[k], labels[k]
    kind = "by her rules" if lab["source"].startswith("Sam's rule") else "directly"
    for j in JUDGES:
        for p in PROMPTS:
            meta = {"item": k, "sub_id": s["sub_id"], "persona": s["persona"], "arm": s["arm"]}
            try:
                if p == "rules":
                    ans = rules_judge[j].judge(s["answer"], s["sub_question"], meta)["answer"]
                else:
                    ans = memora_judge(client[j], s["answer"], s["sub_question"], meta)[0]
            except NotCached:
                ans = None
            hl[(s["set"], j, p)].append((lab["label"], ans, kind))
for st in ("check", "tune"):
    for j in JUDGES:
        for p in PROMPTS:
            pairs = [x for x in hl[(st, j, p)] if x[1] is not None]
            if not pairs:
                continue
            agree = sum(l == a for l, a, _ in pairs)
            split = "; ".join(f"labels given {kind}: {sum(l == a for l, a, kd in pairs if kd == kind)} of "
                              f"{sum(kd == kind for _, _, kd in pairs)}" for kind in ("directly", "by her rules"))
            print(f"  {st:5} set, {j:5} judges, {p:6}: {agree} of {len(pairs)} ({100 * agree / len(pairs):.1f}%), "
                  f"kappa {cohen_kappa([l for l, _, _ in pairs], [a for _, a, _ in pairs]):.3f}   ({split})")
print("  (judge_v1 was adjusted on the tune set with Gemma; Memora's prompt was not adjusted at all)")

# -- examples ------------------------------------------------------------------------------------------
print("\n== Examples: forgetting sub-questions where Gemma's verdict changes between the prompts ==")
ex = [(r, w) for r in complete if r["type"] == "forgetting_absence" for w in WRITERS
      if r["v"][("rules", w, "gemma")] != r["v"][("memora", w, "gemma")]]
for r, w in random.Random(45).sample(ex, min(6, len(ex))):
    print(f"--- {r['persona']} {r['arm']} {r['sub_id']} (expected {r['expected']}); {w} wrote; Gemma: rules "
          f"{r['v'][('rules', w, 'gemma')]}, Memora's prompt {r['v'][('memora', w, 'gemma')]}; Qwen: rules "
          f"{r['v'][('rules', w, 'qwen')]}, Memora's prompt {r['v'][('memora', w, 'qwen')]}")
    print(f"  Q: {r['sub']}")
    t = " ".join((r["text"][w] or "").split())
    print(f"  Answer: {t[:400]}{'…' if len(t) > 400 else ''}")
    why = " ".join((r["why"].get((w, "gemma")) or "").split())
    print(f"  Gemma's explanation (Memora's prompt): {why[:300]}")

with open(OUT / "verdicts.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({**r, "v": {"|".join(c): x for c, x in r["v"].items()},
                            "why": {"|".join(c): x for c, x in r["why"].items()}}) + "\n")
print(f"\nVerdicts: {(OUT / 'verdicts.jsonl').relative_to(ROOT)}; progress: {(OUT / 'progress.log').relative_to(ROOT)}")
