"""E49: how much FAMA can an answer earn with no memory at all, from the persona's profile alone?

Question: the Memora paper explains why plain LLMs stay competitive on recommending (§4, Table 4): models
"can still receive credit by generating plausible suggestions that are broadly consistent with the user's
current preferences, even when explicit memory retrieval is incomplete, because they can infer likely
preferences from persona-aligned conversational cues". The paper does not measure this. The persona cues
come from Memora's own pipeline: each persona has a preference archetype per domain (e.g. sales_manager's
books are `thriller_fan`), and both the current and the outdated items in the criteria are drawn from that
archetype's pool. So how much credit does knowing only the profile earn, against answering from memory?
Method (development personas only; Qwen writes, Gemma judges): the 45 Stage 0 questions (15 per persona)
answered again with no memories, under two conditions, one at a time:
  generic   the question alone, system prompt "You are a helpful AI assistant." No memory, no profile.
            How much credit is there for a generic answer?
  persona   Memora's answering prompt, verbatim (as E46), with the persona's profile in place of the
            memories: one line with the persona's description, then one line per preference domain with
            the archetype's description, from Memora's generation metadata
            (data_generation/meta_data/persona_traits.json, preference_metadata.json). Lines are shown as
            "i. text", without the relevance score (there is none). Nothing from any conversation.
Each answer is judged by Gemma twice: with `judge_v1` (the validated judge, D61) and with Memora's judge
prompt, verbatim and read as Memora reads it (as E45 and E47). The comparison is E47's output
(`results/e47/memora.jsonl`): the same questions answered by Qwen from the real top-50 memories with Memora's
answering prompt (E46's `memora`) and with `answer_v1` (Stage 0), each arm, judged both ways.
Reported, by category and overall: FAMA/MPA/FAA for every condition under both judge prompts; on
recommending, the paired FAMA difference persona − memory (each arm) with a bootstrap interval (seed 49);
how many forgetting sub-questions each condition is said to mention; examples. Also, without model calls,
how many recommending criteria name an item from the persona's own archetype pool (current and outdated).
Not Memora's setup: Qwen at temperature 0 with thinking off (Memora: gpt-4o-mini); our retrieval for the
comparison; 15 recommending questions, so intervals are wide. Memora itself, with no memories, returns a
fixed refusal and makes no call ("I don't have enough information..."); `generic` is the closest model
answer to that.
Pausing: every call is cached; Ctrl-C and rerun to resume. `--cached-only` (or MEMRECON_E49_CACHED_ONLY=1)
makes no calls. `MEMRECON_E49_PILOT=N` takes N questions per persona (not evidence).
`MEMRECON_E49_CONDITIONS` to choose and order them (default generic,persona). Needs the agent (Qwen) and
judge (Gemma) servers, E46's Stage 0 answers and E47's output.
"""
import ast
import json
import os
import random
import re
import sys
import time

from common import ROOT
from memrecon.fama import fama_score, table_cell
from memrecon.judge import JUDGE_PROMPT, Judge
from memrecon.llm import ModelCallError, OutputTruncated, cache_key, client as model_client, endpoint
from memrecon.memora import dev_personas
from memrecon.monitor import wait_for_memory

sys.stdout.reconfigure(line_buffering=True)
CACHED_ONLY = "--cached-only" in sys.argv or os.environ.get("MEMRECON_E49_CACHED_ONLY") == "1"
PILOT = int(os.environ.get("MEMRECON_E49_PILOT", "0") or 0)
CONDITIONS = [c.strip() for c in os.environ.get("MEMRECON_E49_CONDITIONS", "generic,persona").split(",") if c.strip()]
for c in CONDITIONS:
    if c not in ("generic", "persona"):
        sys.exit(f"Unknown condition {c!r}: use generic, persona.")
ANSWERS = ROOT / "results" / "stage0" / "answers"
E47 = ROOT / "results" / "e47" / "memora.jsonl"
META = ROOT / "data" / "memora" / "data_generation" / "meta_data"
OUT = ROOT / "results" / "e49"
CATS = ("remembering", "reasoning", "recommending")


def consts(script, names):
    """String constants read from another explore script, so the prompts are byte-identical to it."""
    tree = ast.parse((ROOT / "scripts" / "explore" / script).read_text(encoding="utf-8"))
    return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names}


ANS = consts("E46_answer_prompt.py", ("MEMORA_SYSTEM", "MEMORA_USER"))      # Memora's answering prompt
JDG = consts("E47_answer_prompt_memora_judge.py", ("MEMORA_SYSTEM", "MEMORA_USER"))  # Memora's judge prompt
assert len(ANS) == 2 and len(JDG) == 2, "prompt constants not found in E46/E47"
GENERIC_SYSTEM = "You are a helpful AI assistant."


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
    """Memora's reading of a judge reply (memory_to_answer.py, _evaluate_with_single_judge), as E45/E47."""
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


def call(c, messages, meta):
    limit = 1024
    for attempt in range(1, 4):
        try:
            return c.chat(messages, max_tokens=limit, tag="E49", meta={**meta, "attempt": attempt}).content
        except OutputTruncated:
            limit *= 2
        except ModelCallError as e:
            if "reasoning" in str(e) or "unreadable" in str(e):
                continue
            return None
    return None


def memora_judge(response, question, meta):
    messages = [{"role": "system", "content": JDG["MEMORA_SYSTEM"]},
                {"role": "user", "content": JDG["MEMORA_USER"].replace("{generated_answer}", response).replace("{eval_question}", question)}]
    content = call(gemma, messages, {"step": "judge-memora", **meta})
    return None if content is None else memora_read(content)


# -- the persona profiles -----------------------------------------------------------------------------
for f in ("persona_traits.json", "preference_metadata.json"):
    if not (META / f).exists():
        sys.exit(f"Missing {(META / f).relative_to(ROOT)}: the Memora clone needs data_generation/ (commit a6493188).")
TRAITS = json.loads((META / "persona_traits.json").read_text(encoding="utf-8"))["persona_profiles"]
PREFS = json.loads((META / "preference_metadata.json").read_text(encoding="utf-8"))["preference_personas"]


def profile_lines(persona):
    t = TRAITS[persona]
    lines = [f"Profile: {t['description']}."]
    for domain, archetype in (t["category_traits"].get("preference_memory") or {}).items():
        if archetype and archetype in PREFS.get(domain, {}):
            lines.append(f"{domain.capitalize()}: {PREFS[domain][archetype]['description']}")
    return lines


def pool(persona):
    """Every item string in the persona's preference archetypes, lower-cased."""
    items = set()
    for domain, archetype in (TRAITS[persona]["category_traits"].get("preference_memory") or {}).items():
        for v in (PREFS.get(domain, {}).get(archetype, {}).get("preferences") or {}).values():
            if isinstance(v, list):
                items.update(str(x).lower() for x in v if str(x).strip())
    return items


def answer(cond, row, persona):
    if cond == "generic":
        messages = [{"role": "system", "content": GENERIC_SYSTEM}, {"role": "user", "content": row["question"]}]
    else:
        mem = "\n".join(f"{i}. {t}" for i, t in enumerate(profile_lines(persona), start=1))
        messages = [{"role": "system", "content": ANS["MEMORA_SYSTEM"]},
                    {"role": "user", "content": ANS["MEMORA_USER"].replace("{question}", row["question"]).replace("{memory_text}", mem)}]
    content = call(writer, messages, {"step": f"answer-{cond}", "persona": persona, "question": row["question_id"]})
    return (content or "").strip() or None


def config_dir():
    name = os.environ.get("MEMRECON_E46_CONFIG")
    if name:
        return ANSWERS / name
    done = [p for p in ANSWERS.iterdir() if p.is_dir() and all((p / q / a / "answers.jsonl").exists()
                                                               for q in dev_personas() for a in ("note", "log"))]
    if not done:
        sys.exit(f"No complete Stage 0 answering run in {ANSWERS}.")
    return max(done, key=lambda p: p.stat().st_mtime)


# -- scoring --------------------------------------------------------------------------------------------
def per_q(r, key):
    pres = [(s, v) for s, v in zip(r["subs"], r[key]) if s["type"] == "memory_presence"]
    forg = [(s, v) for s, v in zip(r["subs"], r[key]) if s["type"] != "memory_presence"]
    pc = sum(v == str(s["expected"]).lower() for s, v in pres)
    fc = sum(v == str(s["expected"]).lower() for s, v in forg)
    return fama_score(pc, len(pres), fc, len(forg)), (pc / len(pres) if pres else None), (fc / len(forg) if forg else None)


def parts(rows, key):
    q = [per_q(r, key) for r in rows]
    mp = [x[1] for x in q if x[1] is not None]
    fa = [x[2] for x in q if x[2] is not None]
    return (table_cell(x[0] for x in q), 100 * sum(mp) / max(1, len(mp)),
            (100 * sum(fa) / len(fa)) if fa else float("nan"), [x[0] for x in q])


def boot(d, reps=2000, seed=49):
    rng = random.Random(seed)
    stats = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(reps))
    return 100 * sum(d) / len(d), 100 * stats[int(0.025 * reps)], 100 * stats[int(0.975 * reps) - 1]


# -- run ------------------------------------------------------------------------------------------------
cfg = config_dir()
print(f"Questions: {cfg.relative_to(ROOT)} (Stage 0, note arm's rows); comparison: {E47.relative_to(ROOT)}")
print(f"Qwen = {endpoint('agent').model}; Gemma = {endpoint('judge').model}; judge prompts {JUDGE_PROMPT} and Memora's")
print("Conditions, in order: " + ", ".join(CONDITIONS))
print("Mode: " + ("CACHED ONLY (no model calls; partial results)" if CACHED_ONLY else "full")
      + (f"; PILOT {PILOT} questions per persona (not evidence)" if PILOT else ""))
for p in dev_personas():
    print(f"  {p} profile, as shown to the persona condition:")
    for i, t in enumerate(profile_lines(p), start=1):
        print(f"    {i}. {t}")
print()

writer = Gate(model_client("agent"))
gemma = Gate(model_client("judge"))
judge = Judge(gemma, tag="E49")
OUT.mkdir(parents=True, exist_ok=True)
progress = open(OUT / "progress.log", "a", encoding="utf-8")

qs = []
for persona in dev_personas():
    rows = [json.loads(x) for x in (cfg / persona / "note" / "answers.jsonl").read_text().splitlines() if x.strip()]
    qs += [(persona, r) for r in (rows[:PILOT] if PILOT else rows)]

# Comparison rows from E47 (real memories): keyed by (persona, question_id), per arm.
ref = {}
if E47.exists():
    for x in E47.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            ref[(r["persona"], r["arm"], r["question_id"])] = r
else:
    print(f"Note: {E47.relative_to(ROOT)} not found; the comparison with real memories is skipped (run E46 and E47).\n")

# Descriptive, no model calls: recommending criteria naming an item from the persona's own archetype pool.
print("== Recommending criteria and the persona's archetype pool (no model calls) ==")
for typ in ("memory_presence", "forgetting_absence"):
    n = hit = 0
    for persona, r in qs:
        if r["category"] != "recommending":
            continue
        items = pool(persona)
        for s in r["sub_questions"]:
            if s["type"] != typ:
                continue
            n += 1
            low = s["text"].lower()
            hit += any(re.search(r"(?<![a-z])" + re.escape(it) + r"(?![a-z])", low) for it in items if len(it) > 2)
    print(f"  {typ:20} {hit} of {n} name an item from the persona's own archetype pool")
print()

done = {}
for cond in CONDITIONS:
    t0, s0 = time.monotonic(), (writer.sent, gemma.sent)
    out = []
    for n, (persona, row) in enumerate(qs, start=1):
        if n % 5 == 1:
            wait_for_memory(say=print)
        try:
            text = answer(cond, row, persona)
            if text is None:
                continue
            meta = {"persona": persona, "writer": f"qwen-{cond}"}
            v1 = [judge.judge(text, s["text"], {**meta, "sub_id": s["id"]})["answer"] for s in row["sub_questions"]]
            mem = [memora_judge(text, s["text"], {**meta, "sub_id": s["id"]}) for s in row["sub_questions"]]
        except NotCached:
            continue
        if None in v1 or None in mem:
            continue
        out.append({"persona": persona, "question_id": row["question_id"], "category": row["category"],
                    "question": row["question"], "subs": row["sub_questions"], "text": text, "v1": v1, "mem": mem})
        progress.write(f"{time.strftime('%H:%M:%S')} {cond} {n}/{len(qs)} sent agent {writer.sent} judge {gemma.sent}\n")
        progress.flush()
    done[cond] = out
    with open(OUT / f"{cond}.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    lines = [f"== Condition: {cond}: {len(out)} of {len(qs)} questions answered and judged =="]
    lines.append(f"  {'category':14}{'n':>4}   judge_v1: FAMA   MPA   FAA  | Memora's judge: FAMA   MPA   FAA")
    for cat in CATS + ("all",):
        sub = [r for r in out if cat == "all" or r["category"] == cat]
        if sub:
            a, b = parts(sub, "v1"), parts(sub, "mem")
            lines.append(f"  {cat:14}{len(sub):4}   {a[0]:15.1f}{a[1]:6.1f}{a[2]:6.1f}  | {b[0]:20.1f}{b[1]:6.1f}{b[2]:6.1f}")
    lines.append(f"({time.monotonic() - t0:.0f} s; calls sent: agent {writer.sent - s0[0]}, judge {gemma.sent - s0[1]}"
                 + (f"; not yet cached: {writer.skipped + gemma.skipped}" if CACHED_ONLY else "") + ")")
    text = "\n".join(lines)
    (OUT / f"{cond}-report.txt").write_text(text + "\n", encoding="utf-8")
    print(text + "\n")
progress.close()

# -- against answering from real memories ------------------------------------------------------------------
if ref and done:
    print("== Every condition side by side (questions every condition has; FAMA, with MPA/FAA) ==")
    keys = None
    for rows in done.values():
        ks = {(r["persona"], r["question_id"]) for r in rows}
        keys = ks if keys is None else keys & ks
    keys = {k for k in keys if all((k[0], arm, k[1]) in ref for arm in ("note", "log"))}
    print(f"{len(keys)} questions\n")
    for cat in CATS + ("all",):
        ks = sorted(k for k in keys if cat == "all" or ref[(k[0], "note", k[1])]["category"] == cat)
        if not ks:
            continue
        print(f"  {cat} (n={len(ks)})            judge_v1: FAMA   MPA   FAA  | Memora's judge: FAMA   MPA   FAA")
        cells = []
        for cond, rows in done.items():
            by = {(r["persona"], r["question_id"]): r for r in rows}
            cells.append((f"{cond} (no memory)", [by[k] for k in ks], "v1", "mem"))
        for arm in ("note", "log"):
            rr = [ref[(k[0], arm, k[1])] for k in ks]
            cells.append((f"memora prompt, {arm}", rr, "cond_v1", "cond_mem"))
            cells.append((f"answer_v1, {arm}", rr, "base_v1", "base_mem"))
        for name, rr, k1, k2 in cells:
            a, b = parts(rr, k1), parts(rr, k2)
            print(f"    {name:26}{a[0]:12.1f}{a[1]:6.1f}{a[2]:6.1f}  | {b[0]:20.1f}{b[1]:6.1f}{b[2]:6.1f}")
        print()
    if "persona" in done:
        by = {(r["persona"], r["question_id"]): r for r in done["persona"]}
        ks = sorted(k for k in keys if ref[(k[0], "note", k[1])]["category"] == "recommending")
        if ks:
            print("== Recommending: persona only − answering from memory (Memora's answering prompt), paired by question ==")
            for arm in ("note", "log"):
                for jn, kp, kr in (("judge_v1", "v1", "cond_v1"), ("Memora's judge", "mem", "cond_mem")):
                    p = [per_q(by[k], kp)[0] for k in ks]
                    m = [per_q(ref[(k[0], arm, k[1])], kr)[0] for k in ks]
                    d, lo, hi = boot([x - y for x, y in zip(p, m)])
                    share = (sum(p) / sum(m)) if sum(m) else float("nan")
                    print(f"  {arm:4} {jn:15} persona {100 * sum(p) / len(p):5.1f}  memory {100 * sum(m) / len(m):5.1f}  "
                          f"difference {d:+6.1f} [{lo:+.1f}, {hi:+.1f}]  persona reaches {100 * share:4.0f}% of memory's FAMA  (n={len(ks)})")
            print()

# -- how often each condition is said to mention an outdated item -----------------------------------------
if done:
    print("== Forgetting sub-questions the judge says the answer mentions (should be 'no') ==")
    for cond, rows in done.items():
        for jn, k in (("judge_v1", "v1"), ("Memora's judge", "mem")):
            n = sum(1 for r in rows for s in r["subs"] if s["type"] != "memory_presence")
            y = sum(v == "yes" for r in rows for s, v in zip(r["subs"], r[k]) if s["type"] != "memory_presence")
            print(f"  {cond:8} {jn:15} {y} of {n}")
    print()

# -- examples ---------------------------------------------------------------------------------------------
if "persona" in done:
    rec = [r for r in done["persona"] if r["category"] == "recommending"]
    if rec:
        best = max(rec, key=lambda r: per_q(r, "mem")[0])
        pick = [best] + random.Random(49).sample([r for r in rec if r is not best], min(2, len(rec) - 1))
        print("== Examples: persona-only answers to recommending questions (highest FAMA under Memora's judge, then 2 at random) ==")
        for r in pick:
            print(f"--- {r['persona']} {r['question_id']}: {r['question']}")
            t = " ".join(r["text"].split())
            print(f"  Answer: {t[:600]}{'…' if len(t) > 600 else ''}")
            for s, a, b in zip(r["subs"], r["v1"], r["mem"]):
                e = str(s["expected"]).lower()
                print(f"    [{'P' if s['type'] == 'memory_presence' else 'F'}] expected {e:3}  judge_v1 {a:3}  Memora's {b:7}  {s['text']}")
            print(f"  FAMA: judge_v1 {100 * per_q(r, 'v1')[0]:.0f}, Memora's judge {100 * per_q(r, 'mem')[0]:.0f}")
        print()
print(f"Per-condition reports: {OUT.relative_to(ROOT)}/<condition>-report.txt; answers and verdicts: {OUT.relative_to(ROOT)}/<condition>.jsonl")
