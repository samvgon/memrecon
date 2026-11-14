"""E50: the same model with the whole history in context, against the same model with retrieved memories.

Question: the Memora paper compares memory agents, which all answer through GPT-4o-mini, with frontier LLMs
(GPT-5.2, Claude Sonnet 4.5, Gemini 3 Pro, Qwen3-32B) reading the history in context. GPT-4o-mini is never
run in the LLM setting, so every agent-versus-LLM comparison changes the answering model as well as the
memory. Holding the model fixed: how does Qwen do reading the whole history, against Qwen answering from its
retrieved top-50 memories (E46/E47), on the same questions?
Method (development personas only; Qwen writes, Gemma judges): the 45 Stage 0 questions answered once each
by Qwen under Memora's LLM-setting prompt, verbatim (evals/model_eval/model_based_evaluator.py,
`build_question_prompt`, a6493188), with the history built as Memora's `build_conversation_context` builds it:
"## Session <id> - <date>" then "User:" / "Assistant:" turns, every session in order, most recent kept first
if the budget is exceeded (whole sessions, then part of the next if more than 100 tokens remain, and Memora's
truncation note). Memora reads `session_date` for the current date and the released questions use
`question_date`, so its prompt never carries a date (as in the agent setting); neither does this one.
Budget: Memora's formula on our server's context, int(ctx x 0.7) - 10,000 tokens of history, with ctx =
AGENT_CTX from config/models.env (131,072; `MEMRECON_E50_CTX` to override). Tokens are counted with the
agent server's own tokenizer (llama-server /tokenize); the script prints each persona's real history length
and whether anything was cut. (E09's 62-74k was a characters/4 estimate.) The history comes first and the
question last, so llama-server can reuse the processed history across a persona's 15 questions.
Each answer is judged by Gemma with `judge_v1` and with Memora's judge prompt (as E49). Compared with E47's
answers to the same questions from the real top-50 memories (Memora's agent prompt and `answer_v1`, each arm)
and with E49's no-memory answers: FAMA/MPA/FAA by category, paired FAMA differences full history − memory
(each arm, each judge) with bootstrap intervals (seed 50), outdated items mentioned, examples.
Not Memora's setup: Qwen3.6-35B-A3B at temperature 0 with thinking off (Memora: temperature 0.7, and for
Qwen3-32B a 40,960-token window that its code cuts to about 18.7k tokens of history); our judges.
Pausing: every call is cached; Ctrl-C and rerun to resume. `--cached-only` (or MEMRECON_E50_CACHED_ONLY=1)
makes no calls. `MEMRECON_E50_PILOT=N` takes N questions per persona (not evidence). Needs the agent and judge
servers, Stage 0's answers and E47's output (E49's output optional). Expect a minute or two for each
persona's first question (the whole history is processed), then much less.
"""
import ast
import json
import os
import random
import sys
import time
import urllib.request

from common import ROOT
from memrecon.fama import fama_score, table_cell
from memrecon.judge import JUDGE_PROMPT, Judge
from memrecon.llm import ModelCallError, OutputTruncated, cache_key, client as model_client, endpoint, models_file, roles
from memrecon.memora import dev_personas, sessions
from memrecon.monitor import wait_for_memory

sys.stdout.reconfigure(line_buffering=True)
CACHED_ONLY = "--cached-only" in sys.argv or os.environ.get("MEMRECON_E50_CACHED_ONLY") == "1"
PILOT = int(os.environ.get("MEMRECON_E50_PILOT", "0") or 0)
ANSWERS = ROOT / "results" / "stage0" / "answers"
E47 = ROOT / "results" / "e47" / "memora.jsonl"
E49 = ROOT / "results" / "e49"
OUT = ROOT / "results" / "e50"
CATS = ("remembering", "reasoning", "recommending")

# Memora's LLM-setting prompt (build_question_prompt, a6493188), verbatim, with no date (see the docstring).
LLM_SYSTEM = """You are an AI assistant with memory capabilities. The conversation history below contains ALL your previous conversations with this user - this history is PROVIDED TO YOU in this prompt.

Your task is to answer the user's question based EXCLUSIVELY on the conversation history that is provided below. The conversation history contains all the information you need to answer questions about the user's preferences, activities, goals, and past interactions.

CRITICAL INSTRUCTIONS:
1. The conversation history is PROVIDED IN THIS PROMPT - you have full access to it
2. Answer the question using ONLY information from the conversation history provided below
3. Do NOT say "I don't have access" or "I cannot access" - you DO have access because the history is right here in this prompt
4. Extract relevant information from the conversation history and use it to answer the question
5. If you learned about user preferences, activities, content, or goals from the conversations below, use them in your response
6. If information was mentioned in earlier conversations but later updated or deleted, use only the most recent information
7. Be specific and reference relevant details from the conversations
8. If the conversation history doesn't contain enough information to answer the question, say so clearly (but do NOT say you don't have access)
9. Do not make up information that wasn't in the conversations"""
LLM_USER = """{context}

---

# Current Question
User: {question}

Based on the conversation history above, please provide your answer:"""
HEADER = "# Conversation History\nThe following contains all your previous conversations with this user, in chronological order:\n\n"


def consts(script, names):
    tree = ast.parse((ROOT / "scripts" / "explore" / script).read_text(encoding="utf-8"))
    return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names}


JDG = consts("E47_answer_prompt_memora_judge.py", ("MEMORA_SYSTEM", "MEMORA_USER"))  # Memora's judge prompt
assert len(JDG) == 2, "judge prompt constants not found in E47"


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
    """Memora's reading of a judge reply, as E45/E47/E49."""
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


def call(c, messages, meta, limit=1024):
    for attempt in range(1, 4):
        try:
            return c.chat(messages, max_tokens=limit, tag="E50", meta={**meta, "attempt": attempt}).content
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


# -- the history, as Memora's build_conversation_context builds it ------------------------------------------
TOK_URL = endpoint("agent").base_url.removesuffix("/v1") + "/tokenize"
_tok_cache = {}


def ntok(text):
    """Tokens by the agent server's own tokenizer (llama-server /tokenize)."""
    if text not in _tok_cache:
        req = urllib.request.Request(TOK_URL, data=json.dumps({"content": text}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            _tok_cache[text] = len(json.loads(r.read())["tokens"])
    return _tok_cache[text]


def session_text(s):
    parts = [f"\n## Session {s.get('session_id', 'unknown')} - {s.get('date', 'unknown')}\n"]
    for turn in s.get("conversation", []):
        if turn.get("speaker") == "ai_agent":
            parts.append(f"Assistant: {turn.get('message', '')}")
        elif turn.get("speaker") == "user":
            parts.append(f"User: {turn.get('message', '')}")
    parts.append("")
    return "\n".join(parts)


def history(persona, budget):
    convs = sorted(sessions(persona), key=lambda s: s.get("session_id", 0))
    meta = {"total": len(convs), "included": 0, "partial": False, "truncated": False}
    cur = ntok(HEADER)
    avail = budget - cur
    kept = []
    for s in reversed(convs):
        t = session_text(s)
        n = ntok(t)
        if cur + n <= avail:
            kept.append((s.get("session_id", 0), t))
            cur += n
            meta["included"] += 1
            continue
        if cur < avail and n > avail - cur and avail - cur > 100:  # part of this session, as Memora does
            part = f"\n## Session {s.get('session_id', 'unknown')} - {s.get('date', 'unknown')}\n"
            pn = ntok(part)
            for turn in s.get("conversation", []):
                tt = f"{'Assistant' if turn.get('speaker') == 'ai_agent' else 'User'}: {turn.get('message', '')}\n"
                if pn + ntok(tt) <= avail - cur:
                    part += tt
                    pn += ntok(tt)
                else:
                    break
            if pn > ntok(part.split("\n")[0]):
                part += "\n[Conversation truncated due to token limit...]"
                kept.append((s.get("session_id", 0), part))
                cur += pn
                meta["included"] += 1
                meta["partial"] = True
        meta["truncated"] = True
    kept.sort(key=lambda x: x[0] if isinstance(x[0], (int, float)) else 0)
    parts = [HEADER] + [t for _, t in kept]
    if meta["truncated"]:
        parts.insert(1, f"[Note: Showing {meta['included']} of {meta['total']} conversations due to token limits. Most recent conversations are prioritized.]\n\n")
    ctx = "\n".join(parts)
    meta["tokens"] = ntok(ctx)
    meta["oldest_kept"] = min((k for k, _ in kept), default=None)
    return ctx, meta


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


def boot(d, reps=2000, seed=50):
    rng = random.Random(seed)
    stats = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(reps))
    return 100 * sum(d) / len(d), 100 * stats[int(0.025 * reps)], 100 * stats[int(0.975 * reps) - 1]


def config_dir():
    name = os.environ.get("MEMRECON_E46_CONFIG")
    if name:
        return ANSWERS / name
    done = [p for p in ANSWERS.iterdir() if p.is_dir() and all((p / q / a / "answers.jsonl").exists()
                                                               for q in dev_personas() for a in ("note", "log"))]
    if not done:
        sys.exit(f"No complete Stage 0 answering run in {ANSWERS}.")
    return max(done, key=lambda p: p.stat().st_mtime)


# -- run ------------------------------------------------------------------------------------------------
ctx_size = int(os.environ.get("MEMRECON_E50_CTX") or roles(models_file())["agent"].get("CTX") or 131072)
budget = int(ctx_size * 0.7) - 10000
cfg = config_dir()
print(f"Questions: {cfg.relative_to(ROOT)} (Stage 0, note arm's rows); comparison: {E47.relative_to(ROOT)}"
      + (f" and {E49.relative_to(ROOT)}" if E49.exists() else ""))
print(f"Qwen = {endpoint('agent').model}; Gemma = {endpoint('judge').model}; judge prompts {JUDGE_PROMPT} and Memora's")
print(f"Context {ctx_size:,} tokens -> history budget {budget:,} (Memora's formula: ctx x 0.7 - 10,000); tokenizer {TOK_URL}")
print("Mode: " + ("CACHED ONLY (no model calls; partial results)" if CACHED_ONLY else "full")
      + (f"; PILOT {PILOT} questions per persona (not evidence)" if PILOT else ""))

writer = Gate(model_client("agent"))
gemma = Gate(model_client("judge"))
judge = Judge(gemma, tag="E50")
OUT.mkdir(parents=True, exist_ok=True)
progress = open(OUT / "progress.log", "a", encoding="utf-8")

ctxs = {}
print("\n== Each persona's history as shown ==")
for p in dev_personas():
    try:
        ctx, m = history(p, budget)
    except OSError as e:
        sys.exit(f"Could not reach the agent server's tokenizer at {TOK_URL} ({e}); start it with scripts/serve.sh agent.")
    ctxs[p] = ctx
    whole = ntok(HEADER + "\n".join(session_text(s) for s in sessions(p)))
    print(f"  {p:24} whole history {whole:,} tokens; shown {m['tokens']:,} tokens, {m['included']} of {m['total']} sessions"
          + (f" (cut: oldest kept session {m['oldest_kept']}{', one partial' if m['partial'] else ''})" if m["truncated"] else " (nothing cut)"))
print()

out = []
t0 = time.monotonic()
for persona in dev_personas():
    rows = [json.loads(x) for x in (cfg / persona / "note" / "answers.jsonl").read_text().splitlines() if x.strip()]
    for n, row in enumerate(rows[:PILOT] if PILOT else rows, start=1):
        wait_for_memory(say=print)
        messages = [{"role": "system", "content": LLM_SYSTEM},
                    {"role": "user", "content": LLM_USER.replace("{context}", ctxs[persona]).replace("{question}", row["question"])}]
        try:
            t1 = time.monotonic()
            text = (call(writer, messages, {"step": "answer-full-history", "persona": persona, "question": row["question_id"]}) or "").strip() or None
            took = time.monotonic() - t1
            if text is None:
                continue
            meta = {"persona": persona, "writer": "qwen-full-history"}
            v1 = [judge.judge(text, s["text"], {**meta, "sub_id": s["id"]})["answer"] for s in row["sub_questions"]]
            mem = [memora_judge(text, s["text"], {**meta, "sub_id": s["id"]}) for s in row["sub_questions"]]
        except NotCached:
            continue
        if None in v1 or None in mem:
            continue
        out.append({"persona": persona, "question_id": row["question_id"], "category": row["category"],
                    "question": row["question"], "subs": row["sub_questions"], "text": text, "v1": v1, "mem": mem})
        progress.write(f"{time.strftime('%H:%M:%S')} {persona} {n}/{len(rows)} answer {took:.0f} s; sent agent {writer.sent} judge {gemma.sent}\n")
        progress.flush()
progress.close()
with open(OUT / "full_history.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"== Full history: {len(out)} questions answered and judged ({time.monotonic() - t0:.0f} s; calls sent: agent {writer.sent}, judge {gemma.sent}"
      + (f"; not yet cached: {writer.skipped + gemma.skipped}" if CACHED_ONLY else "") + ") ==\n")

# -- against retrieved memories and no memory -------------------------------------------------------------
ref = {}
if E47.exists():
    for x in E47.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            ref[(r["persona"], r["arm"], r["question_id"])] = r
nomem = {}
for cond in ("generic", "persona"):
    f = E49 / f"{cond}.jsonl"
    if f.exists():
        nomem[cond] = {(r["persona"], r["question_id"]): r for r in map(json.loads, filter(str.strip, f.read_text(encoding="utf-8").splitlines()))}

full = {(r["persona"], r["question_id"]): r for r in out}
keys = sorted(k for k in full if all((k[0], arm, k[1]) in ref for arm in ("note", "log")))
if not ref:
    print(f"{E47.relative_to(ROOT)} not found; comparison skipped (run E46 and E47).")
elif keys:
    print(f"== Side by side ({len(keys)} questions every condition has; FAMA, with MPA/FAA) ==\n")
    for cat in CATS + ("all",):
        ks = [k for k in keys if cat == "all" or full[k]["category"] == cat]
        if not ks:
            continue
        print(f"  {cat} (n={len(ks)})            judge_v1: FAMA   MPA   FAA  | Memora's judge: FAMA   MPA   FAA")
        cells = [("full history", [full[k] for k in ks], "v1", "mem")]
        for arm in ("note", "log"):
            rr = [ref[(k[0], arm, k[1])] for k in ks]
            cells.append((f"memories, Memora prompt, {arm}", rr, "cond_v1", "cond_mem"))
            cells.append((f"memories, answer_v1, {arm}", rr, "base_v1", "base_mem"))
        for cond, by in nomem.items():
            if all(k in by for k in ks):
                cells.append((f"no memory, {cond}", [by[k] for k in ks], "v1", "mem"))
        for name, rr, k1, k2 in cells:
            a, b = parts(rr, k1), parts(rr, k2)
            print(f"    {name:32}{a[0]:7.1f}{a[1]:6.1f}{a[2]:6.1f}  | {b[0]:20.1f}{b[1]:6.1f}{b[2]:6.1f}")
        print()
    print("== Full history − retrieved memories (Memora's agent prompt), paired by question ==")
    for cat in CATS + ("all",):
        ks = [k for k in keys if cat == "all" or full[k]["category"] == cat]
        for arm in ("note", "log"):
            for jn, kf, kr in (("judge_v1", "v1", "cond_v1"), ("Memora's judge", "mem", "cond_mem")):
                f_ = [per_q(full[k], kf)[0] for k in ks]
                m_ = [per_q(ref[(k[0], arm, k[1])], kr)[0] for k in ks]
                d, lo, hi = boot([x - y for x, y in zip(f_, m_)])
                print(f"  {cat:13} {arm:4} {jn:15} full {100 * sum(f_) / len(f_):5.1f}  memories {100 * sum(m_) / len(m_):5.1f}  "
                      f"difference {d:+6.1f} [{lo:+.1f}, {hi:+.1f}]  (n={len(ks)})")
    print()

# -- outdated items mentioned, and examples -----------------------------------------------------------------
if out:
    n = sum(1 for r in out for s in r["subs"] if s["type"] != "memory_presence")
    for jn, k in (("judge_v1", "v1"), ("Memora's judge", "mem")):
        y = sum(v == "yes" for r in out for s, v in zip(r["subs"], r[k]) if s["type"] != "memory_presence")
        print(f"Forgetting sub-questions the judge says the full-history answer mentions ({jn}): {y} of {n}")
    print()
    rng = random.Random(50)
    print("== Examples: one full-history answer per category, at random ==")
    for cat in CATS:
        pool_ = [r for r in out if r["category"] == cat]
        if not pool_:
            continue
        r = rng.choice(pool_)
        print(f"--- {r['persona']} {r['question_id']}: {r['question']}")
        t = " ".join(r["text"].split())
        print(f"  Answer: {t[:600]}{'…' if len(t) > 600 else ''}")
        for s, a, b in zip(r["subs"], r["v1"], r["mem"]):
            print(f"    [{'P' if s['type'] == 'memory_presence' else 'F'}] expected {str(s['expected']).lower():3}  judge_v1 {a:3}  Memora's {b:7}  {s['text']}")
        print(f"  FAMA: judge_v1 {100 * per_q(r, 'v1')[0]:.0f}, Memora's judge {100 * per_q(r, 'mem')[0]:.0f}")
print(f"\nAnswers and verdicts: {(OUT / 'full_history.jsonl').relative_to(ROOT)}; progress: {(OUT / 'progress.log').relative_to(ROOT)}")
