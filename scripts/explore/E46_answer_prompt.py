"""E46: how much do the answer prompt and the order of memories in it move FAMA?

Question: our answer prompt (`answer_v1`, D60) shows the agent its memories newest first, each with its
date, and says the newer memory is what is true now. Memora's answering prompt shows them by relevance,
with relevance scores, no dates and no such rule. FAMA's forgetting half asks the answer not to present
an outdated fact as current, so how memories are shown could matter as much as what is in them.
Method (development personas only): Stage 0's answers (Qwen, `answer_v1`, the same top-50 memories) are
the baseline. Qwen answers the same 90 questions again from the same 50 memories under each condition,
and Gemma judges every answer with `judge_v1` (the validated judge, D61):
  memora     Memora's answering prompt, verbatim (evals/agent_eval/memory_to_answer.py, `generate_answer`,
             a6493188): memories by relevance as "i. text (relevance: 0.xx)", no dates, no newer-wins rule,
             the reply taken as the answer (no answer block). Ours vs theirs as a package.
  relevance  answer_v1 with the memories ordered by relevance instead of newest first (dates and the
             newer-wins rule kept; "newest first" in the prompt changed to "most relevant first"). Order alone.
  nodates    answer_v1 with the dates hidden (newest-first order and the newer-wins rule kept; the sentence
             saying each memory has a date removed). Does the model use the dates, or the position?
  memora_dates  Memora's answering prompt with one change: each memory line shows its date,
             "i. (date) text (relevance: 0.xx)"; still by relevance, no newer-wins rule, prompt text verbatim.
             Does adding the dates alone recover what Memora's prompt loses? (Added Nov 12 after E46's first
             three conditions; they replay from the cache.)
The prompt changes for relevance and nodates are made by exact replacements in answer_v1, listed below and
checked to apply once each. Relevance is the retrieval similarity the Stage 0 answers recorded.
Conditions run one at a time, in the order above (or MEMRECON_E46_CONDITIONS, comma-separated), and each
condition's full report is printed, and saved to results/e46/<condition>-report.txt, as soon as it is done;
a comparison of all finished conditions comes last.
Not the same as Memora's setup: Qwen at temperature 0 with thinking off (Memora: gpt-4o-mini at 0.7, at
most 500 tokens); our retrieval's top 50, not Memora's memory systems'.
Pausing: every call is cached, so Ctrl-C at any time and rerun to resume (finished conditions replay in
seconds). `--cached-only` (or MEMRECON_E46_CACHED_ONLY=1) makes no calls and reports on what is done.
`MEMRECON_E46_PILOT=N` takes only N questions per persona and arm (not evidence).
Needs the agent (Qwen) and judge (Gemma) servers; not the embedding server.
"""
import json
import os
import random
import sys
import time
from collections import defaultdict

from common import ROOT
from memrecon.answer import ANSWER_PROMPT, Answerer
from memrecon.fama import fama_score, table_cell
from memrecon.ingest import load_prompt
from memrecon.judge import JUDGE_PROMPT, Judge
from memrecon.llm import ModelCallError, OutputTruncated, cache_key, client as model_client, endpoint
from memrecon.memora import dev_personas
from memrecon.monitor import wait_for_memory

sys.stdout.reconfigure(line_buffering=True)
CACHED_ONLY = "--cached-only" in sys.argv or os.environ.get("MEMRECON_E46_CACHED_ONLY") == "1"
PILOT = int(os.environ.get("MEMRECON_E46_PILOT", "0") or 0)
CONDITIONS = [c.strip() for c in os.environ.get("MEMRECON_E46_CONDITIONS", "memora,relevance,nodates,memora_dates").split(",") if c.strip()]
ANSWERS = ROOT / "results" / "stage0" / "answers"
OUT = ROOT / "results" / "e46"

# Memora's answering prompt, verbatim (memory_to_answer.py, generate_answer, a6493188).
MEMORA_SYSTEM = (
    "You are a helpful AI assistant with access to a user's personal memory system.\n        \nYour task is to "
    "answer questions based ONLY on the user's stored memories and preferences.\n\nGuidelines:\n1. Use ONLY the "
    "information from the provided memories\n2. Be specific and reference actual items/preferences from memory\n"
    "3. If the question is about recommendations, suggest based on similar items in memory\n4. Be conversational "
    "and helpful\n5. Don't make up information not in the memories"
)
MEMORA_USER = ("User's Question: {question}\n\nUser's Relevant Memories:\n{memory_text}\n\n"
               "Please provide a helpful answer based on these memories.")

# Exact changes to answer_v1 (system, user template) for the two variants.
EDITS = {
    "relevance": [("system", "newest first, each with its date.", "most relevant first, each with its date."),
                  ("user", "Memories, newest first:", "Memories, most relevant first:")],
    "nodates": [("system", "Each memory has the date of the conversation it came from. When memories disagree, "
                           "the newer one is what is true now.", "When memories disagree, the newer one is what is true now."),
                ("system", "newest first, each with its date.", "newest first.")],
}
for c in CONDITIONS:
    if c not in ("memora", "relevance", "nodates", "memora_dates"):
        sys.exit(f"Unknown condition {c!r}: use memora, relevance, nodates, memora_dates.")


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


def config_dir():
    name = os.environ.get("MEMRECON_E46_CONFIG")
    if name:
        return ANSWERS / name
    done = [p for p in ANSWERS.iterdir() if p.is_dir() and all((p / q / a / "answers.jsonl").exists()
                                                               for q in dev_personas() for a in ("note", "log"))]
    if not done:
        sys.exit(f"No complete Stage 0 answering run in {ANSWERS}.")
    return max(done, key=lambda p: p.stat().st_mtime)


def edited(cond):
    system, user = load_prompt(ANSWER_PROMPT)
    for part, old, new in EDITS[cond]:
        text = system if part == "system" else user
        if text.count(old) != 1:
            sys.exit(f"{cond}: {old!r} should appear once in {ANSWER_PROMPT}'s {part}, found {text.count(old)}.")
        text = text.replace(old, new)
        system, user = (text, user) if part == "system" else (system, text)
    return system, user


def by_relevance(row):
    """(text, date, score) of the memories shown, most similar first (ties: the newer first, as shown)."""
    items = [(t, d, row["scores"].get(m, 0.0), n) for n, (m, t, d) in enumerate(zip(row["shown"], row["shown_text"], row["shown_dates"]))]
    return [(t, d, s) for t, d, s, _ in sorted(items, key=lambda x: (-x[2], x[3]))]


def ask_memora(c, row, persona, dates=False):
    mem = "\n".join(f"{i}. ({d}) {t} (relevance: {s:.2f})" if dates and d else f"{i}. {t} (relevance: {s:.2f})"
                    for i, (t, d, s) in enumerate(by_relevance(row), start=1))
    messages = [{"role": "system", "content": MEMORA_SYSTEM},
                {"role": "user", "content": MEMORA_USER.replace("{question}", row["question"]).replace("{memory_text}", mem)}]
    limit = 1024
    for attempt in range(1, 4):
        try:
            result = c.chat(messages, max_tokens=limit, tag="E46",
                            meta={"step": "answer-memora" + ("-dates" if dates else ""), "persona": persona, "question": row["question_id"], "attempt": attempt})
        except OutputTruncated:
            limit *= 2
            continue
        except ModelCallError as e:
            if "reasoning" in str(e) or "unreadable" in str(e):
                continue
            return None
        return (result.content or "").strip() or None
    return None


def answer(cond, row, persona):
    if cond in ("memora", "memora_dates"):
        return ask_memora(writer, row, persona, dates=cond == "memora_dates")
    system, template = PROMPTS[cond]
    if cond == "relevance":
        mem = "\n".join(f"[{k}] ({d}) {t}" if d else f"[{k}] {t}" for k, (t, d, _) in enumerate(by_relevance(row), start=1))
    else:  # nodates: the order shown, no dates
        mem = "\n".join(f"[{k}] {t}" for k, t in enumerate(row["shown_text"], start=1))
    user = template.format(date=row.get("date", ""), question=row["question"], memories=mem or "(none)")
    return Answerer(writer, None, None, persona=persona, tag="E46")._ask(system, user, row["question_id"])[0]


def verdicts(text, row, persona, arm, who):
    out = []
    for sub in row["sub_questions"]:
        v = judge.judge(text, sub["text"], {"persona": persona, "arm": arm, "sub_id": sub["id"], "writer": who})["answer"]
        out.append(v)
    return out


def scores(rows, key):
    """FAMA, MPA, FAA (x100) over questions, from per-sub-question verdicts under `key`."""
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


def boot_diff(a, b, reps=2000, seed=46):
    """Mean of b - a (x100) over paired questions, with a 95% bootstrap interval."""
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    stats = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(reps))
    return 100 * sum(d) / len(d), 100 * stats[int(0.025 * reps)], 100 * stats[int(0.975 * reps) - 1]


def report(cond, rows):
    lines = [f"== Condition: {cond} (against Stage 0's answer_v1; Qwen writes, Gemma judges with {JUDGE_PROMPT}) =="]
    lines.append(f"Questions answered and judged: {len(rows)}; answers that failed: {sum(r['cond_text'] is None for r in rows)}; "
                 f"answers identical to Stage 0's: {sum(r['cond_text'] == r['base_text'] for r in rows)}")
    ok = [r for r in rows if r["cond_text"] is not None and None not in r["cond_v"] and None not in r["base_v"]]
    lines.append(f"{'':12}{'':10}{'FAMA':>8}{'MPA':>8}{'FAA':>8}   FAMA difference, 95% interval")
    for arm in ("note", "log", "both"):
        sub = [r for r in ok if arm == "both" or r["arm"] == arm]
        if not sub:
            continue
        b, c = scores(sub, "base_v"), scores(sub, "cond_v")
        d, lo, hi = boot_diff(b[3], c[3])
        lines.append(f"  {arm:10}{'answer_v1':10}{b[0]:8.1f}{b[1]:8.1f}{b[2]:8.1f}")
        lines.append(f"  {'':10}{cond:10}{c[0]:8.1f}{c[1]:8.1f}{c[2]:8.1f}   {d:+.1f} [{lo:+.1f}, {hi:+.1f}]  (n={len(sub)})")
    lines.append("By question category (both arms), FAMA answer_v1 -> " + cond + ":")
    for cat in sorted({r["category"] for r in ok}):
        sub = [r for r in ok if r["category"] == cat]
        lines.append(f"  {cat:14} n={len(sub):3}  {scores(sub, 'base_v')[0]:5.1f} -> {scores(sub, 'cond_v')[0]:5.1f}")
    flips = defaultdict(int)
    for r in ok:
        for s, a, b in zip(r["subs"], r["base_v"], r["cond_v"]):
            e = str(s["expected"]).lower()
            if (a == e) != (b == e):
                flips[(s["type"], "now passes" if b == e else "now fails")] += 1
    lines.append("Sub-questions whose verdict changes: " + (", ".join(f"{t} {w} {n}" for (t, w), n in sorted(flips.items())) or "none"))
    ex = [(r, s, a, b) for r in ok for s, a, b in zip(r["subs"], r["base_v"], r["cond_v"])
          if s["type"] != "memory_presence" and a != b]
    if ex:
        lines.append("Examples (forgetting sub-questions whose verdict changes):")
        for r, s, a, b in random.Random(46).sample(ex, min(3, len(ex))):
            lines.append(f"--- {r['persona']} {r['arm']} {s['id']} (expected {str(s['expected']).lower()}): answer_v1 {a}, {cond} {b}")
            lines.append(f"  Q: {s['text']}")
            for name, t in (("answer_v1", r["base_text"]), (cond, r["cond_text"])):
                t = " ".join((t or "").split())
                lines.append(f"  {name}: {t[:350]}{'…' if len(t) > 350 else ''}")
    return "\n".join(lines)


cfg = config_dir()
print(f"Answers: {cfg.relative_to(ROOT)} (Stage 0, Qwen, {ANSWER_PROMPT}); judge Gemma with {JUDGE_PROMPT}")
print(f"Qwen = {endpoint('agent').model}; Gemma = {endpoint('judge').model}")
print("Conditions, in order: " + ", ".join(CONDITIONS))
print("Mode: " + ("CACHED ONLY (no model calls; partial results)" if CACHED_ONLY else "full")
      + (f"; PILOT {PILOT} questions per unit (not evidence)" if PILOT else ""))
for cond in CONDITIONS:
    if cond in EDITS:
        print(f"  {cond}: " + "; ".join(f"{part}: {old!r} -> {new!r}" for part, old, new in EDITS[cond]))
print()

writer = Gate(model_client("agent"))
judge = Judge(Gate(model_client("judge")), tag="E46")
PROMPTS = {c: edited(c) for c in CONDITIONS if c in EDITS}
OUT.mkdir(parents=True, exist_ok=True)
progress = open(OUT / "progress.log", "a", encoding="utf-8")

units = []
for persona in dev_personas():
    for arm in ("note", "log"):
        rows = [json.loads(x) for x in (cfg / persona / arm / "answers.jsonl").read_text().splitlines() if x.strip()]
        units.append((persona, arm, rows[:PILOT] if PILOT else rows))

done = {}
for cond in CONDITIONS:
    t0, sent0 = time.monotonic(), (writer.sent, judge.client.sent)
    out = []
    for persona, arm, rows in units:
        for n, row in enumerate(rows, start=1):
            if n % 5 == 1:
                wait_for_memory(say=print)
            try:
                base_v = verdicts(row["answer"], row, persona, arm, "qwen") if row["answer"] else [None] * len(row["sub_questions"])
                text = answer(cond, row, persona)
                cond_v = verdicts(text, row, persona, arm, f"qwen-{cond}") if text else [None] * len(row["sub_questions"])
            except NotCached:
                continue
            out.append({"persona": persona, "arm": arm, "question_id": row["question_id"], "category": row["category"],
                        "subs": row["sub_questions"], "base_text": row["answer"], "cond_text": text, "base_v": base_v, "cond_v": cond_v})
            progress.write(f"{time.strftime('%H:%M:%S')} {cond} {persona} {arm} {n}/{len(rows)} "
                           f"sent agent {writer.sent} judge {judge.client.sent}\n")
            progress.flush()
    done[cond] = out
    with open(OUT / f"{cond}.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    text = report(cond, out) if out else f"== Condition: {cond}: nothing answered yet =="
    text += (f"\n({time.monotonic() - t0:.0f} s; calls sent: agent {writer.sent - sent0[0]}, judge {judge.client.sent - sent0[1]}"
             + (f"; not yet cached: {writer.skipped + judge.client.skipped}" if CACHED_ONLY else "") + ")")
    (OUT / f"{cond}-report.txt").write_text(text + "\n", encoding="utf-8")
    print(text + "\n")
progress.close()

# -- all finished conditions side by side -----------------------------------------------------------
print("== All conditions (questions every condition answered; Gemma judges) ==")
keys = None
for cond, rows in done.items():
    ok = {(r["persona"], r["arm"], r["question_id"]) for r in rows if r["cond_text"] is not None and None not in r["cond_v"] and None not in r["base_v"]}
    keys = ok if keys is None else keys & ok
if not keys:
    sys.exit(0)
first = next(iter(done.values()))
base_rows = [r for r in first if (r["persona"], r["arm"], r["question_id"]) in keys]
print(f"{len(keys)} questions")
for arm in ("note", "log", "both"):
    b = scores([r for r in base_rows if arm == "both" or r["arm"] == arm], "base_v")
    print(f"  {arm:5} {'answer_v1':10} FAMA {b[0]:5.1f}  MPA {b[1]:5.1f}  FAA {b[2]:5.1f}")
    for cond, rows in done.items():
        sub = [r for r in rows if (r["persona"], r["arm"], r["question_id"]) in keys and (arm == "both" or r["arm"] == arm)]
        c = scores(sub, "cond_v")
        print(f"  {'':5} {cond:10} FAMA {c[0]:5.1f}  MPA {c[1]:5.1f}  FAA {c[2]:5.1f}")

# -- the dates alone: Memora's prompt with and without them -----------------------------------------
if done.get("memora") and done.get("memora_dates"):
    print("\n== Memora's prompt: with dates against without (paired by question; Gemma judges with judge_v1) ==")
    a = {(r["persona"], r["arm"], r["question_id"]): r for r in done["memora"]
         if r["cond_text"] is not None and None not in r["cond_v"]}
    b = {(r["persona"], r["arm"], r["question_id"]): r for r in done["memora_dates"]
         if r["cond_text"] is not None and None not in r["cond_v"]}
    both = sorted(set(a) & set(b))
    for arm in ("note", "log", "both"):
        ks = [k for k in both if arm == "both" or k[1] == arm]
        if not ks:
            continue
        x, y = scores([a[k] for k in ks], "cond_v"), scores([b[k] for k in ks], "cond_v")
        base = scores([a[k] for k in ks], "base_v")
        d, lo, hi = boot_diff(x[3], y[3])
        print(f"  {arm:5} n={len(ks):3}  answer_v1 {base[0]:5.1f} | memora {x[0]:5.1f} (FAA {x[2]:5.1f}) -> memora_dates {y[0]:5.1f} "
              f"(FAA {y[2]:5.1f}): {d:+.1f} [{lo:+.1f}, {hi:+.1f}]")
print(f"\nPer-condition reports: {(OUT).relative_to(ROOT)}/<condition>-report.txt; progress: {(OUT / 'progress.log').relative_to(ROOT)}")
