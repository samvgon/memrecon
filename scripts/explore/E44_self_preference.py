"""E44: does each judge favour answers written by its own model family (self-preference)?

Question: Memora scores answers with LLM judges, and our judge (Gemma 4, D16, D61) was chosen from a
different family than the agent (Qwen) partly so it would not favour the agent's answers. Does an LLM
judge favour answers from its own family, and how much would FAMA move if the judge shared the agent's?
Method (development personas only, the frozen setup): the agent's answers from the Stage 0 answering run
(`results/stage0/answers/<config>/`, Qwen, `answer_v1`) and a second set written by Gemma from exactly
the same prompts (the same 50 memories, shown the same way), so the two sets differ only in who wrote
them. Every sub-question of every answer is judged by both models with `judge_v1`: a 2 x 2 of writer by
judge. Self-preference is the difference in differences of the pass rate:
  (Qwen judge - Gemma judge on Qwen's answers) - (Qwen judge - Gemma judge on Gemma's answers),
positive when each judge is kinder to its own family's answers; this removes each judge's general
leniency and each writer's general quality. A bootstrap over questions (seed 44) gives its interval.
Also: FAMA, MPA and FAA in each cell, agreement between the judges, and both judges against Sam's hand
labels (the 70 judge-check items, all Qwen answers; the 50 check items are reported separately because
judge_v1 was adjusted on the 20 tune items with Gemma).
Limits: one model per family and one prompt; 90 answers per writer, so only a fairly large effect would
show; Gemma answers from memories that Qwen wrote (ingestion is Qwen's), which is the realistic case for a
judge of a memory agent; "pass" uses Memora's expected answer, not a human label, except in the
hand-label section.
Pausing: every call is cached (llm.py), so stop with Ctrl-C at any time and rerun to resume; finished
calls replay in seconds. Work is ordered question by question with all four cells together, so a partial
run is still balanced. `--cached-only` (or MEMRECON_E44_CACHED_ONLY=1) makes no model calls and reports on
what is done so far. `MEMRECON_E44_PILOT=N` takes only N questions per persona and arm (not evidence).
Needs the agent (Qwen) and judge (Gemma) servers; not the embedding server (the memories are reused).
Config: MEMRECON_E44_CONFIG (default: the newest full control run with the current prompts and answers).
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
from memrecon.judge import JUDGE_PROMPT, Judge, cohen_kappa
from memrecon.llm import cache_key, client as model_client, endpoint
from memrecon.memora import dev_personas
from memrecon.monitor import wait_for_memory

CACHED_ONLY = "--cached-only" in sys.argv or os.environ.get("MEMRECON_E44_CACHED_ONLY") == "1"
PILOT = int(os.environ.get("MEMRECON_E44_PILOT", "0") or 0)
ANSWERS = ROOT / "results" / "stage0" / "answers"
OUT = ROOT / "results" / "e44"
LABELS = ROOT / "notes" / "labels"
WRITERS = ("qwen", "gemma")   # who wrote the answer
JUDGES = ("qwen", "gemma")    # who judged it
ROLE = {"qwen": "agent", "gemma": "judge"}


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
    name = os.environ.get("MEMRECON_E44_CONFIG")
    if name:
        return ANSWERS / name
    done = [p for p in ANSWERS.iterdir() if p.is_dir() and all((p / q / a / "answers.jsonl").exists()
                                                               for q in dev_personas() for a in ("note", "log"))]
    if not done:
        sys.exit(f"No complete Stage 0 answering run in {ANSWERS} (run scripts/run_stage0_answers.py first).")
    return max(done, key=lambda p: p.stat().st_mtime)


def memories_text(row):
    """The memories exactly as present() shows them: '[k] (date) text', newest first."""
    lines = [f"[{k}] ({d}) {t}" if d else f"[{k}] {t}" for k, (t, d) in enumerate(zip(row["shown_text"], row["shown_dates"]), start=1)]
    return "\n".join(lines) or "(none)"


def boot(values_by_q, reps=2000, seed=44):
    """95% bootstrap interval of a statistic computed from per-question tuples (resampling questions)."""
    rng = random.Random(seed)
    keys = list(values_by_q)
    stats = []
    for _ in range(reps):
        sample = [values_by_q[rng.choice(keys)] for _ in keys]
        stats.append(did(sample))
    stats.sort()
    return stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


def did(per_q):
    """Difference in differences of the pass rate from per-question (pass, total) per cell."""
    tot = defaultdict(lambda: [0, 0])
    for cells in per_q:
        for cell, (p, n) in cells.items():
            tot[cell][0] += p
            tot[cell][1] += n
    r = {c: p / n if n else float("nan") for c, (p, n) in tot.items()}
    return (r[("qwen", "qwen")] - r[("qwen", "gemma")]) - (r[("gemma", "qwen")] - r[("gemma", "gemma")])


cfg = config_dir()
print(f"Answers: {cfg.relative_to(ROOT)} (Qwen, {ANSWER_PROMPT}); judge prompt {JUDGE_PROMPT}")
print(f"Qwen = {endpoint('agent').model}; Gemma = {endpoint('judge').model}")
print("Mode: " + ("CACHED ONLY (no model calls; partial results)" if CACHED_ONLY else "full") + (f"; PILOT {PILOT} questions per unit (not evidence)" if PILOT else ""))
print()

client = {w: Gate(model_client(ROLE[w])) for w in WRITERS}
judge = {j: Judge(client[j], tag="E44") for j in JUDGES}
system, template = load_prompt(ANSWER_PROMPT)
OUT.mkdir(parents=True, exist_ok=True)
progress = open(OUT / "progress.log", "a", encoding="utf-8")

rows = []          # one per sub-question: writer x judge verdicts
reproduced = [0, 0]  # Qwen answers rebuilt from the cache: same text, total
gemma_failed = 0
t0 = time.monotonic()
for persona in dev_personas():
    for arm in ("note", "log"):
        answered = [json.loads(x) for x in (cfg / persona / arm / "answers.jsonl").read_text().splitlines() if x.strip()]
        if PILOT:
            answered = answered[:PILOT]
        for n, row in enumerate(answered, start=1):
            if n % 5 == 1:
                wait_for_memory(say=print)
            user = template.format(date=row.get("date", ""), question=row["question"], memories=memories_text(row))
            text = {"qwen": row["answer"]}
            # check that the prompt is rebuilt exactly: Qwen's answer should come back from the cache
            try:
                again, _, _ = Answerer(client["qwen"], None, None, persona=persona, tag="E44")._ask(system, user, row["question_id"])
                reproduced[0] += again == row["answer"]
            except NotCached:
                pass
            reproduced[1] += 1
            try:
                text["gemma"], err, _ = Answerer(client["gemma"], None, None, persona=persona, tag="E44")._ask(system, user, row["question_id"])
            except NotCached:
                continue
            gemma_failed += text["gemma"] is None
            for sub in row["sub_questions"]:
                v = {}
                for w in WRITERS:
                    for j in JUDGES:
                        if text[w] is None:
                            v[(w, j)] = None
                            continue
                        try:
                            v[(w, j)] = judge[j].judge(text[w], sub["text"], {"persona": persona, "arm": arm, "sub_id": sub["id"], "writer": w})["answer"]
                        except NotCached:
                            v[(w, j)] = None
                rows.append({"persona": persona, "arm": arm, "question_id": row["question_id"], "category": row["category"],
                             "sub_id": sub["id"], "type": sub["type"], "expected": str(sub["expected"]).lower(), "sub": sub["text"],
                             "text": text, "v": v})
            progress.write(f"{time.strftime('%H:%M:%S')} {persona} {arm} {n}/{len(answered)} "
                           f"sent qwen {client['qwen'].sent} gemma {client['gemma'].sent}\n")
            progress.flush()
progress.close()

complete = [r for r in rows if all(x is not None for x in r["v"].values())]
print(f"Sub-questions with all four verdicts: {len(complete)} of {len(rows)} reached "
      f"({time.monotonic() - t0:.0f} s; calls sent: Qwen {client['qwen'].sent}, Gemma {client['gemma'].sent}"
      + (f"; not yet cached: {client['qwen'].skipped + client['gemma'].skipped}" if CACHED_ONLY else "") + ")")
print(f"Qwen's answers rebuilt from the cache with the same text: {reproduced[0]} of {reproduced[1]} (checks the prompts are identical)")
print(f"Gemma's answers that failed: {gemma_failed}")
if not complete:
    sys.exit(0)

# -- the 2 x 2 ---------------------------------------------------------------------------------------
print("\n== Pass rate (judge's answer equals Memora's expected answer), writer by judge ==")
print(f"{'':16}{'Qwen judges':>14}{'Gemma judges':>14}{'difference':>12}")
rate = {}
for w in WRITERS:
    for j in JUDGES:
        rate[(w, j)] = sum(r["v"][(w, j)] == r["expected"] for r in complete) / len(complete)
    print(f"{w.capitalize() + ' wrote':16}{100 * rate[(w, 'qwen')]:13.1f}%{100 * rate[(w, 'gemma')]:13.1f}%{100 * (rate[(w, 'qwen')] - rate[(w, 'gemma')]):+11.1f}")
by_q = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for r in complete:
    for c, x in r["v"].items():
        by_q[(r["persona"], r["arm"], r["question_id"])][c][0] += x == r["expected"]
        by_q[(r["persona"], r["arm"], r["question_id"])][c][1] += 1
per_q = {k: {c: tuple(pn) for c, pn in cells.items()} for k, cells in by_q.items()}
d = did(per_q.values())
lo, hi = boot(per_q)
print(f"\nSelf-preference (difference in differences): {100 * d:+.1f} points; 95% bootstrap interval "
      f"[{100 * lo:+.1f}, {100 * hi:+.1f}] over {len(per_q)} questions")
print("  positive: each judge is kinder to its own family's answers; an interval covering 0 means no clear effect")

print("\n== By sub-question type (pass rate) ==")
for t in sorted({r["type"] for r in complete}):
    sub = [r for r in complete if r["type"] == t]
    cells = "  ".join(f"{w}/{j} {100 * sum(r['v'][(w, j)] == r['expected'] for r in sub) / len(sub):5.1f}%" for w in WRITERS for j in JUDGES)
    print(f"  {t:20} n={len(sub):4}  {cells}   (writer/judge)")

print("\n== FAMA, MPA, FAA in each cell (questions with all their sub-questions judged) ==")
qs = defaultdict(list)
for r in complete:
    qs[(r["persona"], r["arm"], r["question_id"])].append(r)
for arm in ("note", "log", "both"):
    for w in WRITERS:
        line = []
        for j in JUDGES:
            f, mp, fa = [], [], []
            for (p, a, q), subs in qs.items():
                if arm != "both" and a != arm:
                    continue
                pres = [s for s in subs if s["type"] == "memory_presence"]
                forg = [s for s in subs if s["type"] != "memory_presence"]
                pc = sum(s["v"][(w, j)] == s["expected"] for s in pres)
                fc = sum(s["v"][(w, j)] == s["expected"] for s in forg)
                f.append(fama_score(pc, len(pres), fc, len(forg)))
                if pres:
                    mp.append(pc / len(pres))
                if forg:
                    fa.append(fc / len(forg))
            line.append(f"{j} judges: FAMA {table_cell(f):5.1f}  MPA {100 * sum(mp) / max(1, len(mp)):5.1f}  FAA {100 * sum(fa) / max(1, len(fa)):5.1f}")
        print(f"  {arm:5} {w:5} wrote | " + " | ".join(line))

print("\n== Agreement between the judges ==")
for w in WRITERS:
    a = [r["v"][(w, "qwen")] for r in complete]
    b = [r["v"][(w, "gemma")] for r in complete]
    agree = sum(x == y for x, y in zip(a, b))
    print(f"  on {w}'s answers: {agree} of {len(a)} ({100 * agree / len(a):.1f}%), kappa {cohen_kappa(a, b):.3f}; "
          f"Qwen says yes {a.count('yes')}, Gemma says yes {b.count('yes')}")

# -- against Sam's hand labels (Qwen's answers only) -------------------------------------------------
print("\n== Both judges against Sam's hand labels (70 judge-check items; Qwen's answers) ==")
sample = {json.loads(x)["item"]: json.loads(x) for x in (LABELS / "2026-11-08-judge-sample.jsonl").read_text().splitlines() if x.strip()}
labels = {json.loads(x)["item"]: json.loads(x) for x in (LABELS / "2026-11-08-judge-labels-resolved.jsonl").read_text().splitlines() if x.strip()}
hl = defaultdict(list)
for k in sorted(sample):
    s = sample[k]
    for j in JUDGES:
        try:
            ans = judge[j].judge(s["answer"], s["sub_question"], {"item": k, "sub_id": s["sub_id"], "persona": s["persona"], "arm": s["arm"]})["answer"]
        except NotCached:
            ans = None
        hl[(s["set"], j)].append((labels[k]["label"], ans))
for st in ("check", "tune"):
    for j in JUDGES:
        pairs = [(l, a) for l, a in hl[(st, j)] if a is not None]
        if pairs:
            agree = sum(l == a for l, a in pairs)
            print(f"  {st:5} set, {j:5} judges: {agree} of {len(pairs)} agree with Sam ({100 * agree / len(pairs):.1f}%), "
                  f"kappa {cohen_kappa([l for l, _ in pairs], [a for _, a in pairs]):.3f}"
                  + ("   (judge_v1 was adjusted on these with Gemma)" if st == "tune" and j == "gemma" else ""))

# -- examples ----------------------------------------------------------------------------------------
print("\n== Examples where the judges split, each siding with its own family ==")
own = [r for r in complete if r["v"][("qwen", "qwen")] == r["expected"] != r["v"][("qwen", "gemma")]
       or r["v"][("gemma", "gemma")] == r["expected"] != r["v"][("gemma", "qwen")]]
for r in random.Random(44).sample(own, min(8, len(own))):
    w = "qwen" if r["v"][("qwen", "qwen")] == r["expected"] != r["v"][("qwen", "gemma")] else "gemma"
    print(f"--- {r['persona']} {r['arm']} {r['sub_id']} ({r['type']}, expected {r['expected']}); {w} wrote; "
          f"Qwen judge {r['v'][(w, 'qwen')]}, Gemma judge {r['v'][(w, 'gemma')]}")
    print(f"  Q: {r['sub']}")
    t = " ".join((r["text"][w] or "").split())
    print(f"  Answer: {t[:500]}{'…' if len(t) > 500 else ''}")

with open(OUT / "verdicts.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({**r, "v": {f"{w}|{j}": x for (w, j), x in r["v"].items()}}) + "\n")
print(f"\nVerdicts: {(OUT / 'verdicts.jsonl').relative_to(ROOT)}; progress: {(OUT / 'progress.log').relative_to(ROOT)}")
