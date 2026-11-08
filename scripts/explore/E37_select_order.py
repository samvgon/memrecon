"""E37: does the order step 1 lists the memories in change which ones the agent chooses, and in what order?

Question (D54, E35): recall's step 1 (`select_v1`) lists the memories retrieval found newest first and
asks which are useful for the user's latest message, most useful first. In E35, when the agent chose
every memory shown, it kept the listed order 84 times and reversed it 22 times. Is that recency (the
newest is usually the most useful) or position (the first listed is chosen first, whatever it is)? And
how much does what is chosen depend on the order at all?
Method: every turn of E35 (the full Stage 0 control run, development personas) where step 1 was shown 2
or 3 memories and answered, asked again exactly as recall asks it, each call on its own (no call sees
another's answer), in these orders:
- newest first, as recall lists them (from the call cache: this is E35's own answer, checked);
- newest first again, with a different request seed (temperature is 0, so this measures how much the
  answer varies with nothing changed; the seed is only part of the request, not of the prompt);
- other orders, drawn at random (seed 37) from those that are not newest first: 3 for a turn with 3
  memories (of 5), 1 for a turn with 2 (reversed, the only one).
Reported: how often the chosen set, the first choice and the whole answer stay the same as newest first;
how often each position is chosen (first, second, third) in each kind of order, and how often the first
choice is the memory listed first; and, for one memory, how often it is chosen when listed first rather
than last (within the same turn, so what it says is held fixed).
Config: MEMRECON_E37_CONFIG (default: the newest full Stage 0 control run). Pilot:
MEMRECON_E37_PILOT=N asks only N turns (not evidence). Answers are saved to results/e37/answers.jsonl.
Limits: one model, temperature 0; one random draw of orders per turn; the turns come from one run, so
the memories shown are those retrieval found there.
"""
import itertools
import json
import os
import random
import time
from collections import Counter, defaultdict

from common import ROOT
from memrecon.ingest import load_prompt, render_conversation
from memrecon.llm import client as model_client
from memrecon.memora import dev_personas, sessions as load_sessions
from memrecon.memory import At, MemoryStore
from memrecon.monitor import wait_for_memory
from memrecon.recall import Discard, Recaller, parse_select_answer

SEED = 37
PILOT = int(os.environ.get("MEMRECON_E37_PILOT", "0") or 0)
CONTROL = ROOT / "results" / "stage0" / "control"
OUT = ROOT / "results" / "e37"
KINDS = ["newest first", "newest first, asked again", "random order"]


def config_dir():
    name = os.environ.get("MEMRECON_E37_CONFIG")
    if name:
        return CONTROL / name
    full = [p for p in CONTROL.iterdir() if p.is_dir() and (p / "config.json").exists()
            and json.loads((p / "config.json").read_text())["sessions"] == "all"]
    return max(full, key=lambda p: p.stat().st_mtime)


def listing(order, mem):
    """Step 1's list, as MemoryStore.present writes it: "[k] (date) text", one per line."""
    lines = []
    for k, mid in enumerate(order, start=1):
        text, date = mem[mid]
        lines.append(f"[{k}] ({date}) {text}" if date else f"[{k}] {text}")
    return "\n".join(lines)


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


# -- the turns ------------------------------------------------------------------------------------

cfg = config_dir()
config = json.loads((cfg / "config.json").read_text())
print(f"Stage 0 control run: {cfg.name} (select prompt in that run: {config['select_prompt']})")
turns = []
for persona in dev_personas():
    conv = {s["session_id"]: s for s in load_sessions(persona)}
    for arm in ("note", "log"):
        run = sorted((cfg / persona / arm).glob("run_*"), key=lambda p: int(p.name[4:]))[-1]
        hist = defaultdict(list)  # memory id -> [(session, text, date)]
        for line in open(run / "B" / "events.jsonl", encoding="utf-8"):
            e = json.loads(line)
            if e.get("text") is not None and e.get("id"):
                hist[e["id"]].append((e["at"]["session_id"], e["text"], e.get("date", "")))
        for f in sorted((run / "sessions").glob("session_*.json")):
            d = json.loads(f.read_text())
            sid = d["session"]
            s = conv[sid]
            for t in d["turns"]:
                if not t["select_ok"] or len(t["shown"]) < 2:
                    continue
                mem = {}
                for mid in t["shown"]:
                    _, text, date = [x for x in hist[mid] if x[0] < sid][-1]
                    mem[mid] = (text, date)
                turns.append({"key": f"{persona}|{arm}|{sid}|{t['turn']}", "persona": persona, "arm": arm, "session": sid,
                              "line": t["turn"], "date": s.get("date", ""), "newest": t["shown"], "e35": t["selected"], "mem": mem,
                              "conversation": render_conversation({"conversation": s["conversation"][: t["turn"]]})[0]})

rng = random.Random(SEED)
for t in turns:
    others = [list(p) for p in itertools.permutations(t["newest"]) if list(p) != t["newest"]]
    t["random"] = rng.sample(others, 3 if len(t["newest"]) == 3 else 1)
if PILOT:
    turns = rng.sample(turns, min(PILOT, len(turns)))
n_calls = sum(2 + len(t["random"]) for t in turns)
print(f"Turns with 2 or 3 memories shown: {len(turns)} "
      f"(2 shown: {sum(len(t['newest']) == 2 for t in turns)}, 3 shown: {sum(len(t['newest']) == 3 for t in turns)}; "
      f"note {sum(t['arm'] == 'note' for t in turns)}, log {sum(t['arm'] == 'log' for t in turns)}); {n_calls} questions"
      + ("   (PILOT: not evidence)" if PILOT else ""))
print()

# -- ask ------------------------------------------------------------------------------------------

agent = model_client("agent")
system, template = load_prompt(config["select_prompt"])
OUT.mkdir(parents=True, exist_ok=True)
start = time.monotonic()
asks = []  # one per question: turn, kind, order, chosen (memory ids, most useful first) or None, cached


def ask(t, order, kind, settings):
    rc = Recaller(agent, MemoryStore(t["persona"], mode=t["arm"]), None, Discard(), persona=t["persona"],
                  select_prompt=config["select_prompt"], tag="E37", call_settings=settings)
    labels = {str(k): mid for k, mid in enumerate(order, start=1)}
    user = template.format(date=t["date"], conversation=t["conversation"], memories=listing(order, t["mem"]))
    chosen, error = rc._ask(system, user, lambda r: parse_select_answer(r, labels), At(t["session"], t["line"]), "select")
    asks.append({"turn": t["key"], "arm": t["arm"], "n": len(order), "kind": kind, "order": order,
                 "chosen": None if chosen is None else [labels[x] for x in chosen], "error": error,
                 "cached": all(x["cached"] for x in rc.calls)})


for i, t in enumerate(turns):
    if i % 25 == 0:
        wait_for_memory(say=print)
    ask(t, t["newest"], "newest first", {})
    ask(t, t["newest"], "newest first, asked again", {"seed": 1})
    for order in t["random"]:
        ask(t, order, "random order", {})
print(f"{len(asks)} questions in {time.monotonic() - start:.0f} s "
      f"({sum(a['cached'] for a in asks)} answered from the cache)")
with open(OUT / "answers.jsonl", "w", encoding="utf-8") as f:
    for a in asks:
        f.write(json.dumps(a) + "\n")

by_turn = defaultdict(lambda: defaultdict(list))
for a in asks:
    by_turn[a["turn"]][a["kind"]].append(a)
e35 = {t["key"]: t["e35"] for t in turns}
base = {k: v["newest first"][0]["chosen"] for k, v in by_turn.items()}
print(f"Newest first reproduces E35's answer: {sum(base[k] == e35[k] for k in base)} of {len(base)}")
print(f"Failed answers: " + ", ".join(f"{k} {sum(a['chosen'] is None for a in asks if a['kind'] == k)}" for k in KINDS))
print()

# -- does the answer stay the same? ---------------------------------------------------------------

print("Compared with newest first (each question; failed ones left out)")
print(f"  {'':34}{'questions':>10}{'same set':>14}{'same first':>14}{'same answer':>14}")
for arm in ("note", "log", "both"):
    for n in (2, 3, "all"):
        for kind in KINDS[1:]:
            pairs = [(base[a["turn"]], a["chosen"]) for a in asks if a["kind"] == kind and a["chosen"] is not None
                     and base[a["turn"]] is not None and arm in (a["arm"], "both") and n in (a["n"], "all")]
            if not pairs:
                continue
            ss = sum(set(b) == set(c) for b, c in pairs)
            sf = sum((b[:1] == c[:1]) for b, c in pairs)
            sa = sum(b == c for b, c in pairs)
            label = f"{arm}, {n} shown, {kind}"
            print(f"  {label:34}{len(pairs):>10}{ss:>6} {pct(ss, len(pairs)):>7}{sf:>6} {pct(sf, len(pairs)):>7}{sa:>6} {pct(sa, len(pairs)):>7}")
print("  (\"same first\" counts two empty answers as the same; \"same answer\" is the same memories in the same order)")
print()

# -- position ---------------------------------------------------------------------------------------

print("How often each position is chosen, and how often the first choice is the memory listed first")
print(f"  {'':30}{'questions':>10}{'1st chosen':>14}{'2nd chosen':>14}{'3rd chosen':>14}{'none chosen':>14}{'first choice = 1st':>20}")
for n in (2, 3):
    for kind in KINDS:
        xs = [a for a in asks if a["kind"] == kind and a["n"] == n and a["chosen"] is not None]
        if not xs:
            continue
        cells = []
        for pos in range(3):
            if pos < n:
                k = sum(a["order"][pos] in a["chosen"] for a in xs)
                cells.append(f"{k:>6} {pct(k, len(xs)):>7}")
            else:
                cells.append(f"{'':>14}")
        none = sum(not a["chosen"] for a in xs)
        some = [a for a in xs if a["chosen"]]
        first = sum(a["chosen"][0] == a["order"][0] for a in some)
        print(f"  {f'{n} shown, {kind}':30}{len(xs):>10}" + "".join(cells) + f"{none:>6} {pct(none, len(xs)):>7}"
              f"{first:>6} of {len(some):<4} {pct(first, len(some)):>7}")
print("  Under newest first, position and recency are the same thing; under random orders they are not.")
print()

print("When it chooses every memory shown (2 or more): does it keep the listed order?")
for kind in KINDS:
    xs = [a for a in asks if a["kind"] == kind and a["chosen"] is not None and len(a["chosen"]) == a["n"]]
    kept = sum(a["chosen"] == a["order"] for a in xs)
    rev = sum(a["chosen"] == a["order"][::-1] for a in xs)
    print(f"  {kind:30} {len(xs):>4} answers: listed order {kept:>4} {pct(kept, len(xs))}, reversed {rev:>4} {pct(rev, len(xs))}, "
          f"other {len(xs) - kept - rev:>4}")
print()

print("The same memory listed first or last in the same turn (random orders and newest first; what it says held fixed)")
first_last = Counter()
for k, kinds in by_turn.items():
    qs = [a for kind in ("newest first", "random order") for a in kinds[kind] if a["chosen"] is not None]
    for mid in qs[0]["order"] if qs else []:
        at_first = [mid in a["chosen"] for a in qs if a["order"][0] == mid]
        at_last = [mid in a["chosen"] for a in qs if a["order"][-1] == mid]
        if at_first and at_last:
            f_rate, l_rate = sum(at_first) / len(at_first), sum(at_last) / len(at_last)
            first_last["pairs"] += 1
            first_last["more when first"] += f_rate > l_rate
            first_last["more when last"] += f_rate < l_rate
            first_last["same"] += f_rate == l_rate
print(f"  memories seen both first and last in their turn: {first_last['pairs']}; chosen more often when first "
      f"{first_last['more when first']}, when last {first_last['more when last']}, the same {first_last['same']}")
b, c = first_last["more when first"], first_last["more when last"]
from math import comb
p = 1.0 if b + c == 0 else min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c))
print(f"  sign test, first against last: p = {p:.3f}")
print()


def short(text, n=150):
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


print("== Examples: turns where a random order changed the chosen set (up to 15) ==")
shown = 0
for t in sorted(turns, key=lambda t: t["key"]):
    kinds = by_turn[t["key"]]
    b = kinds["newest first"][0]["chosen"]
    changed = [a for a in kinds["random order"] if a["chosen"] is not None and b is not None and set(a["chosen"]) != set(b)]
    if not changed or shown >= 15:
        continue
    shown += 1
    msg = t["conversation"].rsplit("\n", 1)[-1]
    print(f"--- {t['persona']} {t['arm']} session {t['session']} line {t['line']}")
    print(f"  {short(msg, 200)}")
    names = {mid: chr(ord("A") + i) for i, mid in enumerate(t["newest"])}
    for mid in t["newest"]:
        print(f"  {names[mid]} ({t['mem'][mid][1]}) {short(t['mem'][mid][0])}")
    for a in kinds["newest first"] + kinds["newest first, asked again"] + kinds["random order"]:
        if a["chosen"] is None:
            continue
        print(f"  {a['kind']:27} shown {''.join(names[m] for m in a['order'])} -> chose {''.join(names[m] for m in a['chosen']) or 'none'}")
