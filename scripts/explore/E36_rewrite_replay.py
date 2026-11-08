"""E36: replay recall-time rewrites from the full Stage 0 run (E35) with rewrite prompts v4, v5a and v5b.

Question: in E35, rewrite_v4 sometimes wrote the user's latest message in a recalled memory's place,
replacing one event with another of the same kind ("$5.88 on a coffee" became the "$8.18" just
mentioned): 24 of 422 note rewrites and 48 of 442 log rewrites, by the rough classification below. Does
showing the memory's date on its own line, with how long before this conversation it was (v5a, Sam's
idea), or that plus "The memory and the user's latest message may describe different events of the
same kind" (v5b), stop that, without stopping the rewrites that take in a change the user asked for
("used to love cultural ceremonies, no longer keen") and without breaking the rest?
Method: E35's session records give, for every rewrite, the turn, the memory chosen and v4's answer; the
memory's text and date as they stood at that turn are rebuilt from store B's event file (text events
from earlier sessions; under discard, recall changes nothing). Each v4 rewrite is classified, then the
cases are: every "replaced by a different event", every "updated as the user said", and 50 others per
arm, a sample stratified by class (seed 36). Each case is sent again, exactly as recall sends it, with
each prompt version; v4 is answered from the call cache, so it reproduces E35 (checked). Every answer is
classified the same way. Development personas only.
Classification (rough, from words and numbers; read the examples): identical; "replaced by a different
event" when a number of the memory is lost and a number of the user's message appears, or when under
half of the memory's content words stay and more come from the message, and the user has not said that
something changed; "updated as the user said" for the same when they have, in that message ("no
longer", "used to", "completed", "changed", "from X to Y", ...) or earlier in the session, where only a
change of preference counts ("no longer", "moved on", "used to love", ...; the pilot's Ryan Reynolds
case said so a line earlier); otherwise kept (at least 80% of the memory's
content words), kept plus words from the message, or shortened.
Config: MEMRECON_E36_CONFIG (default: the newest full Stage 0 control run). Pilot: MEMRECON_E36_PILOT=N
replays only N cases (not evidence). Answers are saved to results/e36/answers.jsonl.
Limits: one model, one run of each version (calls are cached, so a rerun repeats the answers); the
classification is a heuristic, not the judge; v5b's sentence was written after seeing these cases.
"""
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

from common import ROOT
from memrecon.ingest import load_prompt, render_conversation
from memrecon.llm import client as model_client
from memrecon.memora import dev_personas, sessions as load_sessions
from memrecon.memory import At, MemoryStore
from memrecon.monitor import wait_for_memory
from memrecon.probes import content_words, numbers_in
from memrecon.recall import Discard, Recaller, memory_slots, parse_rewrite_answer

VERSIONS = ["rewrite_v4", "rewrite_v5a", "rewrite_v5b"]
SHORT = {"rewrite_v4": "v4", "rewrite_v5a": "v5a", "rewrite_v5b": "v5b"}
OTHERS_PER_ARM = 50
SEED = 36
PILOT = int(os.environ.get("MEMRECON_E36_PILOT", "0") or 0)
CONTROL = ROOT / "results" / "stage0" / "control"
OUT = ROOT / "results" / "e36"
STOP = {"user", "i", "my"}
CHANGE = re.compile(r"\b(update|updated|change|changed|instead|no longer|anymore|used to|not as keen|shifted|completed|"
                    r"finished|remove|moved on|now prefer|from \S+ to)\b", re.I)
EARLIER = re.compile(r"\b(no longer|anymore|not as keen|shifted|moved on|now prefer|used to (?:really )?(?:like|love|enjoy))\b", re.I)
GROUPS = ["replaced by a different event", "updated as the user said", "other"]
CLASSES = ["identical", "kept", "kept + words from the message", "shortened", "replaced by a different event",
           "updated as the user said", "failed"]


def config_dir():
    name = os.environ.get("MEMRECON_E36_CONFIG")
    if name:
        return CONTROL / name
    full = [p for p in CONTROL.iterdir() if p.is_dir() and (p / "config.json").exists()
            and json.loads((p / "config.json").read_text())["sessions"] == "all"]
    return max(full, key=lambda p: p.stat().st_mtime)


def cw(text):
    return content_words(text) - STOP


def classify(old, new, msg, said):
    if new is None:
        return "failed"
    if new == old:
        return "identical"
    o, n, m = cw(old), cw(new), cw(msg)
    kept = len(o & n) / max(1, len(o))
    from_msg = len((n & m) - o)
    no, nn, nm = numbers_in(old), numbers_in(new), numbers_in(msg)
    swapped = bool(no - nn) and bool((nn & nm) - no)
    if swapped or (kept < 0.5 and from_msg > len(o & n)):
        return "updated as the user said" if CHANGE.search(msg) or EARLIER.search(said) else "replaced by a different event"
    if kept >= 0.8:
        return "kept" if from_msg == 0 else "kept + words from the message"
    return "shortened"


def group_of(cls):
    return cls if cls in GROUPS[:2] else "other"


def short(t, n=230):
    t = " ".join((t or "—").split())
    return t if len(t) <= n else t[:n - 1] + "…"


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


def stratified(cases, k, rng):
    """k cases, allocated across v4 classes in proportion (largest remainder), each drawn at random."""
    by = defaultdict(list)
    for c in cases:
        by[c["e35_class"]].append(c)
    total = len(cases)
    quota = {cls: k * len(xs) / total for cls, xs in by.items()}
    take = {cls: int(q) for cls, q in quota.items()}
    for cls in sorted(quota, key=lambda c: quota[c] - take[c], reverse=True)[: k - sum(take.values())]:
        take[cls] += 1
    out = []
    for cls in sorted(by):
        out += rng.sample(sorted(by[cls], key=lambda c: c["key"]), min(take[cls], len(by[cls])))
    return out


# -- the cases ------------------------------------------------------------------------------------

cfg = config_dir()
config = json.loads((cfg / "config.json").read_text())
print(f"Stage 0 control run: {cfg.name} (rewrite prompt in that run: {config['rewrite_prompt']})")
cases = []
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
                for mid, v4 in t["rewrites"].items():
                    _, old, mdate = [x for x in hist[mid] if x[0] < sid][-1]
                    msg = s["conversation"][t["turn"] - 1]["message"]
                    c = {"key": f"{persona}|{arm}|{sid}|{t['turn']}|{mid}", "persona": persona, "arm": arm, "session": sid,
                         "line": t["turn"], "mid": mid, "old": old, "memory_date": mdate, "date": s.get("date", ""),
                         "message": msg,
                         "said": " ".join(x["message"] for x in s["conversation"][: t["turn"]] if x.get("speaker") == "user_agent"),
                         "conversation": render_conversation({"conversation": s["conversation"][: t["turn"]]})[0],
                         "e35": v4}
                    c["e35_class"] = classify(old, v4, msg, c["said"])
                    cases.append(c)

rng = random.Random(SEED)
chosen = [c for c in cases if group_of(c["e35_class"]) != "other"]
for arm in ("note", "log"):
    chosen += stratified([c for c in cases if c["arm"] == arm and group_of(c["e35_class"]) == "other"], OTHERS_PER_ARM, rng)
if PILOT:
    chosen = rng.sample(chosen, min(PILOT, len(chosen)))
print(f"Rewrites in the run: {len(cases)} (note {sum(c['arm'] == 'note' for c in cases)}, log {sum(c['arm'] == 'log' for c in cases)})")
print("Cases replayed, by arm and v4's class in E35:")
cnt = Counter((c["arm"], group_of(c["e35_class"])) for c in chosen)
for g in GROUPS:
    print(f"  {g:34} note {cnt[('note', g)]:4}  log {cnt[('log', g)]:4}")
print(f"  {'total':32} {len(chosen)}" + ("   (PILOT: not evidence)" if PILOT else ""))
print()

# -- replay ---------------------------------------------------------------------------------------

agent = model_client("agent")
OUT.mkdir(parents=True, exist_ok=True)
answers = {}
start = time.monotonic()
for version in VERSIONS:
    system, template = load_prompt(version)
    t0 = time.monotonic()
    for i, c in enumerate(chosen):
        if i % 25 == 0:
            wait_for_memory(say=print)
        rc = Recaller(agent, MemoryStore(c["persona"], mode=c["arm"]), None, Discard(), persona=c["persona"],
                      rewrite_prompt=version, tag="E36")
        user = template.format(date=c["date"], conversation=c["conversation"], **memory_slots(c["old"], c["memory_date"], c["date"]))
        text, error = rc._ask(system, user, parse_rewrite_answer, At(c["session"], c["line"]), "rewrite")
        answers[(version, c["key"])] = text
        c[SHORT[version]] = text
        c[SHORT[version] + "_class"] = classify(c["old"], text, c["message"], c["said"])
        c[SHORT[version] + "_cached"] = all(x["cached"] for x in rc.calls)
    print(f"{SHORT[version]}: {len(chosen)} cases in {time.monotonic() - t0:.0f} s")
print(f"Total {time.monotonic() - start:.0f} s")
with open(OUT / "answers.jsonl", "w", encoding="utf-8") as f:
    for c in chosen:
        f.write(json.dumps({k: v for k, v in c.items() if k != "conversation"}) + "\n")

same = sum(c["v4"] == c["e35"] for c in chosen)
print(f"v4 replay reproduces E35's rewrite: {same} of {len(chosen)} ({sum(c['v4_cached'] for c in chosen)} answered from the cache)")
print()

# -- results --------------------------------------------------------------------------------------

print("Class of each answer, by the group the case was in under v4 in E35 (rows: version; % of the group)")
for arm in ("note", "log"):
    for g in GROUPS:
        xs = [c for c in chosen if c["arm"] == arm and group_of(c["e35_class"]) == g]
        if not xs:
            continue
        print(f"\n{arm} / {g} ({len(xs)} cases)")
        print(f"  {'':5}" + "".join(f"{k[:14]:>16}" for k in CLASSES))
        for v in SHORT.values():
            k = Counter(c[v + "_class"] for c in xs)
            print(f"  {v:5}" + "".join(f"{k[cls]:>7} {pct(k[cls], len(xs)):>8}" for cls in CLASSES))

print("\nHeadline (all arms): replaced by a different event, as a share of the cases in each group")
for g in GROUPS:
    xs = [c for c in chosen if group_of(c["e35_class"]) == g]
    print(f"  {g:34}" + "".join(f"  {v} {pct(sum(c[v + '_class'] == 'replaced by a different event' for c in xs), len(xs))}"
                                  for v in SHORT.values()))
xs = [c for c in chosen if group_of(c["e35_class"]) == "updated as the user said"]
print(f"  {'user-asked changes still taken in':32}" + "".join(
    f"  {v} {pct(sum(c[v + '_class'] in ('updated as the user said', 'kept + words from the message') for c in xs), len(xs))}"
    for v in SHORT.values()))
xs = [c for c in chosen if group_of(c["e35_class"]) == "other"]
print(f"  {'others identical':32}" + "".join(f"  {v} {pct(sum(c[v + '_class'] == 'identical' for c in xs), len(xs))}" for v in SHORT.values()))
print(f"  {'failed answers (all cases)':32}" + "".join(f"  {v} {sum(c[v + '_class'] == 'failed' for c in chosen):5}" for v in SHORT.values()))


def show(c):
    print(f"--- {c['persona']} {c['arm']} session {c['session']} line {c['line']} (memory from {c['memory_date']}, conversation {c['date']})")
    print(f"  USER:   {short(c['message'])}")
    print(f"  MEMORY: {short(c['old'])}")
    for v in SHORT.values():
        print(f"  {v + ':':7} [{c[v + '_class']}] {short(c[v])}")


print("\n== Every case v4 replaced by a different event ==")
for c in sorted((c for c in chosen if group_of(c["e35_class"]) == "replaced by a different event"), key=lambda c: c["key"]):
    show(c)
print("\n== User-asked changes that v5a or v5b no longer took in ==")
for c in sorted(chosen, key=lambda c: c["key"]):
    if group_of(c["e35_class"]) == "updated as the user said" and any(
            c[v + "_class"] not in ("updated as the user said", "kept + words from the message") for v in ("v5a", "v5b")):
        show(c)
print("\n== Other cases where v5a or v5b replaced the memory by a different event ==")
for c in sorted(chosen, key=lambda c: c["key"]):
    if group_of(c["e35_class"]) == "other" and any(c[v + "_class"] == "replaced by a different event" for v in ("v5a", "v5b")):
        show(c)
