"""E16: how well does ingestion keep the facts, and how well does retrieval then find them?

Question (D35, D36, D14): on the development personas, (1) does note-arm ingestion keep each fact's
checkable details, invent values, store things from small talk, or fail; how long are its answers
(to set the output limit, provisionally 2,048 tokens) and how long does it take; (2) for each kind of
memory (oracle: flagged turns verbatim; log: the words the model chose to keep, verbatim; note: model
notes), how often does retrieval find a fact when the user is clearly talking about it, and how often
do small-talk turns recall something, at a range of thresholds?
Method: development personas only (notes/split.json). Each persona's sessions are replayed in file
order (A6) three times, once per arm, with the real ingestion code (src/memrecon/ingest.py) and the
agent model: note arm (prompt ingest_note_v1, related memories from retrieval, top 3 per user turn,
up to 10, no threshold), log arm (prompt ingest_log_v1), oracle (flagged turns of add and update
sessions, no model). All calls are cached, so a rerun or a resumed run reuses them.
  Detail survival: for each fact (an add or update session with flagged turns), the candidate elements
  of its flagged turns (E14's patterns) are looked for in the memories created when that session was
  ingested: numbers by their digits, other elements case-insensitively.
  Invented numbers: digit groups in a note that appear nowhere in the session's text (calendar dates
  YYYY-MM-DD are left out, since notes are asked to compute them).
  Retrieval: just before a later update or delete session of the same record (A10) is ingested, each
  of its flagged turns (the turns that talk about the fact; the flags are used only here, on the
  harness side) is a query, and the fact counts as found if one of its memories (a live memory whose
  sources include a turn of the fact's own session) is in the top 3 and reaches the threshold. Small
  talk: each user turn of a no_memory session, before that session is ingested; it "recalls" if
  anything reaches the threshold.
Pilot: set MEMRECON_E16_PILOT=N to ingest only the first N sessions of the first persona and print
timings; that output is not evidence. Progress goes to results/e16_progress.log.
  Memories per session: each Memora add/update session is about one item, so a session that gives
  2 or more memories shows the model splitting one item (D36 asks for one memory per item).
Limits: string matching misses paraphrased details (a weekday turned into a date counts as lost), so
survival is a lower bound; the related-memory step uses no threshold; one model and one prompt version.
"""
import json
import os
import random
import re
import statistics as st
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from common import ROOT, flagged_turns, iter_sessions, probe_elements, record_key, record_value
from memrecon.embed import Embedder
from memrecon.ingest import Ingestor
from memrecon.llm import client as model_client
from memrecon.memory import MemoryStore
from memrecon.retrieval import Retriever

K = 3
THRESHOLDS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
ARMS = ["oracle", "log", "note"]
PILOT = int(os.environ.get("MEMRECON_E16_PILOT", "0") or 0)
PROGRESS = ROOT / "results" / "e16_progress.log"


def progress(msg):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def digits(text):
    return set(re.findall(r"\d+(?:[.,]\d+)*", text.replace(",", "")))


def element_kept(kind, value, memory_text):
    if kind == "number":
        core = re.sub(r"[^\d.]", "", value.replace(",", "")).strip(".")
        return bool(core) and core in memory_text.replace(",", "")
    v = re.sub(r"^[\"“*'\s]+|[\"”*'.,!?:;\s]+$", "", value).lower()
    return bool(v) and v in memory_text.lower()


def replay(persona, sessions, arm, ingestor, store, retriever, results):
    keys = [record_key(s) if s.get("operation") in ("add", "update", "delete") else None for s in sessions]
    fact_session = {}  # record key -> index of the fact's most recent add/update session
    for i, s in enumerate(sessions):
        # --- retrieval checks, before this session is ingested
        live = store.active()
        if live:
            retriever.sync(store)
            if s.get("session_type") == "no_memory":
                for t in s["conversation"]:
                    if t.get("speaker") == "user_agent":
                        top = retriever.scores(t["message"])[:1]
                        results["filler_top1"].append(top[0][1])
            k = keys[i]
            if s.get("operation") in ("update", "delete") and k in fact_session:
                origin = sessions[fact_session[k]]["session_id"]
                targets = {m.id for m in live if any(src.session_id == origin for src in m.sources)}
                for t in flagged_turns(s):
                    ranked = retriever.scores(t["message"])
                    best = max((sc for mid, sc in ranked if mid in targets), default=None)
                    in_top = [(mid, sc) for mid, sc in ranked[:K] if mid in targets]
                    results["positives"].append({"stored": bool(targets), "best": best, "top_score": in_top[0][1] if in_top else None})
        # --- ingest
        t0 = time.monotonic()
        if arm == "note":
            out = ingestor.ingest_note(s)
        elif arm == "log":
            out = ingestor.ingest_log(s)
        else:
            out = ingestor.ingest_oracle(s)
        results["seconds"].append(time.monotonic() - t0)
        results["outcomes"].append((s, out))
        if s.get("operation") in ("add", "update") and keys[i] is not None:
            fact_session[keys[i]] = i
        progress(f"{persona} {arm} session {s['session_id']}: ok={out.ok} attempts={out.attempts} created={len(out.created)} ({results['seconds'][-1]:.1f} s)")


def survival(sessions_outcomes, store):
    kept, total = Counter(), Counter()
    invented, examples = 0, []
    notes_in_filler = 0
    for s, out in sessions_outcomes:
        texts = [store.get(m).text for m in out.created]
        joined = " ".join(texts)
        if s.get("session_type") == "no_memory":
            notes_in_filler += len(texts)
        if s.get("operation") in ("add", "update"):
            for t in flagged_turns(s):
                for kind, value in probe_elements(t["message"], record_value(s)):
                    total[kind] += 1
                    kept[kind] += element_kept(kind, value, joined)
        source_digits = digits(" ".join(t.get("message", "") for t in s.get("conversation", [])))
        for text in texts:
            text_wo_dates = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", text)
            extra = [d for d in digits(text_wo_dates) if d not in source_digits and d.rstrip("0").rstrip(".") not in source_digits]
            if extra:
                invented += 1
                if len(examples) < 5:
                    examples.append((s["session_id"], extra, text[:200]))
    return kept, total, invented, examples, notes_in_filler


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


def pctl(xs, ps=(50, 90, 99)):
    return " ".join(f"p{p} {np.percentile(xs, p):.0f}" for p in ps) + f" max {max(xs):.0f}" if xs else "-"


split = json.loads((ROOT / "notes" / "split.json").read_text())
personas = split["dev"][:1] if PILOT else split["dev"]
embedder = Embedder()
agent = model_client("agent")
report = {arm: defaultdict(list) for arm in ARMS}
stores = {arm: {} for arm in ARMS}
ingestors = {arm: [] for arm in ARMS}
start = time.monotonic()
for persona in personas:
    sessions = [s for _, _, s in iter_sessions(persona)]
    if PILOT:
        sessions = sessions[:PILOT]
    # embed every user turn once, in batches (they are the retrieval queries; cached for later runs)
    embedder.embed([t["message"] for s in sessions for t in s["conversation"] if t.get("speaker") == "user_agent"], "query")
    for arm in ARMS:
        store = MemoryStore(persona, mode="note" if arm == "note" else "log")
        retriever = Retriever(embedder, k=K)
        ingestor = Ingestor(agent, store, retriever if arm == "note" else None, persona=persona, tag="E16")
        replay(persona, sessions, arm, ingestor, store, retriever, report[arm])
        stores[arm][persona] = store
        ingestors[arm].append(ingestor)

if PILOT:
    print("PILOT RUN: not evidence\n")
    for arm in ("note", "log"):
        secs = report[arm]["seconds"]
        fresh = sum(1 for ing in ingestors[arm] for c in ing.calls if not c["cached"])
        if secs:
            print(f"  {arm}: {len(secs)} sessions in {sum(secs):.0f} s ({fresh} fresh calls); at this rate the 3 development personas "
                  f"(about 476 sessions) would take about {st.mean(secs) * 476 / 60:.0f} min for this arm")
    print()
print(f"development personas: {', '.join(personas)}; wall time {time.monotonic() - start:.0f} s\n")

print("INGESTION")
for arm in ("note", "log"):
    outs = [o for _, o in report[arm]["outcomes"]]
    calls = [c for ing in ingestors[arm] for c in ing.calls]
    fresh = [c for c in calls if not c["cached"]]
    att = Counter(o.attempts for o in outs)
    print(f"  {arm}: {len(outs)} sessions, {sum(not o.ok for o in outs)} failed, attempts {dict(sorted(att.items()))}, "
          f"memories {sum(len(o.created) for o in outs)}"
          + (f", superseded {sum(len(o.superseded) for o in outs)}, forgotten {sum(len(o.forgotten) for o in outs)}" if arm == "note" else "")
          + (f", span fallbacks {sum(o.span_fallbacks for o in outs)}" if arm == "log" else ""))
    out_tok = [c["completion_tokens"] for c in calls if c["completion_tokens"] is not None]
    in_tok = [c["prompt_tokens"] for c in calls if c["prompt_tokens"] is not None]
    print(f"    output tokens per answer: {pctl(out_tok)}")
    print(f"    prompt tokens per call:   {pctl(in_tok, (50, 90))}")
    secs = [sec for (s, o), sec in zip(report[arm]["outcomes"], report[arm]["seconds"])]
    print(f"    seconds per session: median {st.median(secs):.1f}, total {sum(secs):.0f} ({len(fresh)} fresh calls, {len(calls) - len(fresh)} from cache)")
    for s, o in report[arm]["outcomes"]:
        if not o.ok:
            print(f"    failed: session {s['session_id']}: {o.error}")
    # one memory per item (D36): how many memories does a session that tells about one item produce?
    per = defaultdict(list)
    for s, o in report[arm]["outcomes"]:
        if s.get("operation") in ("add", "update") and flagged_turns(s):
            per[s.get("session_type")].append(len(o.created))
    print("    memories per add/update session (one item each in Memora): "
          + "; ".join(f"{t} n={len(c)}: 0 {sum(x == 0 for x in c)}, 1 {sum(x == 1 for x in c)}, 2+ {sum(x >= 2 for x in c)}" for t, c in sorted(per.items())))
    other = [len(o.created) for s, o in report[arm]["outcomes"] if s.get("operation") not in ("add", "update")]
    print(f"    memories made in other sessions (delete and small talk): {sum(other)}")
print()

print("DETAIL SURVIVAL (candidate elements of each fact's flagged turns found in the memories made from that session)")
for arm in ARMS:
    kept, total = Counter(), Counter()
    inv = filler = 0
    ex = []
    for persona in personas:
        outs = [(s, o) for s, o in report[arm]["outcomes"] if s.get("persona") == persona]
        k, t, i, e, f = survival(outs, stores[arm][persona])
        kept.update(k)
        total.update(t)
        inv += i
        filler += f
        ex += e
    n_mem = sum(len(o.created) for _, o in report[arm]["outcomes"])
    print(f"  {arm:<6} all {pct(sum(kept.values()), sum(total.values()))} of {sum(total.values())} | "
          + " | ".join(f"{kind} {pct(kept[kind], total[kind])}" for kind in sorted(total))
          + f" | memories with an invented number {inv} of {n_mem} | memories from small-talk sessions {filler}")
    if arm == "note":
        for sid, extra, text in ex:
            print(f"      invented? session {sid} {extra}: {text}")
print()

print(f"RETRIEVAL (top {K}): fact found when the user talks about it, vs small talk recalling anything")
for arm in ARMS:
    pos = report[arm]["positives"]
    fill = report[arm]["filler_top1"]
    stored = [p for p in pos if p["stored"]]
    print(f"  {arm}: {len(pos)} fact-turns ({len(pos) - len(stored)} with the fact not in memory), {len(fill)} small-talk turns")
    if stored:
        best = [p["best"] for p in stored if p["best"] is not None]
        print(f"    similarity of the fact's best memory: median {st.median(best):.2f}; small talk's best match: median {st.median(fill):.2f}" if fill and best else "")
    print(f"    {'threshold':>9} {'found':>8} {'small talk':>11}")
    for thr in THRESHOLDS:
        found = sum(1 for p in pos if p["top_score"] is not None and p["top_score"] >= thr)
        false = sum(1 for x in fill if x >= thr)
        print(f"    {thr:9.2f} {pct(found, len(pos)):>8} {pct(false, len(fill)):>11}")
print("  (found = the fact's memory is in the top 3 and reaches the threshold; misses include facts never stored)")
print()

random.seed(16)
print("10 random note-arm memories with the session they came from:")
pool = [(s, stores["note"][s["persona"]].get(m)) for s, o in report["note"]["outcomes"] for m in o.created]
for s, m in random.sample(pool, min(10, len(pool))):
    print(f"- {s['persona']} session {s['session_id']} ({s.get('session_type')}, {s.get('operation')}), depth {m.depth}: {m.text[:220]}")
