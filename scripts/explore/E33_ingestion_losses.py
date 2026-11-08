"""E33: why the ingestion arms lose details, and why old values linger (E32 follow-up).

Question (D50, E32): in E32 the two-step arms captured more details when told (94.4% vs v1 92.5%) but
lost more of them by the end (2s2 99, 2s3 84, v1 56, v5 62) and kept more old values in active memories
(2s3 63 of 112, v1 47). Before choosing an arm: for each detail lost, what happened to the memories that
held it, in the session it was lost? For each old value still found, is it kept as history, in a memory
never superseded, or carried into a newer memory? For details never found, how near did the memories
come (possible scoring artifacts, e.g. a relative date rewritten as a calendar date)?
Method: as E32 (the frozen key, D50; E23's cached calls; code_match at overlap 0.8), but tracking, after
every session, every active memory that holds each probe. A loss is a session after which a probe that
was found is no longer found; its cause comes from what happened to the memories that held it:
superseded (by a memory written in that session) or forgotten, and whether that session was about the
probe's own record (A10) or another one ("folded"), or about no record. Old values (held, then updated
or removed by the end) found in an active memory are classed by the memory: says it changed (history
words), written before the change and never superseded (stale), or written at or after it (carried).
Never found: the best content-word overlap with a memory written from the probe's session, and for
dates and numbers whether that memory has another date or number.
Pilot: MEMRECON_E33_PILOT=N replays only the first N sessions of the first persona (not evidence).
Limits: as E32; a cause is the last loss of a probe lost at the end; the history test is a word list.
"""
import json
import os
import re
import time
from collections import Counter, defaultdict

from common import ROOT, flagged_turns, iter_sessions, record_key
from memrecon.embed import Embedder
from memrecon.ingest import Ingestor
from memrecon.llm import client as model_client
from memrecon.memory import MemoryStore
from memrecon.probes import Row, build_table, code_match, content_words, grading_view, numbers_in, true_at
from memrecon.retrieval import Retriever

K = 3
ARMS = ["ingest_note_v1", "ingest_note_v5", "two_step_v2", "two_step_v3"]
SHORT = {"ingest_note_v1": "v1", "ingest_note_v5": "v5", "two_step_v2": "2s2", "two_step_v3": "2s3"}
TWO_STEP = {"two_step_v2": ("extract_note_v2", "decide_note_v2"), "two_step_v3": ("extract_note_v2", "decide_note_v3")}
PILOT = int(os.environ.get("MEMRECON_E33_PILOT", "0") or 0)
OUT = ROOT / "results" / "probes"
HISTORY = re.compile(r"\b(used to|previously|originally|formerly|changed from|from \S+ to|updated from|was updated|was changed|"
                     r"instead of|no longer|replac\w+|revised|old|earlier|removed|deleted|dropped)\b", re.I)
DATELIKE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b", re.I)
EXAMPLES = 4


def load_rows(name, persona):
    rows = []
    for x in (OUT / f"{name}_{persona}.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            d = json.loads(x)
            d.pop("persona", None)
            d["record"] = tuple(d["record"])
            rows.append(Row(**d))
    return rows


def record_of(s):
    if s.get("operation") not in ("add", "update", "delete"):
        return None
    k = record_key(s)
    if k is not None:
        return k
    d = s.get("operation_details") or {}
    return ("goal", str(d.get("subcategory"))) if s.get("session_type") == "goal" else ("session", s["session_id"])


def short(t, n=150):
    t = " ".join(t.split())
    return t if len(t) <= n else t[:n - 1] + "…"


def probe_key(e):
    return e["row_id"] or f"{e['record']}|{e['field']}|{e['detail']}|{e['from_session']}"


split = json.loads((ROOT / "notes" / "split.json").read_text())
personas = split["dev"][:1] if PILOT else split["dev"]
embedder = Embedder()
agent = model_client("agent")
start = time.monotonic()
result = {a: defaultdict(list) for a in ARMS}
calls = Counter()

for persona in personas:
    sessions = [s for _, _, s in iter_sessions(persona)]
    if PILOT:
        sessions = sessions[:PILOT]
    rows = load_rows("E30", persona) + load_rows("E31", persona)
    with_rows = {r.session for r in rows}
    bad = set()
    for s in sessions:
        if s.get("operation") in ("add", "update", "delete") and flagged_turns(s) and s["session_id"] not in with_rows:
            bad.add(record_of(s))
    entries = build_table([r for r in rows if r.record not in bad])
    rec_of_session = {s["session_id"]: record_of(s) for s in sessions}
    op_of_session = {s["session_id"]: s.get("operation") or s.get("session_type") for s in sessions}
    last = sessions[-1]["session_id"]
    for arm in ARMS:
        store = MemoryStore(persona, mode="note")
        retriever = Retriever(embedder, k=K)
        ext, dec = TWO_STEP.get(arm, ("extract_note_v1", "decide_note_v1"))
        ing = Ingestor(agent, store, retriever, persona=persona, tag="E33", candidate_threshold=0.0,
                       note_prompt=arm if arm not in TWO_STEP else "ingest_note_v5", extract_prompt=ext, decide_prompt=dec)
        cache = {}
        holders = {}  # probe key -> set of memory ids holding it after the previous session
        info = {}  # probe key -> probe dict
        last_loss = {}  # probe key -> loss record (the latest)
        found_when_told = {}
        for s in sessions:
            if store.active():
                retriever.sync(store)
            ing.ingest_two_step(s) if arm in TWO_STEP else ing.ingest_note(s)
            sid = s["session_id"]
            active = store.active()
            probes = [p for p in grading_view(true_at(entries, sid)) if p["grade"] == "code" and p["state"] == "true"]
            now = {}
            for p in probes:
                pk = probe_key(p)
                info[pk] = p
                hold = set()
                for m in active:
                    c = (pk, m.text)
                    if c not in cache:
                        cache[c] = code_match(p, m.text)
                    if cache[c]:
                        hold.add(m.id)
                now[pk] = hold
                if p["from_session"] == sid:
                    found_when_told[pk] = bool(hold)
                before = holders.get(pk)
                if before and not hold:  # lost in this session
                    fates = []
                    for mid in before:
                        m = store.get(mid)
                        if m.retired == "forgotten":
                            fates.append(("forgotten", None))
                        elif m.retired == "superseded":
                            fates.append(("superseded", store.get(m.superseded_by) if m.superseded_by else None))
                        else:
                            fates.append(("changed in place", m))
                    kinds = {f for f, _ in fates}
                    fate = "forgotten" if kinds == {"forgotten"} else "superseded" if "superseded" in kinds else sorted(kinds)[0]
                    srec = rec_of_session.get(sid)
                    where = ("its own record" if srec == p["record"] else "no record (a session with nothing to store)" if srec is None
                             else "another record")
                    succ = next((m for f, m in fates if m is not None), None)
                    last_loss[pk] = {"session": sid, "fate": fate, "where": where, "op": op_of_session.get(sid),
                                     "before": short(store.get(next(iter(before))).text), "after": short(succ.text) if succ else ""}
            holders = now
            if sid == last:
                for pk, p in info.items():
                    if pk not in now:
                        continue  # no longer holds in the key (updated or removed by the user)
                    if now[pk]:
                        continue
                    if found_when_told.get(pk):
                        loss = last_loss.get(pk)
                        result[arm]["lost"].append({"persona": persona, "detail": p["detail"], "kind": p["kind"],
                                                    **(loss or {"fate": "(found when told, never seen lost)", "where": ""})})
                    else:
                        own = [m for m in store.memories.values() if any(src.session_id == p["from_session"] for src in m.sources)]
                        want = content_words(p["detail"])
                        best = max((len(want & content_words(m.text)) / len(want) for m in own), default=0.0) if want else 0.0
                        other = ""
                        if p["kind"] == "date" and any(DATELIKE.search(m.text) for m in own):
                            other = "a memory from its session has a calendar date"
                        elif p["kind"] == "number" and any(numbers_in(m.text) for m in own):
                            other = "a memory from its session has other numbers"
                        result[arm]["never"].append({"persona": persona, "detail": p["detail"], "kind": p["kind"],
                                                     "own_memories": len(own), "best": best, "other": other,
                                                     "example": short(own[0].text) if own else ""})
                # old values found at the end
                for p in grading_view([e for e in entries if e["state"] == "true" and e["until_session"] is not None]):
                    if p["grade"] != "code":
                        continue
                    hits = [m for m in active if code_match(p, m.text)]
                    if not hits:
                        continue
                    m = hits[0]
                    cls = ("says it changed" if HISTORY.search(m.text)
                           else "written before the change, never superseded" if m.created_at.session_id < p["until_session"]
                           else "written at or after the change, without saying so")
                    result[arm]["old"].append({"persona": persona, "detail": p["detail"], "kind": p["kind"], "class": cls,
                                               "memory": short(m.text)})
        calls.update(("fresh" if not c["cached"] else "cached") for c in ing.calls)
elapsed = time.monotonic() - start

if PILOT:
    print(f"PILOT RUN: not evidence. calls {dict(calls)}, {elapsed:.0f} s\n")
print(f"development personas: {', '.join(personas)}; calls in the replay {dict(calls)}; wall time {elapsed:.0f} s\n")

print("1. DETAILS FOUND WHEN TOLD BUT LOST BY THE END: what happened in the session they were last lost")
for a in ARMS:
    lost = result[a]["lost"]
    c = Counter((x["fate"], x["where"]) for x in lost)
    print(f"  {SHORT[a]}: {len(lost)} lost")
    for (fate, where), n in c.most_common():
        print(f"      {n:4d}  {fate}" + (f", in a session about {where}" if where else ""))
    ops = Counter(x.get("op") for x in lost)
    print("      by the losing session's operation: " + ", ".join(f"{k} {n}" for k, n in ops.most_common()))
    print("      by kind: " + ", ".join(f"{k} {n}" for k, n in Counter(x["kind"] for x in lost).most_common()))
print()

print("2. EXAMPLES OF LOSSES (up to 4 per arm and cause: the detail; the memory that held it; what replaced it)")
for a in ARMS:
    by = defaultdict(list)
    for x in result[a]["lost"]:
        by[(x["fate"], x["where"])].append(x)
    print(f"  {SHORT[a]}:")
    for k, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"    [{k[0]}, {k[1] or '-'}]")
        for x in xs[:EXAMPLES]:
            print(f"      - {x['persona']} session {x.get('session', '?')} [{x['kind']}] {short(x['detail'], 90)}")
            print(f"          held by: {x.get('before', '')}")
            if x.get("after"):
                print(f"          now:     {x['after']}")
print()

print("3. NEVER FOUND (not found right after the session it was told in, nor at the end): how near the session's memories came")
for a in ARMS:
    nv = result[a]["never"]
    b = Counter("no memory from its session" if x["own_memories"] == 0 else "best overlap 0" if x["best"] == 0
                else "best overlap < 0.5" if x["best"] < 0.5 else "best overlap 0.5-0.8" if x["best"] < 0.8 else "overlap >= 0.8 (short detail, words differ)"
                for x in nv)
    print(f"  {SHORT[a]}: {len(nv)}; " + ", ".join(f"{k} {n}" for k, n in b.most_common())
          + "; " + ", ".join(f"{k} {n}" for k, n in Counter(x["other"] for x in nv if x["other"]).most_common()))
    for x in [x for x in nv if x["other"]][:EXAMPLES]:
        print(f"      - {x['persona']} [{x['kind']}] {short(x['detail'], 60)} | memory: {x['example']}")
print()

print("4. OLD VALUES FOUND AT THE END (held once, updated or removed by the end): the memory that has it")
for a in ARMS:
    old = result[a]["old"]
    print(f"  {SHORT[a]}: {len(old)}; " + ", ".join(f"{k} {n}" for k, n in Counter(x["class"] for x in old).most_common()))
    for cls in ("written before the change, never superseded", "written at or after the change, without saying so"):
        for x in [x for x in old if x["class"] == cls][:2]:
            print(f"      - [{cls}] {x['persona']} {short(x['detail'], 60)} | {x['memory']}")
