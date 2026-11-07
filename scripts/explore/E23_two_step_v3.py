"""E23: two-step ingestion with step 2 shown the other facts of the same conversation (decide v3).

Question (D43): in E22 (two-step prompts v2), unnamed pieces of different documents still replaced each
other (folding 29 outside preference sessions; "Dr. Anya Sharma ... will be leading the user's project"
replaced by another project's lead), because step 2 could not tell which document a new fact is about.
Showing step 2 the other facts learned from the same conversation (which include the document's title)
is meant to fix this at candidate threshold 0.0. Does it reduce cross-document folding without losing
E22's gains (update handling, retention, size, detail)?
On the development personas, for one-call v1 and v5 and two-step v2 (extract v2 + decide v2) and v3
(extract v2 + decide v3): E22's sections.
Method: as E22. v1 and v5 replay from the cache. Since D42 (a replace with identical text is no rewrite)
applies to all two-step runs, two-step v2 here can differ slightly from E22 after the one identical
replace E22 made, and may need a few new calls. Two-step v3 makes new calls (about 1,350, cached for
reruns). Candidate threshold MEMRECON_E23_THRESHOLD (default 0.0), top 5 per fact.
Pilot: MEMRECON_E23_PILOT=N ingests only the first N sessions of the first persona and prints timings;
that output is not evidence. Progress goes to results/e23_progress.log.
Limits: as E22.
"""
import json
import os
import random
import re
import statistics as st
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
THRESHOLD = 0.70
THRESHOLDS = [0.60, 0.65, 0.70, 0.75]
VERSIONS = ["ingest_note_v1", "ingest_note_v5", "two_step_v2", "two_step_v3"]
SHORT = {"ingest_note_v1": "v1", "ingest_note_v5": "v5", "two_step_v2": "2s2", "two_step_v3": "2s3"}
TWO_STEP = {"two_step_v2": ("extract_note_v2", "decide_note_v2"), "two_step_v3": ("extract_note_v2", "decide_note_v3")}
NEW = "two_step_v3"
CAND_THRESHOLD = float(os.environ.get("MEMRECON_E23_THRESHOLD", "0.0") or 0.0)
PILOT = int(os.environ.get("MEMRECON_E23_PILOT", "0") or 0)
PROGRESS = ROOT / "results" / "e23_progress.log"

MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mil": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
NUM = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(k|thousand|million|mil|bn|billion|m|b)?\b", re.I)
MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATES = re.compile(rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b{MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTHS}(?:,?\s+\d{{4}})?", re.I)
FINISHED = re.compile(r"\b(finished|completed|done with|wrapped up|checked off|crossed off)\b", re.I)
WORKING = re.compile(r"\b(days? (?:after|from|before|later)|i\.e\.|which is|calculated|counting from)\b|\d\s*[+\-]\s*\d+\s*=", re.I)
HISTORY = re.compile(r"\b(used to|previously|originally|changed from|was (?:originally|previously)|earlier (?:it|this) was)\b", re.I)
WE = re.compile(r"\b(we|our|us)\b", re.I)
RELATIVE = re.compile(r"\b(today|tonight|tomorrow|yesterday|this (?:morning|afternoon|evening|week|weekend|month|year)|"
                      r"(?:next|last) (?:week|weekend|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
                      r"in \d+ days|in (?:a|one|two|three) (?:days?|weeks?|months?)|within the next|end of (?:next|this) (?:week|month))\b", re.I)
NARRATIVE = re.compile(r"\b(updated|the changes?|was changed|has changed|changed (?:it|the)|revised|modified)\b", re.I)
OTHERS = re.compile(r"\b(and others|with others|and (?:their|his|her) (?:team|colleagues|partner|family|friends))\b", re.I)


def progress(msg):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def values(text):
    """Numbers in a text as values, with k / million / billion expanded; dates left out."""
    out = set()
    for m in NUM.finditer(DATES.sub(" ", text)):
        v = float(m.group(1).replace(",", "")) * MULT.get((m.group(2) or "").lower(), 1)
        out.add(round(v, 4))
    return out


def element_kept(kind, value, memory_text):
    """E16's check, plus numbers compared by value."""
    if kind == "number":
        core = re.sub(r"[^\d.]", "", value.replace(",", "")).strip(".")
        if core and core in memory_text.replace(",", ""):
            return True
        want = values(value)
        return bool(want) and want <= values(memory_text)
    v = re.sub(r"^[\"“*'\s]+|[\"”*'.,!?:;\s]+$", "", value).lower()
    return bool(v) and v in memory_text.lower()


class RecordingIngestor(Ingestor):
    """The real Ingestor; also records which memories were shown to the model as related."""

    def _related(self, session, at):
        text, labels = super()._related(session, at)
        self.last_shown = set(labels.values())
        return text, labels


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


def pctl(xs, ps=(50, 90, 99)):
    return " ".join(f"p{p} {np.percentile(xs, p):.0f}" for p in ps) + f" max {max(xs):.0f}" if xs else "-"


def replay(persona, sessions, version, embedder, rep):
    store = MemoryStore(persona, mode="note")
    retriever = Retriever(embedder, k=K)
    ext, dec = TWO_STEP.get(version, ("extract_note_v1", "decide_note_v1"))
    ing = RecordingIngestor(agent, store, retriever, persona=persona, tag="E23",
                            note_prompt=version if version not in TWO_STEP else "ingest_note_v5", candidate_threshold=CAND_THRESHOLD,
                            extract_prompt=ext, decide_prompt=dec)
    keys = [record_key(s) if s.get("operation") in ("add", "update", "delete") else None for s in sessions]
    sessions_of = defaultdict(list)  # record key -> session ids of its add/update sessions so far
    last_change = {}  # record key -> session id of its latest update/delete
    for i, s in enumerate(sessions):
        k = keys[i]
        live = store.active()
        if live:
            retriever.sync(store)
            if s.get("session_type") == "no_memory":
                for t in s["conversation"]:
                    if t.get("speaker") == "user_agent":
                        rep["filler_top1"].append(retriever.scores(t["message"])[0][1])
        # retrieval check (E16, E19): before an update/delete of a record, its earlier memories vs its flagged turns
        if live and k is not None and s.get("operation") in ("update", "delete") and sessions_of[k]:
            origin = sessions_of[k][-1]
            targets = {m.id: m for m in live if any(src.session_id == origin for src in m.sources)}
            for t in flagged_turns(s):
                top = [(mid, sc) for mid, sc in retriever.scores(t["message"])[:K] if mid in targets]
                words = min((len(m.text.split()) for m in targets.values()), default=None)
                rep["found"].append({"top": top[0][1] if top else None, "words": words})
        earlier = {m.id for m in live if k is not None and any(src.session_id in sessions_of[k] for src in m.sources)}
        ing.last_shown = set()
        t0 = time.monotonic()
        out = ing.ingest_two_step(s) if version in TWO_STEP else ing.ingest_note(s)
        if version in TWO_STEP:
            rep["decisions"] += [{**d, "persona": persona, "session": s} for d in ing.decision_log]
        rep["seconds"].append(time.monotonic() - t0)
        rep["outcomes"].append((persona, s, out, earlier, earlier & ing.last_shown))
        if s.get("operation") in ("add", "update") and k is not None:
            sessions_of[k].append(s["session_id"])
        if s.get("operation") in ("update", "delete") and k is not None:
            last_change[k] = s["session_id"]
        progress(f"{persona} {SHORT[version]} session {s['session_id']}: ok={out.ok} attempts={out.attempts} created={len(out.created)} ({rep['seconds'][-1]:.1f} s)")
    # stale copies at the end: active memories of a record whose sources all come from before its last change
    for k, sid in last_change.items():
        old = [m for m in store.active() if m.sources and all(src.session_id < sid for src in m.sources)
               and any(src.session_id in sessions_of[k] for src in m.sources)]
        rep["stale"].append(len(old))
    # retention: records added and never updated or deleted; elements kept at ingestion vs still active at the end
    created_in = {o.session_id: o for p_, s_, o, _, _ in rep["outcomes"] if p_ == persona}
    active_end = store.active()
    for i, s in enumerate(sessions):
        k = keys[i]
        if s.get("operation") != "add" or k is None or k in last_change or not flagged_turns(s):
            continue
        then = " ".join(store.get(m).text for m in created_in[s["session_id"]].created)
        now = " ".join(m.text for m in active_end if any(src.session_id == s["session_id"] for src in m.sources))
        elems = [(kind, value) for t in flagged_turns(s) for kind, value in probe_elements(t["message"], record_value(s))]
        rep["retention"].append({"persona": persona, "session": s, "n": len(elems),
                                 "then": sum(element_kept(kd, v, then) for kd, v in elems),
                                 "now": sum(element_kept(kd, v, now) for kd, v in elems),
                                 "lost": [v for kd, v in elems if element_kept(kd, v, then) and not element_kept(kd, v, now)],
                                 "now_text": now})
    rep["calls"] += ing.calls
    return store


split = json.loads((ROOT / "notes" / "split.json").read_text())
personas = split["dev"][:1] if PILOT else split["dev"]
embedder = Embedder()
agent = model_client("agent")
reports = {v: defaultdict(list) for v in VERSIONS}
stores = {v: {} for v in VERSIONS}
start = time.monotonic()
record_of = {}  # (persona, session id) -> record key or None
for persona in personas:
    sessions = [s for _, _, s in iter_sessions(persona)]
    if PILOT:
        sessions = sessions[:PILOT]
    for s in sessions:
        record_of[(persona, s["session_id"])] = record_key(s) if s.get("operation") in ("add", "update", "delete") else None
    for v in VERSIONS:
        stores[v][persona] = replay(persona, sessions, v, embedder, reports[v])

if PILOT:
    print("PILOT RUN: not evidence\n")
    for v in VERSIONS:
        secs, calls = reports[v]["seconds"], reports[v]["calls"]
        fresh = sum(1 for c in calls if not c["cached"])
        print(f"  {SHORT[v]}: {len(secs)} sessions in {sum(secs):.0f} s ({fresh} fresh calls); 476 sessions would take about "
              f"{st.mean(secs) * 476 / 60:.0f} min at this rate")
    print()
print(f"development personas: {', '.join(personas)}; wall time {time.monotonic() - start:.0f} s\n")


def text(v, persona, mid):
    return stores[v][persona].get(mid).text


print("1. RELIABILITY")
for v in VERSIONS:
    rep = reports[v]
    outs = [o for _, _, o, _, _ in rep["outcomes"]]
    fresh = sum(1 for c in rep["calls"] if not c["cached"])
    out_tok = [c["completion_tokens"] for c in rep["calls"] if c["completion_tokens"] is not None]
    print(f"  {SHORT[v]}: {len(outs)} sessions, {sum(not o.ok for o in outs)} failed, attempts {dict(sorted(Counter(o.attempts for o in outs).items()))}, "
          f"{fresh} fresh calls, {len(rep['calls']) - fresh} from cache")
    print(f"      output tokens per answer: {pctl(out_tok)}; seconds per session: median {st.median(rep['seconds']):.1f}, total {sum(rep['seconds']):.0f}")
    for p, s, o, _, _ in rep["outcomes"]:
        if not o.ok:
            print(f"      failed: {p} session {s['session_id']}: {o.error}")
print("  (attempts > 1: the first answer was unusable or cut off; the reasons are in the attempt files of the audit log)\n")

print("2. GRANULARITY (Memora's add/update sessions are about one record each)")
for v in VERSIONS:
    per, words, dup = defaultdict(list), [], 0
    for p, s, o, _, _ in reports[v]["outcomes"]:
        texts = [text(v, p, m) for m in o.created]
        words += [len(t.split()) for t in texts]
        dup += len(texts) != len(set(texts))
        if s.get("operation") in ("add", "update") and flagged_turns(s):
            per[s.get("session_type")].append(len(texts))
    print(f"  {SHORT[v]}: sessions with the same memory twice: {dup}; words per memory: {pctl(words, (50, 90))}")
    for t, c in sorted(per.items()):
        print(f"      {t:<11} n={len(c):4d}: 0 {sum(x == 0 for x in c):3d} | 1 {sum(x == 1 for x in c):3d} | 2-4 {sum(2 <= x <= 4 for x in c):3d} "
              f"| 5+ {sum(x >= 5 for x in c):3d} | mean {st.mean(c):.2f}")
print()

print("3. DETAIL SURVIVAL AND INVENTED NUMBERS")
for v in VERSIONS:
    kept, total = Counter(), Counter()
    invented, n_mem, ex = 0, 0, []
    for p, s, o, _, _ in reports[v]["outcomes"]:
        texts = [text(v, p, m) for m in o.created]
        n_mem += len(texts)
        if s.get("operation") in ("add", "update"):
            joined = " ".join(texts)
            for t in flagged_turns(s):
                for kind, value in probe_elements(t["message"], record_value(s)):
                    total[kind] += 1
                    kept[kind] += element_kept(kind, value, joined)
        # an updated memory carries details forward from the memories it replaces, so those count as source
        # too (fix, Nov 6: the first run compared with the conversation only, which counted carried-forward
        # numbers as invented)
        replaced = " ".join(text(v, p, m) for m in o.superseded)
        source = values(" ".join(t.get("message", "") for t in s.get("conversation", [])) + " " + replaced)
        for tx in texts:
            extra = sorted(x for x in values(tx) if x not in source)
            if extra:
                invented += 1
                if len(ex) < 4:
                    ex.append((p, s["session_id"], extra, tx[:200]))
    print(f"  {SHORT[v]}: all {pct(sum(kept.values()), sum(total.values()))} of {sum(total.values())} | "
          + " | ".join(f"{kind} {pct(kept[kind], total[kind])}" for kind in sorted(total))
          + f" | memories with a number in neither the conversation nor the memories replaced: {invented} of {n_mem}")
    for p, sid, extra, tx in ex:
        print(f"      {p} session {sid} {extra}: {tx}")
print()

print("4. UPDATE RECOGNITION (memories of the record's earlier sessions, active just before the session)")
for v in VERSIONS:
    up = Counter()
    de = Counter()
    for p, s, o, earlier, shown in reports[v]["outcomes"]:
        if not earlier:
            continue
        retired = earlier & set(o.superseded + o.forgotten)
        if s.get("operation") == "update":
            up["n"] += 1
            up["shown"] += bool(shown)
            up["replaced any"] += bool(earlier & set(o.superseded))
            up["replaced all"] += earlier <= set(o.superseded)
            up["shown, none replaced"] += bool(shown) and not (earlier & set(o.superseded))
        elif s.get("operation") == "delete":
            de["n"] += 1
            de["shown"] += bool(shown)
            de["forgot any"] += bool(earlier & set(o.forgotten))
            de["changed any"] += bool(earlier & set(o.superseded))
            de["all retired"] += earlier == retired
            de["none retired"] += not retired
    stale = reports[v]["stale"]
    print(f"  {SHORT[v]} update sessions n={up['n']}: earlier memory shown {pct(up['shown'], up['n'])}, replaced any {pct(up['replaced any'], up['n'])}, "
          f"replaced all {pct(up['replaced all'], up['n'])}, shown but none replaced {up['shown, none replaced']}")
    print(f"     delete sessions n={de['n']}: shown {pct(de['shown'], de['n'])}, forgot any {pct(de['forgot any'], de['n'])}, "
          f"changed any {pct(de['changed any'], de['n'])}, all retired {pct(de['all retired'], de['n'])}, none retired {pct(de['none retired'], de['n'])}")
    print(f"     at the end: {len(stale)} records with an update or delete; {sum(x > 0 for x in stale)} still have memories from before it "
          f"({sum(stale)} memories; upper bound, see Limits)")
print()

print("5. FINISHED TASKS (update/delete sessions whose flagged turns say finished, completed, done with, ...)")
fin_examples = defaultdict(list)
for v in VERSIONS:
    c = Counter()
    for p, s, o, earlier, _ in reports[v]["outcomes"]:
        if s.get("operation") in ("update", "delete") and any(FINISHED.search(t["message"]) for t in flagged_turns(s)):
            kind = "forgot" if o.forgotten else "changed" if o.superseded else "new only" if o.created else "nothing"
            c[kind] += 1
            c["n"] += 1
            fin_examples[(p, s["session_id"])].append((SHORT[v], kind, [text(v, p, m) for m in o.created]))
    print(f"  {SHORT[v]}: n={c['n']}: forgot {c['forgot']}, changed {c['changed']}, new memory only {c['new only']}, nothing {c['nothing']}")
print()

print("6. WORDING (pattern counts over all memories made)")
for v in VERSIONS:
    w = h = we_n = others = rel = nar = 0
    rel_ex = []
    total = 0
    for p, s, o, _, _ in reports[v]["outcomes"]:
        said_we = any(WE.search(t["message"]) for t in s["conversation"] if t.get("speaker") == "user_agent")
        for m in o.created:
            tx = text(v, p, m)
            total += 1
            w += bool(WORKING.search(tx))
            h += bool(HISTORY.search(tx))
            rel += bool(RELATIVE.search(tx))
            nar += bool(NARRATIVE.search(tx))
            if RELATIVE.search(tx) and len(rel_ex) < 3:
                rel_ex.append(tx[:300])
            if said_we:
                we_n += 1
                others += bool(OTHERS.search(tx))
    print(f"  {SHORT[v]}: {total} memories; working-out {w}; earlier version kept ('used to', 'previously', ...) {h}; "
          f"from sessions where the user said we/our/us: {we_n}, of which say 'and others' (or team, family...) {others}")
    print(f"      change described ('updated', 'the changes', ...) {nar}; date left relative ('next month', 'tomorrow', ...) {rel}")
    for tx in rel_ex:
        print(f"      relative: {tx}")
print()

print(f"7. RETRIEVAL (top {K}): a record's latest memory found by its next update/delete's flagged turns, vs small talk recalling anything")
print(f"  {'threshold':>9} " + " ".join(f"{SHORT[v] + ' found':>9} {SHORT[v] + ' small':>9}" for v in VERSIONS))
for thr in THRESHOLDS:
    cells = []
    for v in VERSIONS:
        f, fill = reports[v]["found"], reports[v]["filler_top1"]
        cells.append(f"{pct(sum(1 for x in f if x['top'] is not None and x['top'] >= thr), len(f)):>9} {pct(sum(x >= thr for x in fill), len(fill)):>9}")
    print(f"  {thr:9.2f} " + " ".join(cells))
for v in VERSIONS:
    f = reports[v]["found"]
    hit = [x["words"] for x in f if x["top"] is not None and x["top"] >= THRESHOLD and x["words"]]
    miss = [x["words"] for x in f if not (x["top"] is not None and x["top"] >= THRESHOLD) and x["words"]]
    print(f"  {SHORT[v]}: {len(f)} turns; words in the record's shortest memory at {THRESHOLD:.2f}: found median "
          f"{st.median(hit) if hit else '-'}, missed median {st.median(miss) if miss else '-'}")
print()


def records(v, persona, mid):
    return {record_of.get((persona, src.session_id)) for src in stores[v][persona].get(mid).sources} - {None}


print("8. CHANGES THAT REPLACE A MEMORY OF ANOTHER RECORD (preference sessions are expected; the rest is folding)")
folded = defaultdict(list)
for v in VERSIONS:
    c = Counter()
    for p, s, o, _, _ in reports[v]["outcomes"]:
        own = record_of.get((p, s["session_id"]))
        for mid in o.superseded:
            c["replaced"] += 1
            if own is None or own not in records(v, p, mid):
                kind = "preference" if s.get("session_type") == "preference" else "other"
                c[kind] += 1
                if kind == "other":
                    new = [m for m in o.created if mid in stores[v][p].get(m).parents]
                    folded[v].append((p, s, text(v, p, mid), [text(v, p, m) for m in new]))
    print(f"  {SHORT[v]}: {c['replaced']} memories replaced; of another record in preference sessions {c['preference']}, elsewhere {c['other']}")
print()

print("9. ADD/UPDATE SESSIONS WITH FLAGGED TURNS THAT SAVED NOTHING")
for v in VERSIONS:
    empty = [(p, s) for p, s, o, _, _ in reports[v]["outcomes"] if s.get("operation") in ("add", "update") and flagged_turns(s) and not o.created]
    print(f"  {SHORT[v]}: {len(empty)}" + "".join(f"\n    - {p} session {s['session_id']}: " + " / ".join(t["message"][:140] for t in flagged_turns(s)) for p, s in empty))
print()

print("10. RETENTION: records added and never updated or deleted (candidate elements kept at ingestion vs in active memories at the end)")
for v in VERSIONS:
    r = reports[v]["retention"]
    n, then, now = sum(x["n"] for x in r), sum(x["then"] for x in r), sum(x["now"] for x in r)
    lost = [x for x in r if x["lost"]]
    print(f"  {SHORT[v]}: {len(r)} records, {n} elements; kept at ingestion {pct(then, n)}, still active at the end {pct(now, n)}; "
          f"records that lost an element after it was saved {len(lost)}; with no active memory at the end {sum(1 for x in r if not x['now_text'])}")
print()

print(f"11. TWO-STEP DECISIONS (candidate threshold {CAND_THRESHOLD:.2f}, top 5 per fact)")
for ts in TWO_STEP:
    print(f" {SHORT[ts]}:")
    dec = reports[ts]["decisions"]
    outs2 = [o for _, _, o, _, _ in reports[ts]["outcomes"]]
    print(f"  {sum(o.facts for o in outs2)} facts from {len(outs2)} sessions; decision calls failing 3 times: {sum(o.decide_failures for o in outs2)}")
    print("  " + ", ".join(f"{a} {n}" for a, n in Counter(d["action"] for d in dec).most_common()))
    by_type = defaultdict(Counter)
    for d in dec:
        by_type[d["session"].get("operation") or "none"][d["action"]] += 1
    for op, c in sorted(by_type.items()):
        print(f"    in {op} sessions: " + ", ".join(f"{a} {n}" for a, n in c.most_common()))
    same, other = [], []
    for d in dec:
        own = record_of.get((d["persona"], d["session"]["session_id"]))
        for mid, score in d["candidates"]:
            (same if own is not None and own in records(ts, d["persona"], mid) else other).append(score)
    print(f"  candidates shown: {len(same)} of the fact's own record, {len(other)} of other records (or of a session about no record)")
    print(f"  {'threshold':>9} {'own record kept':>16} {'others kept':>12}")
    for thr in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
        print(f"  {thr:9.2f} {pct(sum(x >= thr for x in same), len(same)):>16} {pct(sum(x >= thr for x in other), len(other)):>12}")
    if same and other:
        print(f"  similarity: own record median {st.median(same):.2f}, others median {st.median(other):.2f}")
    rep_kind = Counter()
    for d in dec:
        if d["action"] != "replace":
            continue
        own = record_of.get((d["persona"], d["session"]["session_id"]))
        for mid in d["ids"]:
            ok = own is not None and own in records(ts, d["persona"], mid)
            rep_kind[(d["session"].get("session_type"), "own record" if ok else "other record")] += 1
    print("  replacements: " + ", ".join(f"{t} / {k} {n}" for (t, k), n in sorted(rep_kind.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))))
    print()

random.seed(23)
by_session = {v: {(p, s["session_id"]): (s, o, shown) for p, s, o, _, shown in reports[v]["outcomes"]} for v in VERSIONS}
print("SIDE BY SIDE: 6 random add/update sessions (the user's flagged turns, then each version's memories)")
pool = [key for key, (s, o, _) in by_session[NEW].items() if s.get("operation") in ("add", "update") and flagged_turns(s)]
for key in random.sample(pool, min(6, len(pool))):
    s = by_session[NEW][key][0]
    print(f"- {key[0]} session {key[1]} ({s.get('session_type')}, {s.get('operation')}). User: " + " / ".join(t["message"][:160] for t in flagged_turns(s)))
    for v in VERSIONS:
        _, o, _ = by_session[v][key]
        for m in o.created:
            print(f"    {SHORT[v]} * {text(v, key[0], m)[:400]}")
print()

print("two-step v3 UPDATE SESSIONS WHERE AN EARLIER MEMORY WAS SHOWN BUT NOT REPLACED (5 at random)")
missed = [(p, s, o, shown) for p, s, o, earlier, shown in reports[NEW]["outcomes"]
          if s.get("operation") == "update" and shown and not (earlier & set(o.superseded))]
for p, s, o, shown in random.sample(missed, min(5, len(missed))):
    print(f"- {p} session {s['session_id']} ({s.get('session_type')}). User: " + " / ".join(t["message"][:160] for t in flagged_turns(s)))
    for m in sorted(shown):
        print(f"    shown   * {text(NEW, p, m)[:300]}")
    for m in o.created:
        print(f"    written * {text(NEW, p, m)[:300]}")
print()

print("two-step v3 DELETE SESSIONS (5 at random)")
dels = [(p, s, o) for p, s, o, _, _ in reports[NEW]["outcomes"] if s.get("operation") == "delete"]
for p, s, o in random.sample(dels, min(5, len(dels))):
    print(f"- {p} session {s['session_id']}: forgot {len(o.forgotten)}, changed {len(o.superseded)}. User: "
          + " / ".join(t["message"][:160] for t in flagged_turns(s)))
    for m in o.created:
        print(f"    written * {text(NEW, p, m)[:300]}")
print()

print("FINISHED TASKS, both versions (4 at random)")
for key in random.sample(sorted(fin_examples), min(4, len(fin_examples))):
    s = by_session[NEW][key][0]
    print(f"- {key[0]} session {key[1]}. User: " + " / ".join(t["message"][:160] for t in flagged_turns(s)))
    for ver, kind, texts in fin_examples[key]:
        print(f"    {ver} {kind}" + "".join(f"\n      * {t[:200]}" for t in texts))

print()
print("two-step v3 CHANGES THAT FOLDED A MEMORY OF ANOTHER RECORD, not preference sessions (5 at random)")
for p, s, old, new in random.sample(folded[NEW], min(5, len(folded[NEW]))):
    print(f"- {p} session {s['session_id']} ({s.get('session_type')}, {s.get('operation')}). User: " + (" / ".join(t["message"][:140] for t in flagged_turns(s)) or "(no flagged turns)"))
    print(f"    replaced * {old[:300]}")
    for t in new:
        print(f"    with     * {t[:300]}")
print()
print("two-step v3 RECORDS THAT LOST AN ELEMENT AFTER IT WAS SAVED (5 at random)")
lost = [x for x in reports[NEW]["retention"] if x["lost"]]
for x in random.sample(lost, min(5, len(lost))):
    print(f"- {x['persona']} session {x['session']['session_id']}: lost {x['lost']}. User: " + " / ".join(t["message"][:140] for t in flagged_turns(x["session"])))
    print(f"    now * {x['now_text'][:300] or '(no active memory)'}")
