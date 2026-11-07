"""E18: does ingestion prompt v2 fix what E16 and E17 found in v1, without losing details?

Question (D38): on the development personas, compare the note arm's ingestion with prompt
ingest_note_v1 (as in E16) and ingest_note_v2, on
  (1) reliability: failures, attempts per session (a retry means the first answer was unusable), answer
      length, time;
  (2) granularity: memories per add/update session, words per memory, the same memory twice in one answer;
  (3) detail survival and invented numbers (E16's method, with numbers compared by value, since v2
      allows "10k" to be written "10,000"; a number is invented if it is in neither the conversation nor
      the memories the session replaced);
  (4) update recognition: in an update session, were the memories of the record's earlier sessions shown
      to the model (the related memories), and did it replace them? in a delete session, did it forget
      or change them? and at the end of the replay, how many memories of a record from before its last
      update or delete are still active (stale copies)?
  (5) finished tasks: what happens when the user says they finished something (forget, change, or new)?
  (6) wording: working-out left in the text, earlier versions kept in an updated memory ("used to",
      "previously"), a change described instead of the current state ("updated", "the changes"), dates
      left relative to the conversation ("next month", "tomorrow"), what "we" became;
  (7) retrieval at the chosen threshold (0.70, top 3), E16's method.
Method: development personas only (notes/split.json). Each persona's sessions are replayed in file order
(A6) twice, once per prompt version, with the real ingestion code and the agent model, related memories
as in E16 (top 3 per user turn, up to 10, no threshold). The v1 replay is E16's note arm, answered from
the cache (fresh calls should be 0); v2 makes new calls (about 476), cached for reruns. A record is
identified as in E12-E17 (A10). The flags are used only on the harness side, to find the facts' turns.
Pilot: MEMRECON_E18_PILOT=N ingests only the first N sessions of the first persona and prints timings;
that output is not evidence. Progress goes to results/e18_progress.log.
Limits: string and value matching misses paraphrased details, so survival is a lower bound; "stale"
memories are an upper bound (a record split into several memories may leave a part that is still
true, such as a document's title when only its deadline changed); dates written out ("June 2, 2025")
are left out of the invented-number check; one run per version (no seeds); the finished-task and
wording checks are by pattern, and their samples are printed to be read.
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
VERSIONS = ["ingest_note_v1", "ingest_note_v2"]
SHORT = {"ingest_note_v1": "v1", "ingest_note_v2": "v2"}
PILOT = int(os.environ.get("MEMRECON_E18_PILOT", "0") or 0)
PROGRESS = ROOT / "results" / "e18_progress.log"

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
    ing = RecordingIngestor(agent, store, retriever, persona=persona, tag="E18", note_prompt=version)
    keys = [record_key(s) if s.get("operation") in ("add", "update", "delete") else None for s in sessions]
    sessions_of = defaultdict(list)  # record key -> session ids of its add/update sessions so far
    last_change = {}  # record key -> session id of its latest update/delete
    for i, s in enumerate(sessions):
        k = keys[i]
        live = store.active()
        # retrieval check (E16): before an update/delete of a record, its earlier memories vs its flagged turns
        if live and k is not None and s.get("operation") in ("update", "delete") and sessions_of[k]:
            retriever.sync(store)
            origin = sessions_of[k][-1]
            targets = {m.id for m in live if any(src.session_id == origin for src in m.sources)}
            for t in flagged_turns(s):
                top = [(mid, sc) for mid, sc in retriever.scores(t["message"])[:K] if mid in targets and sc >= THRESHOLD]
                rep["found"].append(bool(top))
        earlier = {m.id for m in live if k is not None and any(src.session_id in sessions_of[k] for src in m.sources)}
        ing.last_shown = set()
        t0 = time.monotonic()
        out = ing.ingest_note(s)
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
    rep["calls"] += ing.calls
    return store


split = json.loads((ROOT / "notes" / "split.json").read_text())
personas = split["dev"][:1] if PILOT else split["dev"]
embedder = Embedder()
agent = model_client("agent")
reports = {v: defaultdict(list) for v in VERSIONS}
stores = {v: {} for v in VERSIONS}
start = time.monotonic()
for persona in personas:
    sessions = [s for _, _, s in iter_sessions(persona)]
    if PILOT:
        sessions = sessions[:PILOT]
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

print(f"7. RETRIEVAL at {THRESHOLD:.2f}, top {K}: a record's latest memory found by the flagged turns of its next update/delete")
for v in VERSIONS:
    f = reports[v]["found"]
    print(f"  {SHORT[v]}: {pct(sum(f), len(f))} of {len(f)} turns")
print()

random.seed(18)
by_session = {v: {(p, s["session_id"]): (s, o, shown) for p, s, o, _, shown in reports[v]["outcomes"]} for v in VERSIONS}
print("SIDE BY SIDE: 6 random add/update sessions (the user's flagged turns, then each version's memories)")
pool = [key for key, (s, o, _) in by_session["ingest_note_v2"].items() if s.get("operation") in ("add", "update") and flagged_turns(s)]
for key in random.sample(pool, min(6, len(pool))):
    s = by_session["ingest_note_v2"][key][0]
    print(f"- {key[0]} session {key[1]} ({s.get('session_type')}, {s.get('operation')}). User: " + " / ".join(t["message"][:160] for t in flagged_turns(s)))
    for v in VERSIONS:
        _, o, _ = by_session[v][key]
        for m in o.created:
            print(f"    {SHORT[v]} * {text(v, key[0], m)[:400]}")
print()

print("v2 UPDATE SESSIONS WHERE AN EARLIER MEMORY WAS SHOWN BUT NOT REPLACED (5 at random)")
missed = [(p, s, o, shown) for p, s, o, earlier, shown in reports["ingest_note_v2"]["outcomes"]
          if s.get("operation") == "update" and shown and not (earlier & set(o.superseded))]
for p, s, o, shown in random.sample(missed, min(5, len(missed))):
    print(f"- {p} session {s['session_id']} ({s.get('session_type')}). User: " + " / ".join(t["message"][:160] for t in flagged_turns(s)))
    for m in sorted(shown):
        print(f"    shown   * {text('ingest_note_v2', p, m)[:300]}")
    for m in o.created:
        print(f"    written * {text('ingest_note_v2', p, m)[:300]}")
print()

print("v2 DELETE SESSIONS (5 at random)")
dels = [(p, s, o) for p, s, o, _, _ in reports["ingest_note_v2"]["outcomes"] if s.get("operation") == "delete"]
for p, s, o in random.sample(dels, min(5, len(dels))):
    print(f"- {p} session {s['session_id']}: forgot {len(o.forgotten)}, changed {len(o.superseded)}. User: "
          + " / ".join(t["message"][:160] for t in flagged_turns(s)))
    for m in o.created:
        print(f"    written * {text('ingest_note_v2', p, m)[:300]}")
print()

print("FINISHED TASKS, both versions (4 at random)")
for key in random.sample(sorted(fin_examples), min(4, len(fin_examples))):
    s = by_session["ingest_note_v2"][key][0]
    print(f"- {key[0]} session {key[1]}. User: " + " / ".join(t["message"][:160] for t in flagged_turns(s)))
    for ver, kind, texts in fin_examples[key]:
        print(f"    {ver} {kind}" + "".join(f"\n      * {t[:200]}" for t in texts))
