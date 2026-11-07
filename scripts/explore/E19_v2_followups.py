"""E19: follow-up questions on E18 (ingestion prompts v1 and v2), answered from E18's cached calls.

Question (D38, before writing prompt v3): E18 showed v2 fixing duplicates and stale copies, but also
(a) 14 sessions needing a second attempt, (b) retrieval at 0.70 finding a record's memory less often
(55.4% vs 63.8%), (c) an add session rewriting a memory about a different post, (d) 8 add/update sessions
saving nothing, (e) dates left relative, mostly inside documents' contents, (f) no memory saying "and
others" for "we", (g) changes described ("updated", "the changes") and earlier versions kept ("used to").
For each version:
  1. why were second attempts needed (the feedback the model was sent);
  2. retrieval vs small talk over a range of thresholds (E16's method, so the threshold chosen on v1
     can be checked for v2), and the length of the memories found vs missed;
  3. changes that replace a memory of a different record than the session's own (or in a session about
     no record): unrelated rewrites, which would add depth without a user change;
  4. the add/update sessions that saved nothing, listed;
  5. relative dates in memories about documents (meeting notes, emails, posts, proposals: their
     contents) vs other memories;
  6. samples: flagged turns saying "we" and what was written; memories describing a change or keeping
     an earlier version.
Method: development personas only. Both versions are replayed exactly as in E18 (same code, prompts and
related-memory step), so every call is answered from the cache; the number of fresh calls is printed
and should be 0. A record is identified as in E12-E18 (A10); a memory's records are those of all the
sessions its sources come from (sources include its ancestors').
Limits: pattern-based (dates, "we"); one run per version.
"""
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict

from common import ROOT, flagged_turns, iter_sessions, record_key
from memrecon.embed import Embedder
from memrecon.ingest import Ingestor
from memrecon.llm import client as model_client
from memrecon.memory import MemoryStore
from memrecon.retrieval import Retriever

K = 3
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]
VERSIONS = ["ingest_note_v1", "ingest_note_v2"]
SHORT = {"ingest_note_v1": "v1", "ingest_note_v2": "v2"}
RELATIVE = re.compile(r"\b(today|tonight|tomorrow|yesterday|this (?:morning|afternoon|evening|week|weekend|month|year)|"
                      r"(?:next|last) (?:week|weekend|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
                      r"in \d+ days|in (?:a|one|two|three) (?:days?|weeks?|months?)|within the next|end of (?:next|this) (?:week|month))\b", re.I)
HISTORY = re.compile(r"\b(used to|previously|originally|changed from|was (?:originally|previously)|earlier (?:it|this) was)\b", re.I)
NARRATIVE = re.compile(r"\b(updated|the changes?|was changed|has changed|changed (?:it|the)|revised|modified)\b", re.I)
WE = re.compile(r"\b(we|our)\b", re.I)


class Recorder:
    """Passes calls to the real client and keeps the feedback of every retry."""

    def __init__(self, inner):
        self.inner = inner
        self.feedback = []

    def chat(self, messages, **kw):
        if len(messages) > 2:
            self.feedback.append(messages[-1]["content"])
        return self.inner.chat(messages, **kw)


class RecordingIngestor(Ingestor):
    def _related(self, session, at):
        text, labels = super()._related(session, at)
        self.last_shown = set(labels.values())
        return text, labels


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


def reason(feedback):
    r = feedback.removeprefix("That answer could not be used: ")
    return r.split(" Answer again")[0]


split = json.loads((ROOT / "notes" / "split.json").read_text())
embedder = Embedder()
base = model_client("agent")
res = {v: defaultdict(list) for v in VERSIONS}
stores = {v: {} for v in VERSIONS}
record_of = {}  # (persona, session id) -> record key or None
sessions_by = {}
for persona in split["dev"]:
    sessions = [s for _, _, s in iter_sessions(persona)]
    sessions_by[persona] = sessions
    for s in sessions:
        record_of[(persona, s["session_id"])] = record_key(s) if s.get("operation") in ("add", "update", "delete") else None
    for v in VERSIONS:
        rec = Recorder(base)
        store = MemoryStore(persona, mode="note")
        retriever = Retriever(embedder, k=K)
        ing = RecordingIngestor(rec, store, retriever, persona=persona, tag="E19", note_prompt=v)
        latest = {}  # record key -> latest add/update session id
        for s in sessions:
            k = record_of[(persona, s["session_id"])]
            live = store.active()
            if live:
                retriever.sync(store)
                if s.get("session_type") == "no_memory":
                    for t in s["conversation"]:
                        if t.get("speaker") == "user_agent":
                            res[v]["filler_top1"].append(retriever.scores(t["message"])[0][1])
                if k is not None and s.get("operation") in ("update", "delete") and k in latest:
                    targets = {m.id: m for m in live if any(src.session_id == latest[k] for src in m.sources)}
                    for t in flagged_turns(s):
                        ranked = retriever.scores(t["message"])
                        top = [(mid, sc) for mid, sc in ranked[:K] if mid in targets]
                        best = max((sc for mid, sc in ranked if mid in targets), default=None)
                        words = min((len(m.text.split()) for m in targets.values()), default=None)
                        res[v]["fact"].append({"top": top[0][1] if top else None, "best": best, "words": words})
            ing.last_shown = set()
            out = ing.ingest_note(s)
            res[v]["outcomes"].append((persona, s, out))
            if k is not None and s.get("operation") in ("add", "update"):
                latest[k] = s["session_id"]
        stores[v][persona] = store
        res[v]["feedback"] += rec.feedback
        res[v]["fresh"].append(sum(1 for c in ing.calls if not c["cached"]))


def records(v, persona, mid):
    m = stores[v][persona].get(mid)
    return {record_of.get((persona, src.session_id)) for src in m.sources} - {None}


def text(v, persona, mid):
    return stores[v][persona].get(mid).text


print(f"development personas: {', '.join(split['dev'])}; fresh calls (should be 0): "
      + ", ".join(f"{SHORT[v]} {sum(res[v]['fresh'])}" for v in VERSIONS) + "\n")

print("1. WHY A SECOND ATTEMPT WAS NEEDED (feedback sent to the model)")
for v in VERSIONS:
    c = Counter(reason(f) for f in res[v]["feedback"])
    print(f"  {SHORT[v]}: {sum(c.values())} retries")
    for r, n in c.most_common():
        print(f"      {n:3d}  {r[:160]}")
print()

print(f"2. RETRIEVAL (top {K}): a record's latest memory found by its next update/delete's flagged turns, vs small talk recalling anything")
print(f"  {'threshold':>9} " + " ".join(f"{SHORT[v] + ' found':>9} {SHORT[v] + ' small talk':>13}" for v in VERSIONS))
for thr in THRESHOLDS:
    cells = []
    for v in VERSIONS:
        f, fill = res[v]["fact"], res[v]["filler_top1"]
        cells.append(f"{pct(sum(1 for x in f if x['top'] is not None and x['top'] >= thr), len(f)):>9} {pct(sum(x >= thr for x in fill), len(fill)):>13}")
    print(f"  {thr:9.2f} " + " ".join(cells))
for v in VERSIONS:
    f = res[v]["fact"]
    found = [x["words"] for x in f if x["top"] is not None and x["top"] >= 0.70 and x["words"]]
    missed = [x["words"] for x in f if not (x["top"] is not None and x["top"] >= 0.70) and x["words"]]
    best = [x["best"] for x in f if x["best"] is not None]
    print(f"  {SHORT[v]}: {len(f)} turns, {len(res[v]['filler_top1'])} small-talk turns; best similarity to the record's memory: median "
          f"{st.median(best):.2f}; words in the record's shortest memory: found at 0.70 median {st.median(found) if found else '-'}, "
          f"missed median {st.median(missed) if missed else '-'}")
print()

print("3. CHANGES THAT REPLACE A MEMORY OF ANOTHER RECORD (or in a session about no record)")
unrelated_ex = []
for v in VERSIONS:
    c = Counter()
    for p, s, o in res[v]["outcomes"]:
        own = record_of[(p, s["session_id"])]
        for mid in o.superseded:
            c["replaced"] += 1
            recs = records(v, p, mid)
            if own is None:
                c["in a session about no record"] += 1
                bad = True
            elif own not in recs:
                c["of another record"] += 1
                bad = True
            else:
                bad = False
            if bad and v == "ingest_note_v2":
                new = [m for m in o.created if mid in stores[v][p].get(m).parents]
                unrelated_ex.append((p, s, text(v, p, mid), [text(v, p, m) for m in new]))
    print(f"  {SHORT[v]}: {c['replaced']} memories replaced; of another record {c['of another record']}; "
          f"in a session about no record (small talk or unidentified) {c['in a session about no record']}")
random.seed(19)
print("  v2, 5 at random:")
for p, s, old, new in random.sample(unrelated_ex, min(5, len(unrelated_ex))):
    said = " / ".join(t["message"][:140] for t in flagged_turns(s)) or "(no flagged turns)"
    print(f"  - {p} session {s['session_id']} ({s.get('session_type')}, {s.get('operation')}). User: {said}")
    print(f"      replaced * {old[:300]}")
    for t in new:
        print(f"      with     * {t[:300]}")
print()

print("4. ADD/UPDATE SESSIONS WITH FLAGGED TURNS THAT SAVED NOTHING")
for v in VERSIONS:
    empty = [(p, s) for p, s, o in res[v]["outcomes"] if s.get("operation") in ("add", "update") and flagged_turns(s) and not o.created]
    print(f"  {SHORT[v]}: {len(empty)}")
    for p, s in empty:
        print(f"    - {p} session {s['session_id']} ({s.get('session_type')}, {s.get('operation')}): "
              + " / ".join(t["message"][:160] for t in flagged_turns(s)))
print()

print("5. RELATIVE DATES: memories about documents (their contents) vs other memories")
for v in VERSIONS:
    c = Counter()
    other_ex = []
    for p, s, o in res[v]["outcomes"]:
        doc = (record_of[(p, s["session_id"])] or ("",))[0] == "content"
        for m in o.created:
            tx = text(v, p, m)
            c[("doc" if doc else "other", "n")] += 1
            if RELATIVE.search(tx):
                c[("doc" if doc else "other", "rel")] += 1
                if not doc and len(other_ex) < 5:
                    other_ex.append(tx[:300])
    print(f"  {SHORT[v]}: documents {c[('doc', 'rel')]} of {c[('doc', 'n')]} memories; other {c[('other', 'rel')]} of {c[('other', 'n')]}")
    for tx in other_ex:
        print(f"      other: {tx}")
print()

print('6a. "WE": flagged turns of add/update sessions saying we/our, and what v1 and v2 wrote (6 at random)')
we = [(p, s) for p, s, o in res["ingest_note_v2"]["outcomes"]
      if s.get("operation") in ("add", "update") and any(WE.search(t["message"]) for t in flagged_turns(s))]
by = {v: {(p, s["session_id"]): o for p, s, o in res[v]["outcomes"]} for v in VERSIONS}
print(f"  {len(we)} sessions")
for p, s in random.sample(we, min(6, len(we))):
    print(f"  - {p} session {s['session_id']}. User: " + " / ".join(t["message"][:200] for t in flagged_turns(s) if WE.search(t["message"])))
    for v in VERSIONS:
        for m in by[v][(p, s["session_id"])].created:
            print(f"      {SHORT[v]} * {text(v, p, m)[:300]}")
print()

print("6b. v2 MEMORIES DESCRIBING A CHANGE OR KEEPING AN EARLIER VERSION (6 at random), with what the user said")
pool = [(p, s, text("ingest_note_v2", p, m)) for p, s, o in res["ingest_note_v2"]["outcomes"] for m in o.created
        if HISTORY.search(text("ingest_note_v2", p, m)) or NARRATIVE.search(text("ingest_note_v2", p, m))]
print(f"  {len(pool)} memories")
for p, s, tx in random.sample(pool, min(6, len(pool))):
    print(f"  - {p} session {s['session_id']} ({s.get('session_type')}, {s.get('operation')}). User: "
          + " / ".join(t["message"][:160] for t in flagged_turns(s)))
    print(f"      * {tx[:400]}")
