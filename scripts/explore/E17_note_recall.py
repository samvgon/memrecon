"""E17: with note memories and the chosen threshold, how often would each item be recalled naturally?

Question (D14, A9, power; threshold 0.70 chosen from E16): E15 counted natural recalls on verbatim
memories. With the note arm's memories, how many times is each item recalled by later user turns
during the replay, so how much rewrite depth would natural recall give, and is the injected-query arm
needed? Also, for writing ingestion prompt v2: what do split items and notes made in delete
sessions look like?
Method: development personas only. The note arm's ingestion is replayed exactly as in E16 (same
prompt ingest_note_v1, same related-memory step), so every ingestion call is answered from the cache
and no new model calls are made. This is the discard condition: recall never changes memory. Before
each session is ingested, each of its user turns is a query; a memory is recalled if it is among the
top 3 and reaches the threshold (0.65, 0.70 and 0.75 are shown; 0.70 is the chosen one). A recall
counts for the item (record, A10) that the memory's sources come from. Recalls in the item's own
later update or delete sessions are counted apart from the others ("natural"). Recalls by small-talk
turns are natural recalls too (a real system would make them), and are shown separately.
Limits: discard only (under replace, a rewritten memory's text, and so what it matches, changes);
v1 notes; ingestion's own related-memory step used no threshold, as in E16.
"""
import json
import random
import statistics as st
from collections import Counter, defaultdict

from common import ROOT, iter_sessions, record_key
from memrecon.embed import Embedder
from memrecon.ingest import Ingestor
from memrecon.llm import client as model_client
from memrecon.memory import MemoryStore
from memrecon.retrieval import Retriever

K = 3
THRESHOLDS = [0.65, 0.70, 0.75]
CHOSEN = 0.70


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


split = json.loads((ROOT / "notes" / "split.json").read_text())
embedder = Embedder()
agent = model_client("agent")
natural = {thr: Counter() for thr in THRESHOLDS}
natural_small_talk = {thr: Counter() for thr in THRESHOLDS}
own = {thr: Counter() for thr in THRESHOLDS}
recalls_per_session = {thr: [] for thr in THRESHOLDS}
items = {}  # item -> fact type
fresh_calls = 0
splits, delete_notes = [], []

for persona in split["dev"]:
    sessions = [s for _, _, s in iter_sessions(persona)]
    keys = [record_key(s) if s.get("operation") in ("add", "update", "delete") else None for s in sessions]
    item_of_session = {}  # session_id -> item id (record key, or the session itself when there is none)
    store = MemoryStore(persona, mode="note")
    retriever = Retriever(embedder, k=K)
    ingestor = Ingestor(agent, store, retriever, persona=persona, tag="E17")
    for i, s in enumerate(sessions):
        if s.get("operation") in ("add", "update"):
            item = (persona, keys[i]) if keys[i] is not None else (persona, ("session", s["session_id"]))
            item_of_session[s["session_id"]] = item
            items.setdefault(item, s.get("session_type"))
        # recall, before this session is ingested (memories from earlier sessions only)
        if store.active():
            retriever.sync(store)
            own_item = (persona, keys[i]) if keys[i] is not None and s.get("operation") in ("update", "delete") else None
            n_rec = Counter()
            for t in s["conversation"]:
                if t.get("speaker") != "user_agent":
                    continue
                ranked = retriever.scores(t["message"])[:K]
                for thr in THRESHOLDS:
                    hit_items = set()
                    for mid, score in ranked:
                        if score < thr:
                            continue
                        n_rec[thr] += 1
                        for src in store.get(mid).sources:
                            it = item_of_session.get(src.session_id)
                            if it is not None:
                                hit_items.add(it)
                    for it in hit_items:
                        if it == own_item:
                            own[thr][it] += 1
                        else:
                            natural[thr][it] += 1
                            if s.get("session_type") == "no_memory":
                                natural_small_talk[thr][it] += 1
            for thr in THRESHOLDS:
                recalls_per_session[thr].append(n_rec[thr])
        out = ingestor.ingest_note(s)
        if s.get("operation") in ("add", "update") and len(out.created) >= 2:
            splits.append((persona, s, [store.get(m).text for m in out.created]))
        if s.get("operation") == "delete" and out.created:
            delete_notes.append((persona, s, [store.get(m).text for m in out.created], len(out.forgotten)))
    fresh_calls += sum(1 for c in ingestor.calls if not c["cached"])

print(f"development personas; {len(items)} items (records told about in add/update sessions); k = {K}")
print(f"ingestion calls not found in the cache (should be 0): {fresh_calls}\n")
print("Natural recalls per item over the replay (outside the item's own update/delete sessions):")
print(f"{'threshold':>9} {'>=1':>7} {'>=3':>7} {'>=5':>7} {'>=10':>7} {'mean':>6} {'median':>6} | {'from small talk':>15} | {'recalls per session':>19}")
for thr in THRESHOLDS:
    c = [natural[thr][it] for it in items]
    st_share = sum(natural_small_talk[thr].values()) / max(1, sum(natural[thr].values()))
    rps = recalls_per_session[thr]
    mark = "  <- chosen" if thr == CHOSEN else ""
    print(f"{thr:9.2f} {pct(sum(x >= 1 for x in c), len(c)):>7} {pct(sum(x >= 3 for x in c), len(c)):>7} {pct(sum(x >= 5 for x in c), len(c)):>7} "
          f"{pct(sum(x >= 10 for x in c), len(c)):>7} {st.mean(c):6.2f} {st.median(c):6.1f} | {100 * st_share:14.1f}% | "
          f"mean {st.mean(rps):.2f}, max {max(rps)}{mark}")
print("('from small talk' = share of natural recalls made by turns of no_memory sessions;")
print(" 'recalls per session' = memories recalled per session, i.e. rewrites a replace arm would make)\n")

print(f"By fact type at {CHOSEN:.2f}: share of items with >=1 / >=5 natural recalls, mean")
by_type = defaultdict(list)
for it, t in items.items():
    by_type[t].append(natural[CHOSEN][it])
for t, c in sorted(by_type.items()):
    print(f"  {t:<12} n={len(c):4d}  >=1 {pct(sum(x >= 1 for x in c), len(c))}  >=5 {pct(sum(x >= 5 for x in c), len(c))}  mean {st.mean(c):5.2f}")
print()

random.seed(17)
print(f"Split items: {len(splits)} add/update sessions gave 2+ memories. 8 at random:")
for persona, s, texts in random.sample(splits, min(8, len(splits))):
    print(f"- {persona} session {s['session_id']} ({s.get('session_type')}, {s.get('operation')}): {len(texts)} memories")
    for t in texts:
        print(f"    * {t[:200]}")
print()
print(f"Delete sessions that produced notes: {len(delete_notes)}. 8 at random (with what the user said):")
for persona, s, texts, n_forgot in random.sample(delete_notes, min(8, len(delete_notes))):
    said = " / ".join(t["message"][:120] for t in s["conversation"] if t.get("share_memory"))
    kind = (s.get("operation_details") or {}).get("category") or s.get("session_type")
    print(f"- {persona} session {s['session_id']} ({kind}); forgot {n_forgot}. User: {said}")
    for t in texts:
        print(f"    * {t[:200]}")
