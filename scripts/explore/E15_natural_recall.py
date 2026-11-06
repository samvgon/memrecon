"""E15: with real retrieval, how often would facts be recalled naturally during the replay?

Question (D14, A9, power): E12 counted later mentions by strict string matching, a lower bound. With
embedding retrieval, how many times is each fact recalled by later user turns, and does that build
enough rewrite depth, or is the injected-query arm needed? Also: where should the relevance threshold
sit (D14 says it is set on development personas)?
Method: development personas only (notes/split.json). Replay each persona's sessions in file order
(A6, supported by E13). Memories are the log arm's: each flagged turn of an add or update session,
verbatim, becomes a memory of its fact at the end of that session (A2). Memories of a record are
dropped when a later delete session concerns the same record (A10); the log arm is otherwise
append-only (D29), so an update adds memories rather than replacing them. Every user turn of every
later session is a query. A fact is recalled by a turn if one of its memories is among the k = 3 most
similar live memories and its cosine similarity reaches the threshold; results are shown for several
thresholds. Embeddings: the `embed` role (nomic-embed-text-v1.5 with its search_query /
search_document prefixes), cached in results/embeddings.sqlite.
Two reference sets help place the threshold:
  hits  - turns of later update or delete sessions of the same record (the user is clearly talking
          about that fact): the share of those sessions in which the fact is recalled at least once;
  filler- user turns of no_memory sessions (small talk): the share of those turns that recall anything.
Recalls in a fact's own later update/delete sessions are reported separately from the other ("natural")
recalls, as in E12.
Limits: log-arm memories (verbatim turns) stand in for notes, whose wording would differ; the agent's
own replies are not used as queries; top-k and the thresholds are a grid, not a choice.
"""
import json
import statistics as st
from collections import Counter, defaultdict

import numpy as np

from common import ROOT, flagged_turns, iter_sessions
from memrecon.embed import Embedder

K = 3
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
CONTENT_KEYS = ["project_title", "meeting_title", "email_subject", "email_purpose"]


def record_key(s):
    """Same record identity as E14 (A10)."""
    d = s.get("operation_details") or {}
    t = s.get("session_type")
    if t == "preference":
        item = d.get("item")
        return ("preference", item.strip().lower()) if isinstance(item, str) else None
    if t != "activity":
        return None
    cat, it = d.get("category"), d.get("item")
    if cat is None and isinstance(it, str):
        return ("content", it)
    if not isinstance(it, dict):
        return None
    if cat == "todo_list" and it.get("description"):
        return (cat, it["description"].strip().lower())
    if cat == "calendar_event" and it.get("event_name"):
        return (cat, it["event_name"].strip().lower())
    if cat == "food_expenses" and it.get("amount") is not None:
        return (cat, f"{float(it['amount']):.2f}")
    if cat == "step_tracker" and it.get("step_count") is not None:
        return (cat, str(int(it["step_count"])))
    return None


def fact_type(s):
    d = s.get("operation_details") or {}
    if s.get("session_type") == "activity":
        return "activity/" + (d.get("category") or "content_record")
    return s.get("session_type") or "?"


def run(persona, embedder):
    sessions = [s for _, _, s in iter_sessions(persona)]
    keys = [record_key(s) if s.get("operation") in ("add", "update", "delete") else None for s in sessions]

    # facts and their memories (flagged turns of add/update sessions)
    facts = []  # dicts: session index, key, type, memory texts
    for i, s in enumerate(sessions):
        if s.get("operation") in ("add", "update"):
            turns = [t["message"] for t in flagged_turns(s)]
            if turns:
                facts.append({"i": i, "key": keys[i], "type": fact_type(s), "texts": turns})
    mem_text = [t for f in facts for t in f["texts"]]
    mem_fact = [fi for fi, f in enumerate(facts) for _ in f["texts"]]
    mem_vec = np.asarray(embedder.embed(mem_text, "document"), dtype=np.float32)

    queries = [(i, t["message"]) for i, s in enumerate(sessions) for t in s["conversation"] if t.get("speaker") == "user_agent"]
    q_vec = np.asarray(embedder.embed([q for _, q in queries], "query"), dtype=np.float32)

    # when each memory is live: from the end of its fact's session until a delete of the same record
    born = np.array([facts[fi]["i"] for fi in mem_fact])
    died = np.full(len(mem_fact), len(sessions))
    for m, fi in enumerate(mem_fact):
        k = facts[fi]["key"]
        if k is None:
            continue
        for j in range(facts[fi]["i"] + 1, len(sessions)):
            if keys[j] == k and sessions[j].get("operation") == "delete":
                died[m] = j
                break

    mems_of = defaultdict(list)
    for m, fi in enumerate(mem_fact):
        mems_of[fi].append(m)
    facts_by_key = defaultdict(list)
    for fi, f in enumerate(facts):
        if f["key"] is not None:
            facts_by_key[f["key"]].append(fi)

    def own_session(fi, si):
        """Is session si a later update or delete of fact fi's record?"""
        f = facts[fi]
        return f["key"] is not None and keys[si] == f["key"] and si > f["i"] and sessions[si].get("operation") in ("update", "delete")

    natural = {thr: Counter() for thr in THRESHOLDS}  # fact -> recalls outside its own update/delete sessions
    own = {thr: Counter() for thr in THRESHOLDS}  # fact -> recalls inside them
    own_sessions_hit = {thr: set() for thr in THRESHOLDS}
    own_sessions = set()
    filler_turns = 0
    filler_recalls = Counter()
    top1_filler, top1_own = [], []

    for qi, (si, _) in enumerate(queries):
        live = np.where((born < si) & (died > si))[0]
        if len(live) == 0:
            continue
        sims = mem_vec[live] @ q_vec[qi]
        order = np.argsort(-sims, kind="stable")[:K]
        top = [(int(live[o]), float(sims[o])) for o in order]
        is_filler = sessions[si].get("session_type") == "no_memory"
        if is_filler:
            filler_turns += 1
            top1_filler.append(top[0][1])
        recalled_here = {}
        for m, sim in top:
            fi = mem_fact[m]
            recalled_here[fi] = max(sim, recalled_here.get(fi, -1.0))
        for fi in facts_by_key.get(keys[si], []) if keys[si] is not None else []:
            if own_session(fi, si):
                own_sessions.add((fi, si))
                top1_own.append(float(max(mem_vec[m] @ q_vec[qi] for m in mems_of[fi])))
        for thr in THRESHOLDS:
            hit_any = False
            for fi, sim in recalled_here.items():
                if sim < thr:
                    continue
                hit_any = True
                if own_session(fi, si):
                    own[thr][fi] += 1
                    own_sessions_hit[thr].add((fi, si))
                else:
                    natural[thr][fi] += 1
            if is_filler and hit_any:
                filler_recalls[thr] += 1
    return facts, natural, own, own_sessions, own_sessions_hit, filler_turns, filler_recalls, top1_filler, top1_own


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


split = json.loads((ROOT / "notes" / "split.json").read_text())
embedder = Embedder()
all_facts = 0
agg = {thr: [] for thr in THRESHOLDS}
agg_type = {thr: defaultdict(list) for thr in THRESHOLDS}
own_hit = {thr: 0 for thr in THRESHOLDS}
own_n = 0
filler_n = 0
filler_hit = Counter()
t1f, t1o = [], []
for persona in split["dev"]:
    facts, natural, own, own_s, own_s_hit, fn, fr, a, b = run(persona, embedder)
    all_facts += len(facts)
    own_n += len(own_s)
    filler_n += fn
    t1f += a
    t1o += b
    for thr in THRESHOLDS:
        counts = [natural[thr][fi] for fi in range(len(facts))]
        agg[thr] += counts
        for fi, f in enumerate(facts):
            agg_type[thr][f["type"]].append(natural[thr][fi])
        own_hit[thr] += len(own_s_hit[thr])
        filler_hit[thr] += fr[thr]
    print(f"{persona}: {len(facts)} facts, {sum(len(f['texts']) for f in facts)} memories")

print(f"\ndevelopment personas: {all_facts} facts; k = {K}; {filler_n} filler turns; {own_n} (fact, own update/delete session) pairs")
print(f"embedding calls this run: {embedder.calls} (cached texts are not re-sent)\n")
print("Similarity of the best memory, for reference:")
for name, xs in (("filler turns, top-1 over all live memories", t1f), ("own update/delete turns, best memory of that fact", t1o)):
    if xs:
        q = np.percentile(xs, [10, 25, 50, 75, 90])
        print(f"  {name:<52} p10 {q[0]:.2f}  p25 {q[1]:.2f}  median {q[2]:.2f}  p75 {q[3]:.2f}  p90 {q[4]:.2f}")
print()
print(f"{'threshold':>9} | natural recalls per fact: {'>=1':>7} {'>=3':>7} {'>=5':>7} {'>=10':>7} {'mean':>6} {'median':>6} | {'own-session hit':>15} | {'filler turns recalling':>22}")
for thr in THRESHOLDS:
    c = agg[thr]
    n = len(c)
    print(
        f"{thr:9.2f} | {'':25} {pct(sum(x >= 1 for x in c), n):>7} {pct(sum(x >= 3 for x in c), n):>7} "
        f"{pct(sum(x >= 5 for x in c), n):>7} {pct(sum(x >= 10 for x in c), n):>7} {st.mean(c):6.2f} {st.median(c):6.1f} | "
        f"{pct(own_hit[thr], own_n):>15} | {pct(filler_hit[thr], filler_n):>22}"
    )
print("\nnatural = recalls by later turns outside the fact's own update/delete sessions;")
print("own-session hit = share of a fact's later update/delete sessions in which it is recalled at least once;")
print("filler turns recalling = share of small-talk turns that recall any memory (likely false recalls).")
print()
for thr in (0.60, 0.70):
    print(f"By fact type at threshold {thr:.2f} (share with >=1 and >=5 natural recalls, mean):")
    for t, c in sorted(agg_type[thr].items()):
        print(f"  {t:<28} n={len(c):4d}  >=1 {pct(sum(x >= 1 for x in c), len(c))}  >=5 {pct(sum(x >= 5 for x in c), len(c))}  mean {st.mean(c):5.2f}")
    print()
