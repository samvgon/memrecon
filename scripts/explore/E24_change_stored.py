"""E24: are the user's changes stored? And what are decide v3's identical replaces?

Question (D43, choosing between decide v2 and v3 before freezing ingestion): in E23, decide v3 made 19
replaces whose text was identical to the memory replaced (D42 records them as "none", so the fact is not
stored), 11 of them in update sessions, against 1 for decide v2. If such a fact carried the change itself
("The user added 'Data Science Lead' to the stakeholders"), the change was silently dropped. More
generally: right after each update session, are the details of what the user changed in an active memory?
For two-step v2 (extract v2 + decide v2) and v3 (extract v2 + decide v3), on the development personas:
  1. change stored: for each update session, the candidate elements (E14) of its flagged turns, and the
     share present in an active memory right after the session; sessions with any element missing, listed;
  2. every identical replace of each version: the session, the fact, the memory it left unchanged, and
     whether the fact's own elements are in an active memory right after the session.
Method: both versions are replayed exactly as in E23 (same code, prompts, retriever, threshold 0.0), so
every call is answered from the cache; the number of fresh calls is printed and should be 0.
Limits: string and value matching (E18's element check), so a paraphrased detail counts as missing;
a preference update names the old item as well as the new, and the old need not be kept (shown apart);
an element can be present because an older memory already had it (a removal is not checked here).
"""
import json
import re

from common import ROOT, flagged_turns, iter_sessions, probe_elements, record_value
from memrecon.embed import Embedder
from memrecon.ingest import Ingestor
from memrecon.llm import client as model_client
from memrecon.memory import MemoryStore
from memrecon.retrieval import Retriever

VERSIONS = {"2s2": ("extract_note_v2", "decide_note_v2"), "2s3": ("extract_note_v2", "decide_note_v3")}

MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mil": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
NUM = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(k|thousand|million|mil|bn|billion|m|b)?\b", re.I)


def values(text):
    return {round(float(m.group(1).replace(",", "")) * MULT.get((m.group(2) or "").lower(), 1), 4) for m in NUM.finditer(text)}


def element_kept(kind, value, memory_text):
    """E18's check (numbers also compared by value)."""
    if kind == "number":
        core = re.sub(r"[^\d.]", "", value.replace(",", "")).strip(".")
        if core and core in memory_text.replace(",", ""):
            return True
        want = values(value)
        return bool(want) and want <= values(memory_text)
    v = re.sub(r"^[\"“*'\s]+|[\"”*'.,!?:;\s]+$", "", value).lower()
    return bool(v) and v in memory_text.lower()


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


split = json.loads((ROOT / "notes" / "split.json").read_text())
embedder = Embedder()
agent = model_client("agent")
res = {v: {"sessions": [], "identical": [], "fresh": 0} for v in VERSIONS}
for persona in split["dev"]:
    sessions = [s for _, _, s in iter_sessions(persona)]
    for v, (ext, dec) in VERSIONS.items():
        store = MemoryStore(persona, mode="note")
        ing = Ingestor(agent, store, Retriever(embedder, k=3), persona=persona, tag="E24",
                       note_prompt="ingest_note_v5", candidate_threshold=0.0, extract_prompt=ext, decide_prompt=dec)
        for s in sessions:
            out = ing.ingest_two_step(s)
            active = " ".join(m.text for m in store.active())
            if s.get("operation") == "update":
                elems = [(k, x) for t in flagged_turns(s) for k, x in probe_elements(t["message"], record_value(s))]
                res[v]["sessions"].append((persona, s, elems, [x for k, x in elems if not element_kept(k, x, active)]))
            for d in ing.decision_log:
                if d["action"] == "none (identical replace)":
                    fe = probe_elements(d["fact"])
                    res[v]["identical"].append((persona, s, d["fact"], [store.get(mid).text for mid in d.get("unchanged", [])],
                                                [x for k, x in fe if not element_kept(k, x, active)], len(fe)))
        res[v]["fresh"] += sum(1 for c in ing.calls if not c["cached"])

print("development personas: " + ", ".join(split["dev"]) + "; fresh calls (should be 0): "
      + ", ".join(f"{v} {res[v]['fresh']}" for v in VERSIONS) + "\n")

print("1. CHANGE STORED: update sessions, candidate elements of the flagged turns in an active memory right after")
print("  (preference updates name the old item too, which need not be kept, so they are shown apart)")
for v in VERSIONS:
    for kind in ("activity", "preference", "other"):
        ss = [x for x in res[v]["sessions"] if (x[1].get("session_type") if x[1].get("session_type") in ("activity", "preference") else "other") == kind]
        if not ss:
            continue
        n = sum(len(e) for _, _, e, _ in ss)
        miss = sum(len(m) for _, _, _, m in ss)
        print(f"  {v} {kind:<10}: {len(ss)} sessions, {n} elements, present {pct(n - miss, n)}; sessions with an element missing {sum(1 for *_, m in ss if m)}")
for v in VERSIONS:
    print(f"  {v}, sessions with an element missing:")
    for p, s, e, m in res[v]["sessions"]:
        if m:
            print(f"    - {p} session {s['session_id']}: missing {m}. User: " + " / ".join(t["message"][:150] for t in flagged_turns(s)))
print()

print("2. IDENTICAL REPLACES (recorded as none by D42): the fact, and whether its elements are in an active memory right after")
for v in VERSIONS:
    ids = res[v]["identical"]
    print(f"  {v}: {len(ids)}; facts with an element missing afterwards: {sum(1 for x in ids if x[4])}")
    for p, s, fact, cands, missing, n in ids:
        tag = f"MISSING {missing}" if missing else (f"all {n} elements present" if n else "no checkable elements")
        print(f"    - {p} session {s['session_id']} ({s.get('session_type')}, {s.get('operation')}): {tag}")
        print(f"        fact:  {fact[:250]}")
        print(f"        left unchanged: {cands[0][:250] if cands else '-'}")
