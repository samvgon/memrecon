"""E32: the ingestion arms scored against the frozen answer key, after every session.

Question (D50): the answer key is frozen (extractor version 4, E30, plus the split pass, E31; hand-checked
179 of 183 rows ok). Scored by code against it after every session, which ingestion arm keeps the told
details best: one-call v1 and v5, two-step v2 and v3 (the arms of E23)? This is the measurement the
ingestion freeze waited on (E23, E24: v2 and v3 tied on the earlier, rougher measures).
Method: the answer key is E30's rows and E31's parts for the development personas, made into a table
(build_table, with D49's closing); after each session, the probes that hold then (true_at) and are graded
by code (grading_view: kinds number, date, name, title, item in states true or false) are checked against
every active memory of each arm (code_match, D50: numbers by value, dates by text, day or duration, short
details by alias as whole words, long ones by content-word overlap >= 0.8 in one memory; D51: a detail
attached to another detail only in a memory that also holds that detail, and a preference no longer
liked closes its reasons and examples in the key). A record any
of whose sessions failed extraction in E30 (flagged lines in an add, update or delete session and no rows)
is left out, since its key may be incomplete. Arms are replayed from E23's cached calls (same prompts,
retrieval and candidate threshold 0.0); no new model calls are expected.
Measures: recall (a probe that should hold is found in some active memory) at the end and averaged over
sessions, by kind and by arm; overlap 0.6 and 1.0 for long details; qualifiers kept with their detail;
old values (state false) still found in an active memory; lost after ingestion (found just after its
session, not at the end); paired counts between arms at the end, with an exact sign test.
Pilot: MEMRECON_E32_PILOT=N replays only the first N sessions of the first persona (not evidence).
Limits: code matching misses rewordings of short details and counts a long detail as kept at 80% of its
words; a probe found in a memory that also says it changed counts as present; tasks, preferences, done
and no-longer entries (judge-graded) are not scored here; probes come from flagged lines only (A2).
"""
import json
import math
import os
import time
from collections import Counter, defaultdict

from common import ROOT, flagged_turns, iter_sessions, record_key
from memrecon.embed import Embedder
from memrecon.ingest import Ingestor
from memrecon.llm import client as model_client
from memrecon.memory import MemoryStore
from memrecon.probes import Row, build_table, grading_view, match_with_parent, parent_entry, qualifier_kept, true_at
from memrecon.retrieval import Retriever

K = 3
ARMS = ["ingest_note_v1", "ingest_note_v5", "two_step_v2", "two_step_v3"]
SHORT = {"ingest_note_v1": "v1", "ingest_note_v5": "v5", "two_step_v2": "2s2", "two_step_v3": "2s3"}
TWO_STEP = {"two_step_v2": ("extract_note_v2", "decide_note_v2"), "two_step_v3": ("extract_note_v2", "decide_note_v3")}
OVERLAPS = (0.6, 0.8, 1.0)
PILOT = int(os.environ.get("MEMRECON_E32_PILOT", "0") or 0)
OUT = ROOT / "results" / "probes"
PROGRESS = ROOT / "results" / "e32_progress.log"


def progress(msg):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def load_rows(name, persona):
    path = OUT / f"{name}_{persona}.jsonl"
    rows = []
    for x in path.read_text(encoding="utf-8").splitlines():
        if x.strip():
            d = json.loads(x)
            d.pop("persona", None)
            d["record"] = tuple(d["record"])
            rows.append(Row(**d))
    return rows


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


def sign_test(b, c):
    """Two-sided exact binomial p for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def probe_key(e):
    return e["row_id"] or f"{e['record']}|{e['field']}|{e['detail']}|{e['from_session']}"


split = json.loads((ROOT / "notes" / "split.json").read_text())
personas = split["dev"][:1] if PILOT else split["dev"]
embedder = Embedder()
agent = model_client("agent")
start = time.monotonic()

# -- the answer key ------------------------------------------------------------------------------
key, excluded, sessions_of = {}, {}, {}
for persona in personas:
    rows = load_rows("E30", persona) + load_rows("E31", persona)
    sessions = [s for _, _, s in iter_sessions(persona)]
    if PILOT:
        sessions = sessions[:PILOT]
    sessions_of[persona] = sessions
    with_rows = {r.session for r in rows}
    failed = {s["session_id"] for s in sessions if s.get("operation") in ("add", "update", "delete")
              and flagged_turns(s) and s["session_id"] not in with_rows}
    bad = set()  # a failed session has no rows: its record is identified as in E30 (A10)
    for s in sessions:
        if s["session_id"] in failed:
            k = record_key(s)
            bad.add(k if k is not None else ("session", s["session_id"]))
    excluded[persona] = (failed, bad)
    key[persona] = build_table([r for r in rows if r.record not in bad])

# -- replay the arms and score after every session -------------------------------------------------
score = {a: defaultdict(list) for a in ARMS}  # arm -> lists of per-probe results
calls = Counter()
for persona in personas:
    sessions = sessions_of[persona]
    entries = key[persona]
    for arm in ARMS:
        store = MemoryStore(persona, mode="note")
        retriever = Retriever(embedder, k=K)
        ext, dec = TWO_STEP.get(arm, ("extract_note_v1", "decide_note_v1"))
        ing = Ingestor(agent, store, retriever, persona=persona, tag="E32", candidate_threshold=0.0,
                       note_prompt=arm if arm not in TWO_STEP else "ingest_note_v5", extract_prompt=ext, decide_prompt=dec)
        cache = {}
        by_row = {e["row_id"]: e for e in entries if e.get("row_id")}  # a detail's parent, to find them together (D51)
        first_seen = {}  # probe key -> found right after the session it was told in
        last = sessions[-1]["session_id"]
        for s in sessions:
            if store.active():
                retriever.sync(store)
            out = ing.ingest_two_step(s) if arm in TWO_STEP else ing.ingest_note(s)
            sid = s["session_id"]
            texts = [m.text for m in store.active()]
            probes = [p for p in grading_view(true_at(entries, sid)) if p["grade"] == "code"]
            n_true = n_found = 0
            for p in probes:
                pk = probe_key(p)
                found = {}
                for ov in OVERLAPS:
                    hit = None
                    for t in texts:
                        c = (pk, t, ov)
                        if c not in cache:
                            cache[c] = match_with_parent(p, t, parent_entry(p, by_row), ov)
                        if cache[c]:
                            hit = t
                            break
                    found[ov] = hit
                present = {ov: found[ov] is not None for ov in OVERLAPS}
                if p["state"] == "true":
                    n_true += 1
                    n_found += present[0.8]
                    if p["from_session"] == sid:
                        first_seen[pk] = present[0.8]
                if sid == last:
                    score[arm]["end"].append({
                        "persona": persona, "probe": pk, "kind": p["kind"], "state": p["state"], "present": present,
                        "qualifier": qualifier_kept(p, found[0.8]) if found[0.8] else (False if p.get("qualifier") else None),
                        "detail": p["detail"], "long": len(p["detail"].split()) >= 5, "first": first_seen.get(pk)})
            score[arm]["per_session"].append((persona, sid, n_true, n_found))
            if sid == last:  # values that held once and were replaced or removed by the end
                gone = [p for p in grading_view([e for e in entries if e["state"] == "true" and e["until_session"] is not None])
                        if p["grade"] == "code"]
                for p in gone:
                    score[arm]["old"].append({"persona": persona, "detail": p["detail"], "kind": p["kind"],
                                              "found": any(match_with_parent(p, t, parent_entry(p, by_row)) for t in texts)})
            progress(f"{persona} {SHORT[arm]} session {sid}: ok={out.ok} active={len(texts)} probes={len(probes)}")
        calls.update(("fresh" if not c["cached"] else "cached") for c in ing.calls)
elapsed = time.monotonic() - start

if PILOT:
    print(f"PILOT RUN: not evidence. {sum(len(v) for v in sessions_of.values())} sessions x {len(ARMS)} arms in {elapsed:.0f} s; calls {dict(calls)}\n")

print("1. THE ANSWER KEY (development personas)")
for persona in personas:
    failed, bad = excluded[persona]
    e = key[persona]
    gv = grading_view(e)
    print(f"  {persona}: {len(e)} table entries; code-graded {sum(1 for x in gv if x['grade'] == 'code')}, judge-graded "
          f"{sum(1 for x in gv if x['grade'] == 'judge')}; failed extraction sessions {sorted(failed)}, records left out {len(bad)}")
print(f"  model calls in the replay: {dict(calls)} (fresh calls mean the cache did not cover a session)")
print(f"  wall time {elapsed:.0f} s")
print()

print("2. RECALL AT THE END (probes that should hold, found in some active memory; long details at overlap 0.8)")
print(f"  {'arm':<5} {'probes':>7} {'recall':>8}   by kind")
for a in ARMS:
    end = [r for r in score[a]["end"] if r["state"] == "true"]
    kinds = Counter(r["kind"] for r in end)
    print(f"  {SHORT[a]:<5} {len(end):7d} {pct(sum(r['present'][0.8] for r in end), len(end)):>8}   "
          + ", ".join(f"{k} {pct(sum(r['present'][0.8] for r in end if r['kind'] == k), n).strip()} of {n}" for k, n in kinds.most_common()))
print("  long details (5+ words) at overlap 0.6 / 0.8 / 1.0: " + "; ".join(
    f"{SHORT[a]} " + " / ".join(pct(sum(r['present'][ov] for r in score[a]['end'] if r['state'] == 'true' and r['long']),
                                   sum(1 for r in score[a]['end'] if r['state'] == 'true' and r['long'])).strip() for ov in OVERLAPS) for a in ARMS))
print()

print("3. RECALL OVER TIME (each session's probes that hold then; mean over sessions, and by thirds of the history)")
for a in ARMS:
    per = [((p_, sid), f / n) for p_, sid, n, f in score[a]["per_session"] if n]
    thirds = []
    for persona in personas:
        ps = [x for (p, _), x in per if p == persona]
        n = len(ps)
        thirds.append([ps[i * n // 3:(i + 1) * n // 3] for i in range(3)])
    t = [sum(sum(th[i]) for th in thirds) / max(1, sum(len(th[i]) for th in thirds)) for i in range(3)]
    print(f"  {SHORT[a]:<5} mean {sum(x for _, x in per) / max(1, len(per)):.3f}; first / middle / last third {t[0]:.3f} / {t[1]:.3f} / {t[2]:.3f}")
print()

print("4. KEPT FROM INGESTION TO THE END (probes found right after the session they were told in: still found at the end?)")
for a in ARMS:
    end = [r for r in score[a]["end"] if r["state"] == "true" and r.get("first") is not None]
    then = [r for r in end if r["first"]]
    print(f"  {SHORT[a]:<5} found when told {pct(len(then), len(end))} of {len(end)}; of those, still found at the end {pct(sum(r['present'][0.8] for r in then), len(then))}"
          f" ({sum(1 for r in then if not r['present'][0.8])} lost)")
print()

print("5. QUALIFIERS (numbers and dates with a graded qualifier, at the end: kept in the memory that holds the detail)")
for a in ARMS:
    q = [r for r in score[a]["end"] if r["state"] == "true" and r["qualifier"] is not None]
    print(f"  {SHORT[a]:<5} {sum(1 for r in q if r['qualifier'])} of {len(q)}")
print()

print("6. OLD VALUES (values that held once and were updated or removed by the end, still found in an active memory;")
print("   a memory may keep one as history, 'changed from $X', so this is not by itself an error)")
for a in ARMS:
    old = score[a]["old"]
    false_now = [r for r in score[a]["end"] if r["state"] == "false"]
    print(f"  {SHORT[a]:<5} {sum(r['found'] for r in old)} of {len(old)}; values only ever told as old ('replaced', 'removed' with no earlier row) "
          f"found {sum(r['present'][0.8] for r in false_now)} of {len(false_now)}")
print()

print("7. PAIRED AT THE END (probes that should hold: found by one arm and not the other; exact sign test)")
ends = {a: {(r["persona"], r["probe"]): r["present"][0.8] for r in score[a]["end"] if r["state"] == "true"} for a in ARMS}
common_keys = set.intersection(*(set(v) for v in ends.values()))
for i, a in enumerate(ARMS):
    for b in ARMS[i + 1:]:
        only_a = sum(1 for k in common_keys if ends[a][k] and not ends[b][k])
        only_b = sum(1 for k in common_keys if ends[b][k] and not ends[a][k])
        print(f"  {SHORT[a]} vs {SHORT[b]}: only {SHORT[a]} {only_a}, only {SHORT[b]} {only_b}, p = {sign_test(only_a, only_b):.3g}")
print("  (probes are not independent: several come from one session or record, so p is descriptive)")
print()

print("8. EXAMPLES: probes the best and worst arm disagree on at the end (up to 8 each way)")
rank = sorted(ARMS, key=lambda a: -sum(ends[a][k] for k in common_keys))
best, worst = rank[0], rank[-1]
detail_of = {(r["persona"], r["probe"]): (r["kind"], r["detail"]) for r in score[best]["end"]}
for x, y in ((best, worst), (worst, best)):
    keys_ = [k for k in sorted(common_keys) if ends[x][k] and not ends[y][k]][:8]
    print(f"  found by {SHORT[x]}, not by {SHORT[y]}:")
    for k in keys_:
        print(f"    - {k[0]} [{detail_of[k][0]}] {detail_of[k][1][:120]}")
