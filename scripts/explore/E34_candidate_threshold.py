"""E34: two-step v3 with a candidate threshold of 0.70, against v1 and two-step v3 at 0.0.

Question (E32 and E33 after D51): one-call v1 and two-step v3 tie on recall at the end (85.0% vs 83.5%).
Two-step v3 finds more details when told (94.6% vs 91.9%) but loses more later (70 vs 43), 38 of them
document details superseded while the user added another document: step 2 is shown the 5 most similar
memories however low their similarity (candidate threshold 0.0) and sometimes replaces one. Sam asked
for a candidate threshold to tune when two-step was chosen (D41). At 0.70, E23 measured that step 2
still sees 87% of the candidates of the fact's own record and 38% of the others. With it, does two-step
v3 keep its capture and lose less, and at what cost (updates that no longer find the memory they change,
so old values stay)?
Method: the frozen key and grading of E32 (D50, D51), replaying three arms: v1 and two-step v3 at
threshold 0.0 from E23's cache, and two-step v3 at 0.70, whose step-1 calls are cached and whose step-2
calls are new where the candidates differ (a fact with no candidate at or above 0.70 is added with no
call). Reported as in E32 (recall at the end and over time, found when told, lost, qualifiers, old values,
paired counts) and E33 (cause of each loss, old values by memory), plus step 2's decisions per arm.
Threshold: MEMRECON_E34_THRESHOLD (default 0.70).
Pilot: MEMRECON_E34_PILOT=N replays only the first N sessions of the first persona and prints timings
(not evidence). Progress goes to results/e34_progress.log.
Limits: as E32 and E33; one threshold, chosen from E23's similarities, not tuned on these results.
"""
import json
import math
import os
import re
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
THRESHOLD = float(os.environ.get("MEMRECON_E34_THRESHOLD", "0.70") or 0.70)
ARMS = [("v1", "ingest_note_v1", 0.0), ("2s3", "two_step_v3", 0.0), (f"2s3@{THRESHOLD:.2f}", "two_step_v3", THRESHOLD)]
TWO_STEP = {"two_step_v3": ("extract_note_v2", "decide_note_v3")}
PILOT = int(os.environ.get("MEMRECON_E34_PILOT", "0") or 0)
OUT = ROOT / "results" / "probes"
PROGRESS = ROOT / "results" / "e34_progress.log"
HISTORY = re.compile(r"\b(used to|previously|originally|formerly|changed from|from \S+ to|updated from|was updated|was changed|"
                     r"instead of|no longer|replac\w+|revised|old|earlier|removed|deleted|dropped)\b", re.I)


def progress(msg):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


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


def rtype(record):
    return {"preference": "preferences", "content": "documents", "food_expenses": "expenses and steps",
            "step_tracker": "expenses and steps", "todo_list": "to-dos and calendar", "calendar_event": "to-dos and calendar",
            "goal": "goals"}.get(record[0] if record else "", "other")


def probe_key(e):
    return e["row_id"] or f"{e['record']}|{e['field']}|{e['detail']}|{e['from_session']}"


def short(t, n=150):
    t = " ".join(t.split())
    return t if len(t) <= n else t[:n - 1] + "…"


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


def sign_test(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


split = json.loads((ROOT / "notes" / "split.json").read_text())
personas = split["dev"][:1] if PILOT else split["dev"]
embedder = Embedder()
agent = model_client("agent")
start = time.monotonic()
R = {name: defaultdict(list) for name, _, _ in ARMS}
calls = {name: Counter() for name, _, _ in ARMS}
seconds = defaultdict(list)

for persona in personas:
    sessions = [s for _, _, s in iter_sessions(persona)]
    if PILOT:
        sessions = sessions[:PILOT]
    rows = load_rows("E30", persona) + load_rows("E31", persona)
    with_rows = {r.session for r in rows}
    bad = {record_of(s) for s in sessions if s.get("operation") in ("add", "update", "delete") and flagged_turns(s)
           and s["session_id"] not in with_rows}
    entries = build_table([r for r in rows if r.record not in bad])
    by_row = {e["row_id"]: e for e in entries if e.get("row_id")}
    rec_of_session = {s["session_id"]: record_of(s) for s in sessions}
    last = sessions[-1]["session_id"]
    for name, arm, thr in ARMS:
        store = MemoryStore(persona, mode="note")
        retriever = Retriever(embedder, k=K)
        ext, dec = TWO_STEP.get(arm, ("extract_note_v1", "decide_note_v1"))
        ing = Ingestor(agent, store, retriever, persona=persona, tag="E34", candidate_threshold=thr,
                       note_prompt=arm if arm not in TWO_STEP else "ingest_note_v5", extract_prompt=ext, decide_prompt=dec)
        cache, holders, info, last_loss, told = {}, {}, {}, {}, {}
        for s in sessions:
            if store.active():
                retriever.sync(store)
            t0 = time.monotonic()
            ing.ingest_two_step(s) if arm in TWO_STEP else ing.ingest_note(s)
            seconds[name].append(time.monotonic() - t0)
            if arm in TWO_STEP:
                for d in ing.decision_log:
                    R[name]["decisions"].append((s.get("operation") or "none", d["action"]))
            sid = s["session_id"]
            active = store.active()
            probes = [p for p in grading_view(true_at(entries, sid)) if p["grade"] == "code" and p["state"] == "true"]
            now, n_found = {}, 0
            for p in probes:
                pk = probe_key(p)
                info[pk] = p
                par = parent_entry(p, by_row)
                hold = set()
                for m in active:
                    c = (pk, m.text)
                    if c not in cache:
                        cache[c] = match_with_parent(p, m.text, par)
                    if cache[c]:
                        hold.add(m.id)
                now[pk] = hold
                n_found += bool(hold)
                if p["from_session"] == sid:
                    told[pk] = bool(hold)
                before = holders.get(pk)
                if before and not hold:
                    fates = []
                    for mid in before:
                        m = store.get(mid)
                        fates.append("forgotten" if m.retired == "forgotten" else "superseded" if m.retired == "superseded" else "changed")
                    srec = rec_of_session.get(sid)
                    last_loss[pk] = {"session": sid, "fate": "superseded" if "superseded" in fates else fates[0],
                                     "where": "its own record" if srec == p["record"] else "no record" if srec is None else "another record",
                                     "before": short(store.get(next(iter(before))).text)}
            R[name]["per_session"].append((len(probes), n_found))
            holders = now
            progress(f"{persona} {name} session {sid}: active={len(active)} probes={len(probes)} found={n_found} ({seconds[name][-1]:.1f} s)")
            if sid != last:
                continue
            for pk, p in info.items():
                if pk not in now:
                    continue
                found = bool(now[pk])
                hit = next((store.get(mid).text for mid in now[pk]), "")
                R[name]["end"].append({"persona": persona, "probe": pk, "kind": p["kind"], "type": rtype(p["record"]),
                                       "detail": p["detail"], "found": found, "told": told.get(pk),
                                       "qualifier": qualifier_kept(p, hit) if hit else (False if p.get("qualifier") else None),
                                       "loss": last_loss.get(pk) if not found and told.get(pk) else None})
            for p in grading_view([e for e in entries if e["state"] == "true" and e["until_session"] is not None]):
                if p["grade"] != "code":
                    continue
                hits = [m for m in active if match_with_parent(p, m.text, parent_entry(p, by_row))]
                cls = ("not found" if not hits else "says it changed" if HISTORY.search(hits[0].text)
                       else "never superseded" if hits[0].created_at.session_id < p["until_session"] else "carried")
                R[name]["old"].append({"type": rtype(p["record"]), "class": cls, "detail": p["detail"],
                                       "memory": short(hits[0].text) if hits else ""})
        calls[name].update(("fresh" if not c["cached"] else "cached") for c in ing.calls)
elapsed = time.monotonic() - start
names = [n for n, _, _ in ARMS]

if PILOT:
    print("PILOT RUN: not evidence")
    for n in names:
        print(f"  {n}: {len(seconds[n])} sessions in {sum(seconds[n]):.0f} s; calls {dict(calls[n])}; 476 sessions would take about "
              f"{sum(seconds[n]) / max(1, len(seconds[n])) * 476 / 60:.0f} min at this rate")
    print()
print(f"development personas: {', '.join(personas)}; wall time {elapsed:.0f} s; calls " + "; ".join(f"{n} {dict(calls[n])}" for n in names) + "\n")

print("1. RECALL AT THE END (probes that should hold, found in some active memory)")
for n in names:
    end = R[n]["end"]
    print(f"  {n:<9} {pct(sum(r['found'] for r in end), len(end))} of {len(end)}   " + ", ".join(
        f"{k} {pct(sum(r['found'] for r in end if r['kind'] == k), c).strip()}" for k, c in Counter(r["kind"] for r in end).most_common()))
    print(f"  {'':<9} by record type: " + ", ".join(
        f"{k} {pct(sum(r['found'] for r in end if r['type'] == k), c).strip()} of {c}" for k, c in Counter(r["type"] for r in end).most_common()))
print()

print("2. OVER TIME, WHEN TOLD, AND LOST")
for n in names:
    per = [f / t for t, f in R[n]["per_session"] if t]
    k = len(per)
    thirds = [per[i * k // 3:(i + 1) * k // 3] for i in range(3)]
    end = R[n]["end"]
    told = [r for r in end if r["told"] is not None]
    kept_then = [r for r in told if r["told"]]
    print(f"  {n:<9} mean over sessions {sum(per) / max(1, k):.3f} (thirds {' / '.join(f'{sum(t) / max(1, len(t)):.3f}' for t in thirds)}); "
          f"found when told {pct(len(kept_then), len(told))}; lost by the end {sum(1 for r in kept_then if not r['found'])}")
print("  (thirds are of the sessions in order, across personas)")
print()

print("3. CAUSE OF EACH LOSS (found when told, not at the end: what happened in the session it was last lost)")
for n in names:
    lost = [r for r in R[n]["end"] if r["loss"]]
    print(f"  {n:<9} {len(lost)}: " + ", ".join(f"{f} in a session about {w} {c}" for (f, w), c in
                                               Counter((r['loss']['fate'], r['loss']['where']) for r in lost).most_common())
          + "; folded, by record type: " + ", ".join(f"{k} {c}" for k, c in Counter(
              r["type"] for r in lost if r["loss"]["fate"] == "superseded" and r["loss"]["where"] == "another record").most_common()))
print()

print("4. QUALIFIERS KEPT, AND OLD VALUES (held once, updated or removed by the end)")
for n in names:
    q = [r for r in R[n]["end"] if r["qualifier"] is not None]
    old = R[n]["old"]
    print(f"  {n:<9} qualifiers {sum(1 for r in q if r['qualifier'])} of {len(q)}; old values found {sum(1 for r in old if r['class'] != 'not found')} of {len(old)}: "
          + ", ".join(f"{k} {c}" for k, c in Counter(r["class"] for r in old if r["class"] != "not found").most_common()))
print()

print("5. PAIRED AT THE END (found by one arm and not the other; exact sign test; probes are not independent)")
ends = {n: {(r["persona"], r["probe"]): r["found"] for r in R[n]["end"]} for n in names}
common = set.intersection(*(set(v) for v in ends.values()))
for i, a in enumerate(names):
    for b in names[i + 1:]:
        x = sum(1 for k in common if ends[a][k] and not ends[b][k])
        y = sum(1 for k in common if ends[b][k] and not ends[a][k])
        print(f"  {a} vs {b}: only {a} {x}, only {b} {y}, p = {sign_test(x, y):.3g}")
print()

print("6. STEP 2'S DECISIONS (two-step arms), by the session's operation")
for n in names:
    ds = R[n]["decisions"]
    if not ds:
        continue
    print(f"  {n}: {len(ds)} facts; " + ", ".join(f"{k} {c}" for k, c in Counter(a for _, a in ds).most_common()))
    for op in ("add", "update", "delete", "none"):
        sub = [a for o, a in ds if o == op]
        if sub:
            print(f"      in {op} sessions: " + ", ".join(f"{k} {c}" for k, c in Counter(sub).most_common()))
print()

print("7. EXAMPLES (up to 6 each): lost by 2s3 at 0.0 but kept at the new threshold, and the reverse; old values never superseded at the new threshold")
t_name = names[2]
for x, y in ((t_name, names[1]), (names[1], t_name)):
    keys = [k for k in sorted(common) if ends[x][k] and not ends[y][k]][:6]
    det = {(r["persona"], r["probe"]): r for r in R[y]["end"]}
    print(f"  kept by {x}, not by {y}:")
    for k in keys:
        r = det[k]
        print(f"    - {k[0]} [{r['kind']}, {r['type']}] {short(r['detail'], 90)}" + (f" | lost in session {r['loss']['session']}: {r['loss']['before']}" if r["loss"] else ""))
print(f"  old values never superseded at {t_name}:")
for r in [r for r in R[t_name]["old"] if r["class"] == "never superseded"][:6]:
    print(f"    - [{r['type']}] {short(r['detail'], 60)} | {r['memory']}")
