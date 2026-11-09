"""E43: Stage 0's reported numbers (D20): probe survival, ingestion loss, and determinism with the cache bypassed.

Question (D20, no pass/fail): with the frozen setup (two-step v3 ingestion for the note arm, D52; `ingest_log_v1`
for the log arm; the code grader after D62, passed in E42), (1) how many probe facts survive ingestion, which
sets how many facts Stage 1 can compare; (2) how much the note arm loses against the log arm, which keeps the
user's own words; (3) how much a fresh run differs from the cached one.
Method: (1) and (2) read store B's snapshots from the frozen Stage 0 control run (E38; identical to ingestion
alone) and grade them with the frozen answer key (D50, D51) and grader (D62), as E32 does: each code-graded
probe is checked right after the session it was told in ("found when told"), and every probe holding at the end
is checked in the end-of-history store ("found at the end"); old values (replaced or removed by the end) too. Ingestion
loss compares the two arms on the same probes, paired, with a sign test. No model calls.
(3) re-ingests one development persona (MEMRECON_E43_FRESH, default financial_analyst; 0 to skip) in both arms
with every call made afresh (request seed 1 instead of 0, so no call is read from the cache; temperature stays
0) and compares it with the cached run (store A of E38) after every session, and at the end by probe recall.
Progress goes to results/e43_progress.log.
Limits: the code-graded probes only (numbers, dates, names, titles, items; tasks and preferences go to the judge);
one fresh run of one persona, so (3) shows that runs differ and by how much, not a distribution.
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
from memrecon.memora import dev_personas
from memrecon.memory import MemoryStore
from memrecon.monitor import wait_for_memory
from memrecon.probes import OVERLAP, Row, build_table, grading_view, match_with_parent, parent_entry, true_at
from memrecon.recall import REWRITE_PROMPT, SELECT_PROMPT
from memrecon.retrieval import Retriever
from memrecon.stage0 import ARMS, INGEST_K

CONTROL = ROOT / "results" / "stage0" / "control"
PROBES = ROOT / "results" / "probes"
PROGRESS = ROOT / "results" / "e43_progress.log"
FRESH = [p for p in (os.environ.get("MEMRECON_E43_FRESH", "financial_analyst") or "").split(",") if p and p != "0"]
KINDS = ("number", "date", "name", "title", "item")


def progress(msg):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


def sign_test(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load_rows(name, persona):
    rows = []
    for x in (PROBES / f"{name}_{persona}.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            d = json.loads(x)
            d.pop("persona", None)
            d["record"] = tuple(d["record"])
            rows.append(Row(**d))
    return rows


def control_config():
    for p in sorted(CONTROL.iterdir()):
        f = p / "config.json"
        if f.exists():
            c = json.loads(f.read_text())
            if c.get("sessions") == "all" and c.get("rewrite_prompt") == REWRITE_PROMPT and c.get("select_prompt") == SELECT_PROMPT \
                    and (p / "report.md").exists():
                return p
    raise SystemExit("No finished full control run with the current prompts.")


def key_of(p):
    return p.get("row_id") or f"{p['record']}|{p['field']}|{p['detail']}|{p['from_session']}"


def found_in(p, texts, by_row):
    parent = parent_entry(p, by_row)
    return any(match_with_parent(p, t, parent, OVERLAP) for t in texts)


cfg = control_config()
print(f"Stage 0 control run {cfg.name} (E38); answer key E30 + E31 (D50, D51); grader after D62")
start = time.monotonic()
key, sessions_of = {}, {}
for persona in dev_personas():
    rows = load_rows("E30", persona) + load_rows("E31", persona)
    sessions = [s for _, _, s in iter_sessions(persona)]
    with_rows = {r.session for r in rows}
    failed = {s["session_id"] for s in sessions if s.get("operation") in ("add", "update", "delete") and flagged_turns(s) and s["session_id"] not in with_rows}
    bad = set()
    for s in sessions:
        if s["session_id"] in failed:
            k = record_key(s)
            bad.add(k if k is not None else ("session", s["session_id"]))
    key[persona] = build_table([r for r in rows if r.record not in bad])
    sessions_of[persona] = sessions


def grade_run(persona, snap_dir, entries):
    """Per probe: found when told, found at the end; and old values found at the end."""
    by_row = {e["row_id"]: e for e in entries if e.get("row_id")}
    sessions = sessions_of[persona]
    last = sessions[-1]["session_id"]
    # the probes that should be in memory at the end (as E32): found right after the session they were told in,
    # and at the end; old values: values that held once and were replaced or removed by the end (as E32)
    end_probes = [p for p in grading_view(true_at(entries, last)) if p["grade"] == "code" and p["state"] == "true"]
    told = defaultdict(list)
    for p in end_probes:
        told[p["from_session"]].append(p)
    out = {}
    for sid in sorted(told):
        texts = [m.text for m in MemoryStore.load(snap_dir / f"session_{sid:03d}.json").active()]
        for p in told[sid]:
            out[key_of(p)] = {"kind": p["kind"], "detail": p["detail"], "when_told": found_in(p, texts, by_row), "end": None}
    end_texts = [m.text for m in MemoryStore.load(snap_dir / f"session_{last:03d}.json").active()]
    for p in end_probes:
        out[key_of(p)]["end"] = found_in(p, end_texts, by_row)
    gone = [p for p in grading_view([e for e in entries if e["state"] == "true" and e["until_session"] is not None]) if p["grade"] == "code"]
    old = [found_in(p, end_texts, by_row) for p in gone]
    return out, old, len(end_texts)


# -- (1) and (2): survival and ingestion loss ------------------------------------------------------
res = {arm: {} for arm in ("note", "log")}
olds = {arm: [] for arm in ("note", "log")}
live = Counter()
for persona in dev_personas():
    for arm in ("note", "log"):
        run = sorted((cfg / persona / arm).glob("run_*"), key=lambda p: int(p.name[4:]))[-1]
        r, old, n_live = grade_run(persona, run / "B" / "snapshots", key[persona])
        res[arm].update({(persona, k): v for k, v in r.items()})
        olds[arm] += old
        live[arm] += n_live
        progress(f"{persona} {arm}: graded")

print(f"\nAnswer key: {sum(len(e) for e in key.values())} entries over {len(key)} development personas; "
      f"code-graded probes that should be in memory: {len(res['note'])}")
print("\n1. PROBE SURVIVAL (code-graded probes, state true)")
for arm in ("note", "log"):
    r = res[arm]
    told = [v for v in r.values() if v["when_told"] is not None]
    end = [v for v in r.values() if v["end"] is not None]
    kept = [v for v in told if v["when_told"] and v["end"] is not None]
    print(f"  {arm:4}  found when told {sum(v['when_told'] for v in told):4} of {len(told)} {pct(sum(v['when_told'] for v in told), len(told))};"
          f"  found at the end {sum(v['end'] for v in end):4} of {len(end)} {pct(sum(v['end'] for v in end), len(end))};"
          f"  of those found when told, kept to the end {sum(v['end'] for v in kept)} of {len(kept)} {pct(sum(v['end'] for v in kept), len(kept))};"
          f"  old values found at the end {sum(olds[arm])} of {len(olds[arm])}; live memories at the end {live[arm]}")
    for kind in KINDS:
        e = [v for v in end if v["kind"] == kind]
        t = [v for v in told if v["kind"] == kind]
        print(f"        {kind:6} when told {pct(sum(v['when_told'] for v in t), len(t))} of {len(t):3};  at the end {pct(sum(v['end'] for v in e), len(e))} of {len(e):3}")

print("\n2. INGESTION LOSS: the note arm against the log arm (the user's own words), same probes, paired")
for when in ("when_told", "end"):
    pairs = [(res["note"][k][when], res["log"][k][when]) for k in res["note"] if k in res["log"]
             and res["note"][k][when] is not None and res["log"][k][when] is not None]
    only_log = sum(1 for n, l in pairs if l and not n)
    only_note = sum(1 for n, l in pairs if n and not l)
    print(f"  {'found when told' if when == 'when_told' else 'found at the end':16}  note {pct(sum(n for n, _ in pairs), len(pairs))}, log {pct(sum(l for _, l in pairs), len(pairs))} of {len(pairs)};"
          f"  only log {only_log}, only note {only_note}, sign test p = {sign_test(only_log, only_note):.3g}")
    for kind in KINDS:
        ks = [(res["note"][k][when], res["log"][k][when]) for k in res["note"] if k in res["log"] and res["note"][k]["kind"] == kind
              and res["note"][k][when] is not None and res["log"][k][when] is not None]
        print(f"        {kind:6} note {pct(sum(n for n, _ in ks), len(ks))}, log {pct(sum(l for _, l in ks), len(ks))} of {len(ks):3};"
              f"  only log {sum(1 for n, l in ks if l and not n)}, only note {sum(1 for n, l in ks if n and not l)}")
lost = [(k, v) for k, v in res["note"].items() if v["end"] is not None and not v["end"] and res["log"].get(k, {}).get("end")]
print(f"  Examples, found at the end in the log arm but not the note arm ({len(lost)}; up to 12):")
for (persona, _), v in lost[:12]:
    print(f"    - {persona} [{v['kind']}] {v['detail'][:110]}")

print("\n   Probes available to Stage 1 (should be in memory at the end and found there, so a rewrite can change them):")
for arm in ("note", "log"):
    print(f"    {arm}: {sum(1 for v in res[arm].values() if v['end'])}")
both = sum(1 for k, v in res["note"].items() if v["end"] and res["log"].get(k, {}).get("end"))
print(f"    in both arms: {both}  (CLAUDE.md: about 500 probed facts detect roughly a 6-point paired difference)")
print(f"\n(1) and (2) in {time.monotonic() - start:.0f} s")

# -- (3) determinism with the cache bypassed ---------------------------------------------------------
if FRESH:
    print("\n3. DETERMINISM: one persona re-ingested with every call made afresh (request seed 1), against the cached run")


    class Fresh:
        """The agent client, with seed 1 on every call, so none is read from the cache (temperature stays 0)."""

        def __init__(self, inner):
            self.inner = inner

        def chat(self, messages, **kw):
            kw.setdefault("seed", 1)
            return self.inner.chat(messages, **kw)


    agent, embedder = Fresh(model_client("agent")), Embedder()
    for persona in FRESH:
        for arm in ("note", "log"):
            spec = ARMS[arm]
            run = sorted((cfg / persona / arm).glob("run_*"), key=lambda p: int(p.name[4:]))[-1]
            store = MemoryStore(persona, mode=spec["mode"])
            kwargs = {k: v for k, v in spec.items() if k not in ("mode", "method")}
            ing = Ingestor(agent, store, Retriever(embedder, k=INGEST_K), persona=persona, tag="E43-fresh", **kwargs)
            t0, first_diff, n_equal, fresh_dir = time.monotonic(), None, 0, ROOT / "results" / "e43" / persona / arm
            fresh_dir.mkdir(parents=True, exist_ok=True)
            for s in sessions_of[persona]:
                wait_for_memory(say=progress)
                sid = s["session_id"]
                if store.active():
                    ing.retriever.sync(store)
                getattr(ing, spec["method"])(s)
                store.save(fresh_dir / f"session_{sid:03d}.json")
                same = store.content_json() == MemoryStore.load(run / "A" / "snapshots" / f"session_{sid:03d}.json").content_json()
                n_equal += same
                if not same and first_diff is None:
                    first_diff = sid
                progress(f"fresh {persona} {arm} session {sid}: same as cached {same}")
            calls = Counter("cached" if c.get("cached") else "fresh" for c in ing.calls)
            r, old, n_live = grade_run(persona, fresh_dir, key[persona])
            cached = {k[1]: v for k, v in res[arm].items() if k[0] == persona}
            pairs = [(cached[k]["end"], r[k]["end"]) for k in r if k in cached and r[k]["end"] is not None]
            told = [(cached[k]["when_told"], r[k]["when_told"]) for k in r if k in cached and r[k]["when_told"] is not None]
            print(f"  {persona} {arm}: {len(sessions_of[persona])} sessions in {time.monotonic() - t0:.0f} s; calls {dict(calls)}")
            print(f"    memory identical to the cached run after {n_equal} of {len(sessions_of[persona])} sessions; first difference after session {first_diff}")
            print(f"    found when told: cached {pct(sum(a for a, _ in told), len(told))}, fresh {pct(sum(b for _, b in told), len(told))};"
                  f"  differing probes {sum(a != b for a, b in told)} of {len(told)}")
            print(f"    found at the end: cached {pct(sum(a for a, _ in pairs), len(pairs))}, fresh {pct(sum(b for _, b in pairs), len(pairs))};"
                  f"  differing probes {sum(a != b for a, b in pairs)} of {len(pairs)} (only cached {sum(1 for a, b in pairs if a and not b)}, only fresh {sum(1 for a, b in pairs if b and not a)});"
                  f"  live memories at the end {n_live}")
else:
    print("\n3. DETERMINISM: skipped (MEMRECON_E43_FRESH=0)")
print(f"\nTotal {time.monotonic() - start:.0f} s")
